import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torchvision.utils import save_image
from einops import rearrange
import torch.nn.functional as F

from skimage.morphology import binary_erosion


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    img, source: torch.Tensor, patch_size: int = 16, num_groups: int=196
):
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """
    sw = [[[[1,1],[1,1]]]]
    sw = np.array(sw)
    B, _, h, w = img.shape
    ph = h // patch_size
    pw = w // patch_size
    vis_img = 0
    cmap = generate_colormap(num_groups)
    temp = torch.ones(B, 3, device='cuda')
    for i in range(num_groups):
        mask = (source == i).float().view(B, 1, 1, ph, pw)
        mask = F.interpolate(mask, size=(1, h, w), mode="nearest")
        mask = mask.view(B, 1, h, w)
        color = (mask * img).sum(axis=(2, 3)) /( mask.sum(axis=(2, 3)) + temp)
        mask = mask.cpu().numpy()
        mask_eroded = binary_erosion(mask, sw)
        mask_edge = mask - mask_eroded
        mask_eroded = torch.as_tensor(mask_eroded, device='cuda')
        mask_edge = torch.as_tensor(mask_edge, device='cuda')
        vis_img = vis_img + mask_eroded * color.reshape(B, 3, 1, 1)
        cmapi = np.array(cmap[i]).reshape(3, 1, 1)
        cmapi = torch.as_tensor(cmapi, device='cuda')
        cmapi = cmapi.unsqueeze(0).expand(B, 3, 1, 1)
        vis_img = vis_img + mask_edge * cmapi
    return vis_img

def getCluster(idx, nidx, cos):
    res = []
    cos0 = 0
    cos01= 0
    cos1 =0
    nidxi = 0
    for i in range(len(idx)):
      output = torch.zeros(idx[0].size(0), 196, dtype=int, device='cuda')

      if i == 0:
          output = torch.scatter(output, dim=1, index=idx[i], src=idx[i])
          cosi = torch.gather(idx[i], dim=1, index=cos[i])
          output = torch.scatter(output, dim=1, index=nidx[i], src=cosi)
          cos0 = cosi
      elif i == 1:
          output = torch.scatter(output, dim=1, index=idx[i], src=idx[i])
          nidxi = torch.gather(idx[i-1], dim=1, index=nidx[i])
          cosi = torch.gather(idx[i], dim=1, index=cos[i])
          output = torch.scatter(output, dim=1, index=nidxi, src=cosi)
          temp = torch.gather(output, dim=1, index=cos0)
          cos01 = temp
          cos1 = cosi
          nidx1 = nidxi
          output = torch.scatter(output, dim=1, index=nidx[i-1], src=temp)

      elif i == 2:
          output = torch.scatter(output, dim=1, index=idx[i], src=idx[i])
          nidxi = torch.gather(idx[i - 1], dim=1, index=nidx[i])
          cosi = torch.gather(idx[i], dim=1, index=cos[i])
          output = torch.scatter(output, dim=1, index=nidxi, src=cosi)
          temp1 = torch.gather(output, dim=1, index=cos01)
          temp2 = torch.gather(output, dim=1, index=cos1)
          output = torch.scatter(output, dim=1, index=nidx[i - 2], src=temp1)
          output = torch.scatter(output, dim=1, index=nidx1, src=temp2)


      res.append(output)

    return res

def mask(x, idx, nidx, cos, cls, idx_, nidx_, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)
    output = torch.zeros_like(x)
    icls = torch.gather(cls, dim=1, index=idx_)
    ucls = torch.gather(cls, dim=1, index=nidx_)
    acls = icls.scatter_add(dim=1, index=cos, src=ucls)
    for i in range(60):
        print(cos[i])
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)
    idx2 = nidx.unsqueeze(1).expand(-1, x.size(1), -1)
    cos = cos.unsqueeze(1).expand(-1, x.size(1), -1)
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T]
    extracted = extracted * icls.unsqueeze(1).expand(x.size(0), x.size(1), -1)
    nextracted = torch.gather(x, dim=2, index=idx2)
    nextracted = nextracted * ucls.unsqueeze(1).expand(x.size(0), x.size(1), -1)
    extracted = extracted.scatter_add(dim=2, index=cos, src=nextracted)
    extracted = extracted / acls.unsqueeze(1).expand(x.size(0), x.size(1), -1)
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted)
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    return out_img


def get_deeper_idx(idx1, idx2):
    """
    Args:
        idx1: indices, shape: [B, T1]
        idx2: indices to gather from idx1, shape: [B, T2], T2 <= T1
    """
    return torch.gather(idx1, dim=1, index=idx2)


def get_real_idx(idxs):
    # nh = img_size // patch_size
    # npatch = nh ** 2

    # gather real idx
    for i in range(1, len(idxs)):
        tmp = idxs[i - 1]
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
    return idxs


def save_img_batch(x, path, file_name='img{}', start_idx=0):
    for i, img in enumerate(x):
        save_image(img, os.path.join(path, file_name.format(start_idx + i)))
