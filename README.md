
# Preparation
The reported results in the paper were obtained with models trained using Python3.9 and the following packages
```
torch==1.9.0
torchvision==0.10.0
timm==0.4.12
tensorboardX==2.4
torchprofile==0.0.4
pyarrow==5.0.0
einops==0.4.1
fvcore==0.1.5
scikit-image==0.19.2  
```
These packages can be installed by running `pip install -r requirements.txt`.

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision datasets.ImageFolder, and the training and validation data is expected to be in the train/ folder and val folder respectively.

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
      
```


## Training

To train MTViT  on ImageNet on a single node with 8 gpus for 300 epochs,  run:

Set `--base_keep_rate`  to use a different keep rate, and set `--muti-scale` to configure whether to use multi-scale features.

MTViT-T

```
python3  -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_shrink_base  --base_keep_rate 0.7 --input-size 224 --batch-size 256 --warmup-epochs 5 --shrink_start_epoch 10 --shrink_epochs 100 --epochs 300 --dist-eval  --data-path /path/to/imagenet --output_dir /out/to/dir --muti-scale
```

MTViT-S

```
python3  -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_shrink_base  --base_keep_rate 0.7 --input-size 224 --batch-size 128 --warmup-epochs 5 --shrink_start_epoch 10 --shrink_epochs 100 --epochs 300 --dist-eval  --data-path /path/to/imagenet --output_dir /out/to/dir --muti-scale
```

MTViT-B(The features obtained before token pruning are already sufficiently rich, so multi-scale features are not used.)

```
python3  -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_shrink_base  --base_keep_rate 0.7 --input-size 224 --batch-size 128 --warmup-epochs 5 --shrink_start_epoch 10 --shrink_epochs 100 --epochs 300 --dist-eval  --data-path /path/to/imagenet --output_dir /out/to/dir
```



## Finetuning
```
python3  -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_shrink_base  --base_keep_rate 0.7 --input-size 224 --sched cosine --lr 2e-5 --min-lr 1e-6 --weight-decay 1e-6 --batch-size 256 --shrink_start_epoch 0 --warmup-epochs 0 --shrink_epochs 0  --epochs 30 --dist-eval --finetune /checkpoint.pth  --data-path /path/to/imagenet   --output_dir /out/to/dir
```



## Evaluation

```
python3 main.py --model deit_small_patch16_shrink_base  --base_keep_rate 0.7 --batch-size 256 --eval  --resume /checkpoint.pth  --data-path /path/to/imagenet 
```


## Throughput&Computation
You can measure the throughput of the model by passing `--test_speed` or `--only_test_speed` to `main.py`.  And you can measure the throughput of the model by passing  `--flops` to main.py.



## Visualization
You can visualize the masked image by a command like this:
```
python3 main.py --model deit_small_patch16_shrink_base --base_keep_rate 0.5 --visualize_mask --n_visualization 64 --resume checkpoint --data-path /path/to/imagenet
```



# Acknowledgement

We would like to thank the authors of [DeiT](https://github.com/facebookresearch/deit) , [timm](https://github.com/rwightman/pytorch-image-models) and [EViT](https://github.com/youweiliang/evit), based on which this codebase was built.

