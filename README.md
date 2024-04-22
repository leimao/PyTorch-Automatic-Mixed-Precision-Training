# PyTorch Automatic Mixed Precision Training

## Introduction

## Usages

### Build Docker Image

```bash
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:2.2.0 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/mnt pytorch:2.2.0
```

### Run ResNet50 CIFAR10 Training

#### FP32 Training

```bash
$ python train.py --model_dir saved_models --model_filename resnet50_cifar10_fp32.pt
Training Model...
Training Elapsed Time: 00:17:13
Evaluating Model...
Test Accuracy: 0.876
```

#### AMP Training

```bash
$ python train.py --model_dir saved_models --model_filename resnet50_cifar10_amp.pt --use_amp
Training Model...
Training Elapsed Time: 00:12:29
Evaluating Model...
Test Accuracy: 0.887
```

## References

- [PyTorch Automatic Mixed Precision Training](https://leimao.github.io/blog/PyTorch-Automatic-Mixed-Precision-Training/)
- [PyTorch Automatic Mixed Precision Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [PyTorch AMP Package](https://pytorch.org/docs/stable/amp.html)
