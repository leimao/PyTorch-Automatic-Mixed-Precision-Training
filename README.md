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
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.90      0.89      1000
           1       0.93      0.93      0.93      1000
           2       0.85      0.84      0.85      1000
           3       0.77      0.74      0.75      1000
           4       0.86      0.88      0.87      1000
           5       0.80      0.80      0.80      1000
           6       0.90      0.92      0.91      1000
           7       0.93      0.89      0.91      1000
           8       0.93      0.93      0.93      1000
           9       0.90      0.92      0.91      1000

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000
```

#### AMP Training

```bash
$ python train.py --model_dir saved_models --model_filename resnet50_cifar10_amp.pt --use_amp
Training Model...
Training Elapsed Time: 00:12:29
Evaluating Model...
Test Accuracy: 0.887
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.92      0.91      1000
           1       0.95      0.94      0.94      1000
           2       0.88      0.86      0.87      1000
           3       0.76      0.77      0.76      1000
           4       0.89      0.87      0.88      1000
           5       0.80      0.82      0.81      1000
           6       0.94      0.92      0.93      1000
           7       0.93      0.92      0.93      1000
           8       0.94      0.93      0.93      1000
           9       0.91      0.93      0.91      1000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

## References

- [PyTorch Automatic Mixed Precision Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [PyTorch AMP Package](https://pytorch.org/docs/stable/amp.html)
