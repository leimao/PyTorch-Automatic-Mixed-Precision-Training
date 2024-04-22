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
Epoch: 000 Eval Loss: 6.387 Eval Acc: 0.099
Epoch: 001 Train Loss: 7.443 Train Acc: 0.102 Eval Loss: 2.632 Eval Acc: 0.096 Elapsed Time: 4640ms
Epoch: 002 Train Loss: 3.303 Train Acc: 0.131 Eval Loss: 27.886 Eval Acc: 0.123 Elapsed Time: 4451ms
Epoch: 003 Train Loss: 2.653 Train Acc: 0.171 Eval Loss: 2.093 Eval Acc: 0.192 Elapsed Time: 4401ms
Epoch: 004 Train Loss: 2.215 Train Acc: 0.228 Eval Loss: 1.910 Eval Acc: 0.268 Elapsed Time: 4484ms
Epoch: 005 Train Loss: 2.013 Train Acc: 0.286 Eval Loss: 2.762 Eval Acc: 0.229 Elapsed Time: 4501ms
Epoch: 006 Train Loss: 1.908 Train Acc: 0.315 Eval Loss: 1.708 Eval Acc: 0.364 Elapsed Time: 4524ms
Epoch: 007 Train Loss: 1.793 Train Acc: 0.349 Eval Loss: 1.697 Eval Acc: 0.377 Elapsed Time: 4505ms
Epoch: 008 Train Loss: 1.748 Train Acc: 0.371 Eval Loss: 1.636 Eval Acc: 0.390 Elapsed Time: 4473ms
Epoch: 009 Train Loss: 1.671 Train Acc: 0.394 Eval Loss: 1.515 Eval Acc: 0.440 Elapsed Time: 4418ms
Epoch: 010 Train Loss: 1.607 Train Acc: 0.420 Eval Loss: 1.608 Eval Acc: 0.420 Elapsed Time: 4439ms
Epoch: 011 Train Loss: 1.544 Train Acc: 0.439 Eval Loss: 1.483 Eval Acc: 0.472 Elapsed Time: 4416ms
Epoch: 012 Train Loss: 1.499 Train Acc: 0.459 Eval Loss: 1.434 Eval Acc: 0.482 Elapsed Time: 4527ms
Epoch: 013 Train Loss: 1.456 Train Acc: 0.469 Eval Loss: 1.382 Eval Acc: 0.497 Elapsed Time: 4511ms
Epoch: 014 Train Loss: 1.417 Train Acc: 0.483 Eval Loss: 1.324 Eval Acc: 0.521 Elapsed Time: 4435ms
Epoch: 015 Train Loss: 1.375 Train Acc: 0.500 Eval Loss: 1.339 Eval Acc: 0.521 Elapsed Time: 4441ms
Epoch: 016 Train Loss: 1.341 Train Acc: 0.514 Eval Loss: 1.276 Eval Acc: 0.544 Elapsed Time: 4425ms
Epoch: 017 Train Loss: 1.297 Train Acc: 0.530 Eval Loss: 1.244 Eval Acc: 0.555 Elapsed Time: 4492ms
Epoch: 018 Train Loss: 1.257 Train Acc: 0.545 Eval Loss: 1.217 Eval Acc: 0.566 Elapsed Time: 4422ms
Epoch: 019 Train Loss: 1.222 Train Acc: 0.560 Eval Loss: 1.199 Eval Acc: 0.577 Elapsed Time: 4441ms
Epoch: 020 Train Loss: 1.185 Train Acc: 0.573 Eval Loss: 1.190 Eval Acc: 0.578 Elapsed Time: 4459ms
Epoch: 021 Train Loss: 1.145 Train Acc: 0.592 Eval Loss: 1.154 Eval Acc: 0.595 Elapsed Time: 4584ms
Epoch: 022 Train Loss: 1.111 Train Acc: 0.600 Eval Loss: 1.082 Eval Acc: 0.621 Elapsed Time: 4468ms
Epoch: 023 Train Loss: 1.085 Train Acc: 0.615 Eval Loss: 1.065 Eval Acc: 0.620 Elapsed Time: 4431ms
Epoch: 024 Train Loss: 1.049 Train Acc: 0.627 Eval Loss: 1.060 Eval Acc: 0.628 Elapsed Time: 4543ms
Epoch: 025 Train Loss: 1.023 Train Acc: 0.636 Eval Loss: 1.029 Eval Acc: 0.635 Elapsed Time: 4617ms
Epoch: 026 Train Loss: 0.981 Train Acc: 0.652 Eval Loss: 1.068 Eval Acc: 0.630 Elapsed Time: 4449ms
Epoch: 027 Train Loss: 0.957 Train Acc: 0.660 Eval Loss: 0.956 Eval Acc: 0.667 Elapsed Time: 4460ms
Epoch: 028 Train Loss: 0.931 Train Acc: 0.669 Eval Loss: 1.041 Eval Acc: 0.649 Elapsed Time: 4464ms
Epoch: 029 Train Loss: 0.906 Train Acc: 0.679 Eval Loss: 0.991 Eval Acc: 0.659 Elapsed Time: 4434ms
Epoch: 030 Train Loss: 0.880 Train Acc: 0.688 Eval Loss: 0.894 Eval Acc: 0.689 Elapsed Time: 4427ms
Epoch: 031 Train Loss: 0.841 Train Acc: 0.703 Eval Loss: 0.845 Eval Acc: 0.710 Elapsed Time: 4438ms
Epoch: 032 Train Loss: 0.827 Train Acc: 0.708 Eval Loss: 0.880 Eval Acc: 0.694 Elapsed Time: 4411ms
Epoch: 033 Train Loss: 0.799 Train Acc: 0.718 Eval Loss: 0.843 Eval Acc: 0.709 Elapsed Time: 4444ms
Epoch: 034 Train Loss: 0.786 Train Acc: 0.724 Eval Loss: 0.855 Eval Acc: 0.708 Elapsed Time: 4444ms
Epoch: 035 Train Loss: 0.767 Train Acc: 0.731 Eval Loss: 0.824 Eval Acc: 0.715 Elapsed Time: 4541ms
Epoch: 036 Train Loss: 0.753 Train Acc: 0.735 Eval Loss: 0.790 Eval Acc: 0.730 Elapsed Time: 4504ms
Epoch: 037 Train Loss: 0.735 Train Acc: 0.740 Eval Loss: 0.746 Eval Acc: 0.744 Elapsed Time: 4460ms
Epoch: 038 Train Loss: 0.739 Train Acc: 0.741 Eval Loss: 0.777 Eval Acc: 0.730 Elapsed Time: 4445ms
Epoch: 039 Train Loss: 0.711 Train Acc: 0.751 Eval Loss: 0.763 Eval Acc: 0.738 Elapsed Time: 4437ms
Epoch: 040 Train Loss: 0.694 Train Acc: 0.758 Eval Loss: 0.809 Eval Acc: 0.723 Elapsed Time: 4495ms
Epoch: 041 Train Loss: 0.679 Train Acc: 0.762 Eval Loss: 0.739 Eval Acc: 0.747 Elapsed Time: 4420ms
Epoch: 042 Train Loss: 0.663 Train Acc: 0.769 Eval Loss: 0.739 Eval Acc: 0.742 Elapsed Time: 4445ms
Epoch: 043 Train Loss: 0.654 Train Acc: 0.770 Eval Loss: 0.741 Eval Acc: 0.748 Elapsed Time: 4418ms
Epoch: 044 Train Loss: 0.645 Train Acc: 0.774 Eval Loss: 0.681 Eval Acc: 0.762 Elapsed Time: 4452ms
Epoch: 045 Train Loss: 0.652 Train Acc: 0.772 Eval Loss: 0.734 Eval Acc: 0.749 Elapsed Time: 4415ms
Epoch: 046 Train Loss: 0.643 Train Acc: 0.776 Eval Loss: 0.985 Eval Acc: 0.690 Elapsed Time: 4412ms
Epoch: 047 Train Loss: 0.631 Train Acc: 0.780 Eval Loss: 0.734 Eval Acc: 0.746 Elapsed Time: 4424ms
Epoch: 048 Train Loss: 0.620 Train Acc: 0.785 Eval Loss: 0.696 Eval Acc: 0.761 Elapsed Time: 4432ms
Epoch: 049 Train Loss: 0.599 Train Acc: 0.791 Eval Loss: 0.710 Eval Acc: 0.758 Elapsed Time: 4452ms
Epoch: 050 Train Loss: 0.590 Train Acc: 0.792 Eval Loss: 0.696 Eval Acc: 0.765 Elapsed Time: 4417ms
Epoch: 051 Train Loss: 0.581 Train Acc: 0.797 Eval Loss: 0.855 Eval Acc: 0.718 Elapsed Time: 4540ms
Epoch: 052 Train Loss: 0.578 Train Acc: 0.799 Eval Loss: 0.658 Eval Acc: 0.773 Elapsed Time: 4449ms
Epoch: 053 Train Loss: 0.565 Train Acc: 0.804 Eval Loss: 0.680 Eval Acc: 0.773 Elapsed Time: 4469ms
Epoch: 054 Train Loss: 0.556 Train Acc: 0.806 Eval Loss: 0.696 Eval Acc: 0.766 Elapsed Time: 4586ms
Epoch: 055 Train Loss: 0.552 Train Acc: 0.809 Eval Loss: 0.712 Eval Acc: 0.766 Elapsed Time: 4493ms
Epoch: 056 Train Loss: 0.547 Train Acc: 0.809 Eval Loss: 0.683 Eval Acc: 0.767 Elapsed Time: 4553ms
Epoch: 057 Train Loss: 0.537 Train Acc: 0.812 Eval Loss: 0.673 Eval Acc: 0.773 Elapsed Time: 4580ms
Epoch: 058 Train Loss: 0.540 Train Acc: 0.811 Eval Loss: 0.685 Eval Acc: 0.769 Elapsed Time: 4477ms
Epoch: 059 Train Loss: 0.531 Train Acc: 0.816 Eval Loss: 0.649 Eval Acc: 0.780 Elapsed Time: 4545ms
Epoch: 060 Train Loss: 0.520 Train Acc: 0.819 Eval Loss: 0.596 Eval Acc: 0.797 Elapsed Time: 4628ms
Epoch: 061 Train Loss: 0.521 Train Acc: 0.819 Eval Loss: 0.641 Eval Acc: 0.784 Elapsed Time: 4577ms
Epoch: 062 Train Loss: 0.519 Train Acc: 0.819 Eval Loss: 0.782 Eval Acc: 0.744 Elapsed Time: 4470ms
Epoch: 063 Train Loss: 0.506 Train Acc: 0.824 Eval Loss: 0.642 Eval Acc: 0.785 Elapsed Time: 4467ms
Epoch: 064 Train Loss: 0.504 Train Acc: 0.825 Eval Loss: 0.672 Eval Acc: 0.778 Elapsed Time: 4711ms
Epoch: 065 Train Loss: 0.502 Train Acc: 0.824 Eval Loss: 0.666 Eval Acc: 0.773 Elapsed Time: 4481ms
Epoch: 066 Train Loss: 0.488 Train Acc: 0.832 Eval Loss: 0.671 Eval Acc: 0.780 Elapsed Time: 4528ms
Epoch: 067 Train Loss: 0.486 Train Acc: 0.833 Eval Loss: 0.626 Eval Acc: 0.787 Elapsed Time: 4594ms
Epoch: 068 Train Loss: 0.491 Train Acc: 0.828 Eval Loss: 0.713 Eval Acc: 0.764 Elapsed Time: 4641ms
Epoch: 069 Train Loss: 0.481 Train Acc: 0.833 Eval Loss: 0.618 Eval Acc: 0.791 Elapsed Time: 4523ms
Epoch: 070 Train Loss: 0.479 Train Acc: 0.833 Eval Loss: 0.638 Eval Acc: 0.786 Elapsed Time: 4494ms
Epoch: 071 Train Loss: 0.472 Train Acc: 0.836 Eval Loss: 0.595 Eval Acc: 0.802 Elapsed Time: 4545ms
Epoch: 072 Train Loss: 0.465 Train Acc: 0.838 Eval Loss: 0.643 Eval Acc: 0.795 Elapsed Time: 4551ms
Epoch: 073 Train Loss: 0.468 Train Acc: 0.838 Eval Loss: 0.639 Eval Acc: 0.790 Elapsed Time: 4488ms
Epoch: 074 Train Loss: 0.458 Train Acc: 0.840 Eval Loss: 0.633 Eval Acc: 0.786 Elapsed Time: 4494ms
Epoch: 075 Train Loss: 0.456 Train Acc: 0.841 Eval Loss: 0.570 Eval Acc: 0.812 Elapsed Time: 4451ms
Epoch: 076 Train Loss: 0.461 Train Acc: 0.840 Eval Loss: 0.618 Eval Acc: 0.795 Elapsed Time: 4490ms
Epoch: 077 Train Loss: 0.454 Train Acc: 0.841 Eval Loss: 0.731 Eval Acc: 0.767 Elapsed Time: 4508ms
Epoch: 078 Train Loss: 0.447 Train Acc: 0.846 Eval Loss: 0.685 Eval Acc: 0.771 Elapsed Time: 4438ms
Epoch: 079 Train Loss: 0.451 Train Acc: 0.842 Eval Loss: 0.592 Eval Acc: 0.804 Elapsed Time: 4431ms
Epoch: 080 Train Loss: 0.438 Train Acc: 0.847 Eval Loss: 0.682 Eval Acc: 0.776 Elapsed Time: 4430ms
Epoch: 081 Train Loss: 0.449 Train Acc: 0.844 Eval Loss: 0.610 Eval Acc: 0.796 Elapsed Time: 4453ms
Epoch: 082 Train Loss: 0.446 Train Acc: 0.845 Eval Loss: 0.653 Eval Acc: 0.788 Elapsed Time: 4444ms
Epoch: 083 Train Loss: 0.441 Train Acc: 0.847 Eval Loss: 0.647 Eval Acc: 0.790 Elapsed Time: 4462ms
Epoch: 084 Train Loss: 0.429 Train Acc: 0.852 Eval Loss: 0.580 Eval Acc: 0.813 Elapsed Time: 4460ms
Epoch: 085 Train Loss: 0.431 Train Acc: 0.850 Eval Loss: 0.564 Eval Acc: 0.814 Elapsed Time: 4444ms
Epoch: 086 Train Loss: 0.423 Train Acc: 0.853 Eval Loss: 0.632 Eval Acc: 0.794 Elapsed Time: 4442ms
Epoch: 087 Train Loss: 0.421 Train Acc: 0.854 Eval Loss: 0.613 Eval Acc: 0.802 Elapsed Time: 4457ms
Epoch: 088 Train Loss: 0.421 Train Acc: 0.854 Eval Loss: 0.634 Eval Acc: 0.789 Elapsed Time: 4493ms
Epoch: 089 Train Loss: 0.427 Train Acc: 0.853 Eval Loss: 0.557 Eval Acc: 0.817 Elapsed Time: 4469ms
Epoch: 090 Train Loss: 0.408 Train Acc: 0.859 Eval Loss: 0.625 Eval Acc: 0.793 Elapsed Time: 4440ms
Epoch: 091 Train Loss: 0.422 Train Acc: 0.853 Eval Loss: 0.769 Eval Acc: 0.760 Elapsed Time: 4495ms
Epoch: 092 Train Loss: 0.408 Train Acc: 0.857 Eval Loss: 0.638 Eval Acc: 0.793 Elapsed Time: 4456ms
Epoch: 093 Train Loss: 0.409 Train Acc: 0.858 Eval Loss: 0.706 Eval Acc: 0.782 Elapsed Time: 4479ms
Epoch: 094 Train Loss: 0.414 Train Acc: 0.857 Eval Loss: 0.588 Eval Acc: 0.811 Elapsed Time: 4429ms
Epoch: 095 Train Loss: 0.410 Train Acc: 0.857 Eval Loss: 0.617 Eval Acc: 0.798 Elapsed Time: 4475ms
Epoch: 096 Train Loss: 0.395 Train Acc: 0.863 Eval Loss: 0.716 Eval Acc: 0.770 Elapsed Time: 4486ms
Epoch: 097 Train Loss: 0.402 Train Acc: 0.859 Eval Loss: 0.557 Eval Acc: 0.817 Elapsed Time: 4524ms
Epoch: 098 Train Loss: 0.402 Train Acc: 0.860 Eval Loss: 0.629 Eval Acc: 0.802 Elapsed Time: 4463ms
Epoch: 099 Train Loss: 0.400 Train Acc: 0.861 Eval Loss: 0.647 Eval Acc: 0.787 Elapsed Time: 4471ms
Epoch: 100 Train Loss: 0.394 Train Acc: 0.864 Eval Loss: 0.573 Eval Acc: 0.814 Elapsed Time: 4439ms
Epoch: 101 Train Loss: 0.278 Train Acc: 0.904 Eval Loss: 0.409 Eval Acc: 0.864 Elapsed Time: 4429ms
Epoch: 102 Train Loss: 0.220 Train Acc: 0.925 Eval Loss: 0.413 Eval Acc: 0.864 Elapsed Time: 4452ms
Epoch: 103 Train Loss: 0.203 Train Acc: 0.931 Eval Loss: 0.413 Eval Acc: 0.869 Elapsed Time: 4469ms
Epoch: 104 Train Loss: 0.191 Train Acc: 0.934 Eval Loss: 0.413 Eval Acc: 0.872 Elapsed Time: 4454ms
Epoch: 105 Train Loss: 0.179 Train Acc: 0.940 Eval Loss: 0.406 Eval Acc: 0.871 Elapsed Time: 4446ms
Epoch: 106 Train Loss: 0.169 Train Acc: 0.942 Eval Loss: 0.411 Eval Acc: 0.873 Elapsed Time: 4466ms
Epoch: 107 Train Loss: 0.160 Train Acc: 0.946 Eval Loss: 0.431 Eval Acc: 0.866 Elapsed Time: 4422ms
Epoch: 108 Train Loss: 0.155 Train Acc: 0.948 Eval Loss: 0.424 Eval Acc: 0.868 Elapsed Time: 4479ms
Epoch: 109 Train Loss: 0.148 Train Acc: 0.949 Eval Loss: 0.427 Eval Acc: 0.867 Elapsed Time: 4426ms
Epoch: 110 Train Loss: 0.140 Train Acc: 0.953 Eval Loss: 0.433 Eval Acc: 0.870 Elapsed Time: 4503ms
Epoch: 111 Train Loss: 0.134 Train Acc: 0.955 Eval Loss: 0.440 Eval Acc: 0.867 Elapsed Time: 4460ms
Epoch: 112 Train Loss: 0.130 Train Acc: 0.955 Eval Loss: 0.447 Eval Acc: 0.869 Elapsed Time: 4551ms
Epoch: 113 Train Loss: 0.124 Train Acc: 0.958 Eval Loss: 0.445 Eval Acc: 0.867 Elapsed Time: 4451ms
Epoch: 114 Train Loss: 0.122 Train Acc: 0.958 Eval Loss: 0.449 Eval Acc: 0.871 Elapsed Time: 4556ms
Epoch: 115 Train Loss: 0.118 Train Acc: 0.960 Eval Loss: 0.448 Eval Acc: 0.871 Elapsed Time: 4474ms
Epoch: 116 Train Loss: 0.114 Train Acc: 0.961 Eval Loss: 0.449 Eval Acc: 0.873 Elapsed Time: 4440ms
Epoch: 117 Train Loss: 0.110 Train Acc: 0.962 Eval Loss: 0.455 Eval Acc: 0.869 Elapsed Time: 4715ms
Epoch: 118 Train Loss: 0.108 Train Acc: 0.963 Eval Loss: 0.463 Eval Acc: 0.871 Elapsed Time: 4493ms
Epoch: 119 Train Loss: 0.102 Train Acc: 0.965 Eval Loss: 0.469 Eval Acc: 0.868 Elapsed Time: 4516ms
Epoch: 120 Train Loss: 0.098 Train Acc: 0.967 Eval Loss: 0.471 Eval Acc: 0.870 Elapsed Time: 4546ms
Epoch: 121 Train Loss: 0.097 Train Acc: 0.966 Eval Loss: 0.478 Eval Acc: 0.871 Elapsed Time: 4612ms
Epoch: 122 Train Loss: 0.097 Train Acc: 0.967 Eval Loss: 0.481 Eval Acc: 0.867 Elapsed Time: 4699ms
Epoch: 123 Train Loss: 0.091 Train Acc: 0.968 Eval Loss: 0.482 Eval Acc: 0.867 Elapsed Time: 4579ms
Epoch: 124 Train Loss: 0.092 Train Acc: 0.969 Eval Loss: 0.490 Eval Acc: 0.865 Elapsed Time: 4604ms
Epoch: 125 Train Loss: 0.088 Train Acc: 0.970 Eval Loss: 0.483 Eval Acc: 0.867 Elapsed Time: 4528ms
Epoch: 126 Train Loss: 0.086 Train Acc: 0.970 Eval Loss: 0.494 Eval Acc: 0.869 Elapsed Time: 4449ms
Epoch: 127 Train Loss: 0.080 Train Acc: 0.972 Eval Loss: 0.490 Eval Acc: 0.872 Elapsed Time: 4468ms
Epoch: 128 Train Loss: 0.080 Train Acc: 0.973 Eval Loss: 0.497 Eval Acc: 0.866 Elapsed Time: 4476ms
Epoch: 129 Train Loss: 0.080 Train Acc: 0.972 Eval Loss: 0.513 Eval Acc: 0.867 Elapsed Time: 4426ms
Epoch: 130 Train Loss: 0.076 Train Acc: 0.974 Eval Loss: 0.505 Eval Acc: 0.868 Elapsed Time: 4464ms
Epoch: 131 Train Loss: 0.076 Train Acc: 0.974 Eval Loss: 0.524 Eval Acc: 0.866 Elapsed Time: 4542ms
Epoch: 132 Train Loss: 0.080 Train Acc: 0.973 Eval Loss: 0.510 Eval Acc: 0.867 Elapsed Time: 4538ms
Epoch: 133 Train Loss: 0.074 Train Acc: 0.975 Eval Loss: 0.522 Eval Acc: 0.869 Elapsed Time: 4520ms
Epoch: 134 Train Loss: 0.072 Train Acc: 0.976 Eval Loss: 0.510 Eval Acc: 0.868 Elapsed Time: 4703ms
Epoch: 135 Train Loss: 0.072 Train Acc: 0.975 Eval Loss: 0.513 Eval Acc: 0.868 Elapsed Time: 4540ms
Epoch: 136 Train Loss: 0.068 Train Acc: 0.977 Eval Loss: 0.526 Eval Acc: 0.864 Elapsed Time: 4551ms
Epoch: 137 Train Loss: 0.070 Train Acc: 0.975 Eval Loss: 0.529 Eval Acc: 0.869 Elapsed Time: 4449ms
Epoch: 138 Train Loss: 0.065 Train Acc: 0.977 Eval Loss: 0.529 Eval Acc: 0.866 Elapsed Time: 4502ms
Epoch: 139 Train Loss: 0.061 Train Acc: 0.979 Eval Loss: 0.527 Eval Acc: 0.868 Elapsed Time: 4567ms
Epoch: 140 Train Loss: 0.066 Train Acc: 0.978 Eval Loss: 0.534 Eval Acc: 0.866 Elapsed Time: 4490ms
Epoch: 141 Train Loss: 0.065 Train Acc: 0.978 Eval Loss: 0.540 Eval Acc: 0.863 Elapsed Time: 4534ms
Epoch: 142 Train Loss: 0.063 Train Acc: 0.979 Eval Loss: 0.550 Eval Acc: 0.866 Elapsed Time: 4520ms
Epoch: 143 Train Loss: 0.064 Train Acc: 0.978 Eval Loss: 0.555 Eval Acc: 0.863 Elapsed Time: 4599ms
Epoch: 144 Train Loss: 0.060 Train Acc: 0.980 Eval Loss: 0.534 Eval Acc: 0.868 Elapsed Time: 4528ms
Epoch: 145 Train Loss: 0.063 Train Acc: 0.978 Eval Loss: 0.558 Eval Acc: 0.864 Elapsed Time: 4646ms
Epoch: 146 Train Loss: 0.061 Train Acc: 0.978 Eval Loss: 0.550 Eval Acc: 0.865 Elapsed Time: 4545ms
Epoch: 147 Train Loss: 0.062 Train Acc: 0.979 Eval Loss: 0.552 Eval Acc: 0.865 Elapsed Time: 4513ms
Epoch: 148 Train Loss: 0.057 Train Acc: 0.981 Eval Loss: 0.551 Eval Acc: 0.871 Elapsed Time: 4577ms
Epoch: 149 Train Loss: 0.058 Train Acc: 0.980 Eval Loss: 0.543 Eval Acc: 0.870 Elapsed Time: 4591ms
Epoch: 150 Train Loss: 0.060 Train Acc: 0.979 Eval Loss: 0.557 Eval Acc: 0.867 Elapsed Time: 4475ms
Epoch: 151 Train Loss: 0.046 Train Acc: 0.984 Eval Loss: 0.522 Eval Acc: 0.872 Elapsed Time: 4531ms
Epoch: 152 Train Loss: 0.040 Train Acc: 0.987 Eval Loss: 0.520 Eval Acc: 0.873 Elapsed Time: 4583ms
Epoch: 153 Train Loss: 0.035 Train Acc: 0.989 Eval Loss: 0.517 Eval Acc: 0.875 Elapsed Time: 4553ms
Epoch: 154 Train Loss: 0.032 Train Acc: 0.990 Eval Loss: 0.517 Eval Acc: 0.875 Elapsed Time: 4456ms
Epoch: 155 Train Loss: 0.032 Train Acc: 0.990 Eval Loss: 0.517 Eval Acc: 0.876 Elapsed Time: 4553ms
Epoch: 156 Train Loss: 0.029 Train Acc: 0.991 Eval Loss: 0.520 Eval Acc: 0.876 Elapsed Time: 4535ms
Epoch: 157 Train Loss: 0.029 Train Acc: 0.991 Eval Loss: 0.521 Eval Acc: 0.875 Elapsed Time: 4533ms
Epoch: 158 Train Loss: 0.029 Train Acc: 0.991 Eval Loss: 0.524 Eval Acc: 0.875 Elapsed Time: 4433ms
Epoch: 159 Train Loss: 0.027 Train Acc: 0.992 Eval Loss: 0.528 Eval Acc: 0.876 Elapsed Time: 4519ms
Epoch: 160 Train Loss: 0.027 Train Acc: 0.992 Eval Loss: 0.528 Eval Acc: 0.877 Elapsed Time: 4486ms
Epoch: 161 Train Loss: 0.026 Train Acc: 0.992 Eval Loss: 0.528 Eval Acc: 0.876 Elapsed Time: 4459ms
Epoch: 162 Train Loss: 0.025 Train Acc: 0.992 Eval Loss: 0.530 Eval Acc: 0.875 Elapsed Time: 4491ms
Epoch: 163 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.529 Eval Acc: 0.876 Elapsed Time: 4487ms
Epoch: 164 Train Loss: 0.025 Train Acc: 0.992 Eval Loss: 0.534 Eval Acc: 0.875 Elapsed Time: 4497ms
Epoch: 165 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.537 Eval Acc: 0.875 Elapsed Time: 4571ms
Epoch: 166 Train Loss: 0.023 Train Acc: 0.993 Eval Loss: 0.538 Eval Acc: 0.876 Elapsed Time: 4468ms
Epoch: 167 Train Loss: 0.023 Train Acc: 0.993 Eval Loss: 0.536 Eval Acc: 0.876 Elapsed Time: 4443ms
Epoch: 168 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.541 Eval Acc: 0.875 Elapsed Time: 4527ms
Epoch: 169 Train Loss: 0.022 Train Acc: 0.993 Eval Loss: 0.539 Eval Acc: 0.876 Elapsed Time: 4597ms
Epoch: 170 Train Loss: 0.021 Train Acc: 0.994 Eval Loss: 0.542 Eval Acc: 0.876 Elapsed Time: 4483ms
Epoch: 171 Train Loss: 0.022 Train Acc: 0.994 Eval Loss: 0.540 Eval Acc: 0.875 Elapsed Time: 4551ms
Epoch: 172 Train Loss: 0.022 Train Acc: 0.994 Eval Loss: 0.545 Eval Acc: 0.874 Elapsed Time: 4553ms
Epoch: 173 Train Loss: 0.022 Train Acc: 0.994 Eval Loss: 0.549 Eval Acc: 0.874 Elapsed Time: 4473ms
Epoch: 174 Train Loss: 0.021 Train Acc: 0.994 Eval Loss: 0.547 Eval Acc: 0.875 Elapsed Time: 4674ms
Epoch: 175 Train Loss: 0.020 Train Acc: 0.994 Eval Loss: 0.549 Eval Acc: 0.875 Elapsed Time: 4664ms
Epoch: 176 Train Loss: 0.020 Train Acc: 0.994 Eval Loss: 0.548 Eval Acc: 0.876 Elapsed Time: 4662ms
Epoch: 177 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.554 Eval Acc: 0.875 Elapsed Time: 4652ms
Epoch: 178 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.554 Eval Acc: 0.877 Elapsed Time: 4650ms
Epoch: 179 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.553 Eval Acc: 0.875 Elapsed Time: 4609ms
Epoch: 180 Train Loss: 0.019 Train Acc: 0.995 Eval Loss: 0.554 Eval Acc: 0.876 Elapsed Time: 4545ms
Epoch: 181 Train Loss: 0.018 Train Acc: 0.995 Eval Loss: 0.559 Eval Acc: 0.876 Elapsed Time: 4628ms
Epoch: 182 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.557 Eval Acc: 0.875 Elapsed Time: 4469ms
Epoch: 183 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.561 Eval Acc: 0.876 Elapsed Time: 4642ms
Epoch: 184 Train Loss: 0.018 Train Acc: 0.995 Eval Loss: 0.562 Eval Acc: 0.874 Elapsed Time: 4538ms
Epoch: 185 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.562 Eval Acc: 0.875 Elapsed Time: 4573ms
Epoch: 186 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.569 Eval Acc: 0.873 Elapsed Time: 4554ms
Epoch: 187 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.569 Eval Acc: 0.876 Elapsed Time: 4581ms
Epoch: 188 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.570 Eval Acc: 0.875 Elapsed Time: 4502ms
Epoch: 189 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.571 Eval Acc: 0.875 Elapsed Time: 4483ms
Epoch: 190 Train Loss: 0.016 Train Acc: 0.996 Eval Loss: 0.576 Eval Acc: 0.874 Elapsed Time: 4423ms
Epoch: 191 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.574 Eval Acc: 0.873 Elapsed Time: 4629ms
Epoch: 192 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.575 Eval Acc: 0.874 Elapsed Time: 4629ms
Epoch: 193 Train Loss: 0.015 Train Acc: 0.995 Eval Loss: 0.577 Eval Acc: 0.874 Elapsed Time: 4573ms
Epoch: 194 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.578 Eval Acc: 0.873 Elapsed Time: 4459ms
Epoch: 195 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.575 Eval Acc: 0.876 Elapsed Time: 4472ms
Epoch: 196 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.576 Eval Acc: 0.877 Elapsed Time: 4476ms
Epoch: 197 Train Loss: 0.014 Train Acc: 0.996 Eval Loss: 0.579 Eval Acc: 0.875 Elapsed Time: 4563ms
Epoch: 198 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.576 Eval Acc: 0.876 Elapsed Time: 4511ms
Epoch: 199 Train Loss: 0.014 Train Acc: 0.996 Eval Loss: 0.580 Eval Acc: 0.876 Elapsed Time: 4499ms
Epoch: 200 Train Loss: 0.015 Train Acc: 0.995 Eval Loss: 0.581 Eval Acc: 0.876 Elapsed Time: 4516ms
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
Epoch: 000 Eval Loss: 6.387 Eval Acc: 0.099
Epoch: 001 Train Loss: 6.857 Train Acc: 0.106 Eval Loss: 875.089 Eval Acc: 0.140 Elapsed Time: 3417 ms
Epoch: 002 Train Loss: 2.640 Train Acc: 0.150 Eval Loss: 2.562 Eval Acc: 0.176 Elapsed Time: 2999 ms
Epoch: 003 Train Loss: 2.232 Train Acc: 0.201 Eval Loss: 2.891 Eval Acc: 0.161 Elapsed Time: 3111 ms
Epoch: 004 Train Loss: 2.229 Train Acc: 0.222 Eval Loss: 3.571 Eval Acc: 0.242 Elapsed Time: 3008 ms
Epoch: 005 Train Loss: 2.093 Train Acc: 0.234 Eval Loss: 2.011 Eval Acc: 0.249 Elapsed Time: 3015 ms
Epoch: 006 Train Loss: 1.975 Train Acc: 0.264 Eval Loss: 1.892 Eval Acc: 0.294 Elapsed Time: 3011 ms
Epoch: 007 Train Loss: 1.868 Train Acc: 0.307 Eval Loss: 1.767 Eval Acc: 0.345 Elapsed Time: 3040 ms
Epoch: 008 Train Loss: 1.807 Train Acc: 0.336 Eval Loss: 1.883 Eval Acc: 0.328 Elapsed Time: 3124 ms
Epoch: 009 Train Loss: 1.728 Train Acc: 0.360 Eval Loss: 1.646 Eval Acc: 0.394 Elapsed Time: 3128 ms
Epoch: 010 Train Loss: 1.635 Train Acc: 0.396 Eval Loss: 1.539 Eval Acc: 0.430 Elapsed Time: 3047 ms
Epoch: 011 Train Loss: 1.575 Train Acc: 0.419 Eval Loss: 1.527 Eval Acc: 0.451 Elapsed Time: 3022 ms
Epoch: 012 Train Loss: 1.526 Train Acc: 0.441 Eval Loss: 1.424 Eval Acc: 0.473 Elapsed Time: 3092 ms
Epoch: 013 Train Loss: 1.476 Train Acc: 0.458 Eval Loss: 1.464 Eval Acc: 0.472 Elapsed Time: 3058 ms
Epoch: 014 Train Loss: 1.426 Train Acc: 0.478 Eval Loss: 1.408 Eval Acc: 0.488 Elapsed Time: 3160 ms
Epoch: 015 Train Loss: 1.374 Train Acc: 0.498 Eval Loss: 1.297 Eval Acc: 0.531 Elapsed Time: 3124 ms
Epoch: 016 Train Loss: 1.334 Train Acc: 0.515 Eval Loss: 1.297 Eval Acc: 0.530 Elapsed Time: 3073 ms
Epoch: 017 Train Loss: 1.282 Train Acc: 0.534 Eval Loss: 1.238 Eval Acc: 0.555 Elapsed Time: 3070 ms
Epoch: 018 Train Loss: 1.241 Train Acc: 0.552 Eval Loss: 1.277 Eval Acc: 0.534 Elapsed Time: 3168 ms
Epoch: 019 Train Loss: 1.201 Train Acc: 0.567 Eval Loss: 1.217 Eval Acc: 0.569 Elapsed Time: 3137 ms
Epoch: 020 Train Loss: 1.159 Train Acc: 0.582 Eval Loss: 1.187 Eval Acc: 0.582 Elapsed Time: 3104 ms
Epoch: 021 Train Loss: 1.114 Train Acc: 0.599 Eval Loss: 1.085 Eval Acc: 0.617 Elapsed Time: 3180 ms
Epoch: 022 Train Loss: 1.068 Train Acc: 0.619 Eval Loss: 1.063 Eval Acc: 0.627 Elapsed Time: 3000 ms
Epoch: 023 Train Loss: 1.035 Train Acc: 0.632 Eval Loss: 1.039 Eval Acc: 0.632 Elapsed Time: 3173 ms
Epoch: 024 Train Loss: 0.991 Train Acc: 0.649 Eval Loss: 1.002 Eval Acc: 0.647 Elapsed Time: 3083 ms
Epoch: 025 Train Loss: 0.953 Train Acc: 0.661 Eval Loss: 0.981 Eval Acc: 0.656 Elapsed Time: 3045 ms
Epoch: 026 Train Loss: 0.906 Train Acc: 0.679 Eval Loss: 0.903 Eval Acc: 0.685 Elapsed Time: 3112 ms
Epoch: 027 Train Loss: 0.890 Train Acc: 0.687 Eval Loss: 0.921 Eval Acc: 0.678 Elapsed Time: 3132 ms
Epoch: 028 Train Loss: 0.855 Train Acc: 0.699 Eval Loss: 0.909 Eval Acc: 0.686 Elapsed Time: 2967 ms
Epoch: 029 Train Loss: 0.827 Train Acc: 0.710 Eval Loss: 0.881 Eval Acc: 0.688 Elapsed Time: 3065 ms
Epoch: 030 Train Loss: 0.804 Train Acc: 0.716 Eval Loss: 0.870 Eval Acc: 0.701 Elapsed Time: 3122 ms
Epoch: 031 Train Loss: 0.782 Train Acc: 0.725 Eval Loss: 0.820 Eval Acc: 0.716 Elapsed Time: 3150 ms
Epoch: 032 Train Loss: 0.768 Train Acc: 0.732 Eval Loss: 0.801 Eval Acc: 0.725 Elapsed Time: 3014 ms
Epoch: 033 Train Loss: 0.753 Train Acc: 0.736 Eval Loss: 0.821 Eval Acc: 0.716 Elapsed Time: 3102 ms
Epoch: 034 Train Loss: 0.737 Train Acc: 0.741 Eval Loss: 0.830 Eval Acc: 0.713 Elapsed Time: 3140 ms
Epoch: 035 Train Loss: 0.714 Train Acc: 0.751 Eval Loss: 0.808 Eval Acc: 0.728 Elapsed Time: 3060 ms
Epoch: 036 Train Loss: 0.705 Train Acc: 0.752 Eval Loss: 0.736 Eval Acc: 0.748 Elapsed Time: 3055 ms
Epoch: 037 Train Loss: 0.689 Train Acc: 0.761 Eval Loss: 0.790 Eval Acc: 0.728 Elapsed Time: 3069 ms
Epoch: 038 Train Loss: 0.680 Train Acc: 0.762 Eval Loss: 0.732 Eval Acc: 0.749 Elapsed Time: 3173 ms
Epoch: 039 Train Loss: 0.654 Train Acc: 0.772 Eval Loss: 0.717 Eval Acc: 0.755 Elapsed Time: 2991 ms
Epoch: 040 Train Loss: 0.645 Train Acc: 0.775 Eval Loss: 0.784 Eval Acc: 0.735 Elapsed Time: 3092 ms
Epoch: 041 Train Loss: 0.644 Train Acc: 0.776 Eval Loss: 0.785 Eval Acc: 0.741 Elapsed Time: 3019 ms
Epoch: 042 Train Loss: 0.632 Train Acc: 0.782 Eval Loss: 0.719 Eval Acc: 0.756 Elapsed Time: 3035 ms
Epoch: 043 Train Loss: 0.623 Train Acc: 0.783 Eval Loss: 0.725 Eval Acc: 0.755 Elapsed Time: 3041 ms
Epoch: 044 Train Loss: 0.618 Train Acc: 0.787 Eval Loss: 0.650 Eval Acc: 0.773 Elapsed Time: 3116 ms
Epoch: 045 Train Loss: 0.604 Train Acc: 0.791 Eval Loss: 0.740 Eval Acc: 0.743 Elapsed Time: 3197 ms
Epoch: 046 Train Loss: 0.603 Train Acc: 0.790 Eval Loss: 0.664 Eval Acc: 0.772 Elapsed Time: 3077 ms
Epoch: 047 Train Loss: 0.581 Train Acc: 0.800 Eval Loss: 0.734 Eval Acc: 0.752 Elapsed Time: 3103 ms
Epoch: 048 Train Loss: 0.577 Train Acc: 0.801 Eval Loss: 0.694 Eval Acc: 0.764 Elapsed Time: 3099 ms
Epoch: 049 Train Loss: 0.568 Train Acc: 0.803 Eval Loss: 0.697 Eval Acc: 0.769 Elapsed Time: 3112 ms
Epoch: 050 Train Loss: 0.567 Train Acc: 0.802 Eval Loss: 0.707 Eval Acc: 0.763 Elapsed Time: 3074 ms
Epoch: 051 Train Loss: 0.552 Train Acc: 0.807 Eval Loss: 0.817 Eval Acc: 0.737 Elapsed Time: 3071 ms
Epoch: 052 Train Loss: 0.552 Train Acc: 0.809 Eval Loss: 0.608 Eval Acc: 0.794 Elapsed Time: 3286 ms
Epoch: 053 Train Loss: 0.542 Train Acc: 0.813 Eval Loss: 0.667 Eval Acc: 0.776 Elapsed Time: 3104 ms
Epoch: 054 Train Loss: 0.534 Train Acc: 0.814 Eval Loss: 0.628 Eval Acc: 0.793 Elapsed Time: 3150 ms
Epoch: 055 Train Loss: 0.528 Train Acc: 0.818 Eval Loss: 0.655 Eval Acc: 0.783 Elapsed Time: 3176 ms
Epoch: 056 Train Loss: 0.528 Train Acc: 0.816 Eval Loss: 0.705 Eval Acc: 0.763 Elapsed Time: 3067 ms
Epoch: 057 Train Loss: 0.513 Train Acc: 0.821 Eval Loss: 0.651 Eval Acc: 0.779 Elapsed Time: 3107 ms
Epoch: 058 Train Loss: 0.515 Train Acc: 0.822 Eval Loss: 0.752 Eval Acc: 0.760 Elapsed Time: 3042 ms
Epoch: 059 Train Loss: 0.507 Train Acc: 0.826 Eval Loss: 0.665 Eval Acc: 0.779 Elapsed Time: 2998 ms
Epoch: 060 Train Loss: 0.505 Train Acc: 0.826 Eval Loss: 0.651 Eval Acc: 0.779 Elapsed Time: 3078 ms
Epoch: 061 Train Loss: 0.502 Train Acc: 0.827 Eval Loss: 0.626 Eval Acc: 0.797 Elapsed Time: 2928 ms
Epoch: 062 Train Loss: 0.493 Train Acc: 0.830 Eval Loss: 0.615 Eval Acc: 0.791 Elapsed Time: 2978 ms
Epoch: 063 Train Loss: 0.488 Train Acc: 0.832 Eval Loss: 0.743 Eval Acc: 0.761 Elapsed Time: 3097 ms
Epoch: 064 Train Loss: 0.486 Train Acc: 0.832 Eval Loss: 0.654 Eval Acc: 0.789 Elapsed Time: 3169 ms
Epoch: 065 Train Loss: 0.486 Train Acc: 0.831 Eval Loss: 0.633 Eval Acc: 0.789 Elapsed Time: 3094 ms
Epoch: 066 Train Loss: 0.484 Train Acc: 0.833 Eval Loss: 0.744 Eval Acc: 0.760 Elapsed Time: 2986 ms
Epoch: 067 Train Loss: 0.481 Train Acc: 0.836 Eval Loss: 0.638 Eval Acc: 0.790 Elapsed Time: 2969 ms
Epoch: 068 Train Loss: 0.468 Train Acc: 0.840 Eval Loss: 0.623 Eval Acc: 0.790 Elapsed Time: 2991 ms
Epoch: 069 Train Loss: 0.461 Train Acc: 0.841 Eval Loss: 0.645 Eval Acc: 0.787 Elapsed Time: 2961 ms
Epoch: 070 Train Loss: 0.463 Train Acc: 0.840 Eval Loss: 0.682 Eval Acc: 0.778 Elapsed Time: 2950 ms
Epoch: 071 Train Loss: 0.456 Train Acc: 0.840 Eval Loss: 0.667 Eval Acc: 0.780 Elapsed Time: 3121 ms
Epoch: 072 Train Loss: 0.450 Train Acc: 0.845 Eval Loss: 0.620 Eval Acc: 0.793 Elapsed Time: 2987 ms
Epoch: 073 Train Loss: 0.455 Train Acc: 0.843 Eval Loss: 0.585 Eval Acc: 0.804 Elapsed Time: 3051 ms
Epoch: 074 Train Loss: 0.439 Train Acc: 0.847 Eval Loss: 0.626 Eval Acc: 0.793 Elapsed Time: 2960 ms
Epoch: 075 Train Loss: 0.446 Train Acc: 0.846 Eval Loss: 0.609 Eval Acc: 0.800 Elapsed Time: 3092 ms
Epoch: 076 Train Loss: 0.433 Train Acc: 0.850 Eval Loss: 0.586 Eval Acc: 0.807 Elapsed Time: 3095 ms
Epoch: 077 Train Loss: 0.437 Train Acc: 0.850 Eval Loss: 0.589 Eval Acc: 0.808 Elapsed Time: 3071 ms
Epoch: 078 Train Loss: 0.429 Train Acc: 0.852 Eval Loss: 0.661 Eval Acc: 0.783 Elapsed Time: 3126 ms
Epoch: 079 Train Loss: 0.433 Train Acc: 0.850 Eval Loss: 0.642 Eval Acc: 0.791 Elapsed Time: 3114 ms
Epoch: 080 Train Loss: 0.426 Train Acc: 0.854 Eval Loss: 0.666 Eval Acc: 0.778 Elapsed Time: 3120 ms
Epoch: 081 Train Loss: 0.430 Train Acc: 0.851 Eval Loss: 0.569 Eval Acc: 0.807 Elapsed Time: 3029 ms
Epoch: 082 Train Loss: 0.416 Train Acc: 0.859 Eval Loss: 0.606 Eval Acc: 0.804 Elapsed Time: 3142 ms
Epoch: 083 Train Loss: 0.418 Train Acc: 0.858 Eval Loss: 0.648 Eval Acc: 0.794 Elapsed Time: 3037 ms
Epoch: 084 Train Loss: 0.412 Train Acc: 0.858 Eval Loss: 0.589 Eval Acc: 0.811 Elapsed Time: 3111 ms
Epoch: 085 Train Loss: 0.421 Train Acc: 0.854 Eval Loss: 0.540 Eval Acc: 0.822 Elapsed Time: 3115 ms
Epoch: 086 Train Loss: 0.412 Train Acc: 0.857 Eval Loss: 0.592 Eval Acc: 0.808 Elapsed Time: 3210 ms
Epoch: 087 Train Loss: 0.410 Train Acc: 0.859 Eval Loss: 0.578 Eval Acc: 0.808 Elapsed Time: 3057 ms
Epoch: 088 Train Loss: 0.410 Train Acc: 0.858 Eval Loss: 0.641 Eval Acc: 0.788 Elapsed Time: 3119 ms
Epoch: 089 Train Loss: 0.407 Train Acc: 0.861 Eval Loss: 0.623 Eval Acc: 0.799 Elapsed Time: 3180 ms
Epoch: 090 Train Loss: 0.399 Train Acc: 0.862 Eval Loss: 0.674 Eval Acc: 0.781 Elapsed Time: 3056 ms
Epoch: 091 Train Loss: 0.398 Train Acc: 0.864 Eval Loss: 0.651 Eval Acc: 0.785 Elapsed Time: 3186 ms
Epoch: 092 Train Loss: 0.398 Train Acc: 0.862 Eval Loss: 0.642 Eval Acc: 0.793 Elapsed Time: 3150 ms
Epoch: 093 Train Loss: 0.394 Train Acc: 0.865 Eval Loss: 0.550 Eval Acc: 0.819 Elapsed Time: 3158 ms
Epoch: 094 Train Loss: 0.389 Train Acc: 0.866 Eval Loss: 0.626 Eval Acc: 0.802 Elapsed Time: 3066 ms
Epoch: 095 Train Loss: 0.394 Train Acc: 0.862 Eval Loss: 0.533 Eval Acc: 0.824 Elapsed Time: 3131 ms
Epoch: 096 Train Loss: 0.386 Train Acc: 0.865 Eval Loss: 0.568 Eval Acc: 0.815 Elapsed Time: 3115 ms
Epoch: 097 Train Loss: 0.380 Train Acc: 0.869 Eval Loss: 0.608 Eval Acc: 0.802 Elapsed Time: 3032 ms
Epoch: 098 Train Loss: 0.383 Train Acc: 0.867 Eval Loss: 0.640 Eval Acc: 0.797 Elapsed Time: 3082 ms
Epoch: 099 Train Loss: 0.385 Train Acc: 0.866 Eval Loss: 0.658 Eval Acc: 0.792 Elapsed Time: 3121 ms
Epoch: 100 Train Loss: 0.382 Train Acc: 0.867 Eval Loss: 0.561 Eval Acc: 0.817 Elapsed Time: 3152 ms
Epoch: 101 Train Loss: 0.268 Train Acc: 0.909 Eval Loss: 0.385 Eval Acc: 0.872 Elapsed Time: 3042 ms
Epoch: 102 Train Loss: 0.213 Train Acc: 0.927 Eval Loss: 0.378 Eval Acc: 0.876 Elapsed Time: 3063 ms
Epoch: 103 Train Loss: 0.194 Train Acc: 0.934 Eval Loss: 0.383 Eval Acc: 0.875 Elapsed Time: 3216 ms
Epoch: 104 Train Loss: 0.179 Train Acc: 0.938 Eval Loss: 0.375 Eval Acc: 0.879 Elapsed Time: 3159 ms
Epoch: 105 Train Loss: 0.171 Train Acc: 0.941 Eval Loss: 0.374 Eval Acc: 0.882 Elapsed Time: 3093 ms
Epoch: 106 Train Loss: 0.163 Train Acc: 0.945 Eval Loss: 0.378 Eval Acc: 0.882 Elapsed Time: 3270 ms
Epoch: 107 Train Loss: 0.153 Train Acc: 0.948 Eval Loss: 0.384 Eval Acc: 0.879 Elapsed Time: 3193 ms
Epoch: 108 Train Loss: 0.150 Train Acc: 0.949 Eval Loss: 0.383 Eval Acc: 0.881 Elapsed Time: 3295 ms
Epoch: 109 Train Loss: 0.144 Train Acc: 0.951 Eval Loss: 0.384 Eval Acc: 0.882 Elapsed Time: 3105 ms
Epoch: 110 Train Loss: 0.138 Train Acc: 0.952 Eval Loss: 0.390 Eval Acc: 0.883 Elapsed Time: 3133 ms
Epoch: 111 Train Loss: 0.130 Train Acc: 0.955 Eval Loss: 0.391 Eval Acc: 0.882 Elapsed Time: 3182 ms
Epoch: 112 Train Loss: 0.124 Train Acc: 0.957 Eval Loss: 0.401 Eval Acc: 0.880 Elapsed Time: 3053 ms
Epoch: 113 Train Loss: 0.123 Train Acc: 0.958 Eval Loss: 0.399 Eval Acc: 0.884 Elapsed Time: 3174 ms
Epoch: 114 Train Loss: 0.120 Train Acc: 0.958 Eval Loss: 0.402 Eval Acc: 0.881 Elapsed Time: 3232 ms
Epoch: 115 Train Loss: 0.116 Train Acc: 0.960 Eval Loss: 0.410 Eval Acc: 0.882 Elapsed Time: 3297 ms
Epoch: 116 Train Loss: 0.111 Train Acc: 0.962 Eval Loss: 0.406 Eval Acc: 0.882 Elapsed Time: 3006 ms
Epoch: 117 Train Loss: 0.110 Train Acc: 0.963 Eval Loss: 0.404 Eval Acc: 0.882 Elapsed Time: 3252 ms
Epoch: 118 Train Loss: 0.106 Train Acc: 0.964 Eval Loss: 0.420 Eval Acc: 0.882 Elapsed Time: 3084 ms
Epoch: 119 Train Loss: 0.100 Train Acc: 0.967 Eval Loss: 0.426 Eval Acc: 0.876 Elapsed Time: 3068 ms
Epoch: 120 Train Loss: 0.099 Train Acc: 0.966 Eval Loss: 0.418 Eval Acc: 0.881 Elapsed Time: 3089 ms
Epoch: 121 Train Loss: 0.099 Train Acc: 0.966 Eval Loss: 0.430 Eval Acc: 0.880 Elapsed Time: 3298 ms
Epoch: 122 Train Loss: 0.097 Train Acc: 0.967 Eval Loss: 0.428 Eval Acc: 0.880 Elapsed Time: 3226 ms
Epoch: 123 Train Loss: 0.092 Train Acc: 0.968 Eval Loss: 0.437 Eval Acc: 0.879 Elapsed Time: 3076 ms
Epoch: 124 Train Loss: 0.090 Train Acc: 0.969 Eval Loss: 0.441 Eval Acc: 0.879 Elapsed Time: 3114 ms
Epoch: 125 Train Loss: 0.088 Train Acc: 0.970 Eval Loss: 0.438 Eval Acc: 0.879 Elapsed Time: 3080 ms
Epoch: 126 Train Loss: 0.087 Train Acc: 0.971 Eval Loss: 0.450 Eval Acc: 0.882 Elapsed Time: 3137 ms
Epoch: 127 Train Loss: 0.082 Train Acc: 0.972 Eval Loss: 0.446 Eval Acc: 0.878 Elapsed Time: 3096 ms
Epoch: 128 Train Loss: 0.080 Train Acc: 0.972 Eval Loss: 0.443 Eval Acc: 0.882 Elapsed Time: 3006 ms
Epoch: 129 Train Loss: 0.079 Train Acc: 0.973 Eval Loss: 0.461 Eval Acc: 0.877 Elapsed Time: 2985 ms
Epoch: 130 Train Loss: 0.077 Train Acc: 0.973 Eval Loss: 0.453 Eval Acc: 0.878 Elapsed Time: 3104 ms
Epoch: 131 Train Loss: 0.077 Train Acc: 0.973 Eval Loss: 0.457 Eval Acc: 0.879 Elapsed Time: 3095 ms
Epoch: 132 Train Loss: 0.078 Train Acc: 0.973 Eval Loss: 0.453 Eval Acc: 0.881 Elapsed Time: 3040 ms
Epoch: 133 Train Loss: 0.074 Train Acc: 0.974 Eval Loss: 0.470 Eval Acc: 0.878 Elapsed Time: 3009 ms
Epoch: 134 Train Loss: 0.073 Train Acc: 0.975 Eval Loss: 0.462 Eval Acc: 0.879 Elapsed Time: 3029 ms
Epoch: 135 Train Loss: 0.072 Train Acc: 0.975 Eval Loss: 0.475 Eval Acc: 0.878 Elapsed Time: 3135 ms
Epoch: 136 Train Loss: 0.074 Train Acc: 0.974 Eval Loss: 0.473 Eval Acc: 0.876 Elapsed Time: 3072 ms
Epoch: 137 Train Loss: 0.070 Train Acc: 0.976 Eval Loss: 0.477 Eval Acc: 0.880 Elapsed Time: 2991 ms
Epoch: 138 Train Loss: 0.067 Train Acc: 0.978 Eval Loss: 0.480 Eval Acc: 0.877 Elapsed Time: 3155 ms
Epoch: 139 Train Loss: 0.067 Train Acc: 0.977 Eval Loss: 0.485 Eval Acc: 0.880 Elapsed Time: 3159 ms
Epoch: 140 Train Loss: 0.065 Train Acc: 0.978 Eval Loss: 0.491 Eval Acc: 0.875 Elapsed Time: 3012 ms
Epoch: 141 Train Loss: 0.065 Train Acc: 0.977 Eval Loss: 0.481 Eval Acc: 0.878 Elapsed Time: 3104 ms
Epoch: 142 Train Loss: 0.063 Train Acc: 0.978 Eval Loss: 0.487 Eval Acc: 0.880 Elapsed Time: 3075 ms
Epoch: 143 Train Loss: 0.065 Train Acc: 0.977 Eval Loss: 0.492 Eval Acc: 0.876 Elapsed Time: 3236 ms
Epoch: 144 Train Loss: 0.063 Train Acc: 0.978 Eval Loss: 0.507 Eval Acc: 0.875 Elapsed Time: 3046 ms
Epoch: 145 Train Loss: 0.065 Train Acc: 0.977 Eval Loss: 0.509 Eval Acc: 0.874 Elapsed Time: 3162 ms
Epoch: 146 Train Loss: 0.065 Train Acc: 0.978 Eval Loss: 0.506 Eval Acc: 0.876 Elapsed Time: 3065 ms
Epoch: 147 Train Loss: 0.060 Train Acc: 0.980 Eval Loss: 0.508 Eval Acc: 0.876 Elapsed Time: 3067 ms
Epoch: 148 Train Loss: 0.061 Train Acc: 0.979 Eval Loss: 0.499 Eval Acc: 0.877 Elapsed Time: 3073 ms
Epoch: 149 Train Loss: 0.060 Train Acc: 0.980 Eval Loss: 0.522 Eval Acc: 0.873 Elapsed Time: 3161 ms
Epoch: 150 Train Loss: 0.061 Train Acc: 0.979 Eval Loss: 0.492 Eval Acc: 0.879 Elapsed Time: 3209 ms
Epoch: 151 Train Loss: 0.046 Train Acc: 0.985 Eval Loss: 0.472 Eval Acc: 0.885 Elapsed Time: 3081 ms
Epoch: 152 Train Loss: 0.039 Train Acc: 0.987 Eval Loss: 0.465 Eval Acc: 0.886 Elapsed Time: 3204 ms
Epoch: 153 Train Loss: 0.036 Train Acc: 0.988 Eval Loss: 0.470 Eval Acc: 0.886 Elapsed Time: 3265 ms
Epoch: 154 Train Loss: 0.033 Train Acc: 0.990 Eval Loss: 0.468 Eval Acc: 0.885 Elapsed Time: 2984 ms
Epoch: 155 Train Loss: 0.032 Train Acc: 0.990 Eval Loss: 0.473 Eval Acc: 0.886 Elapsed Time: 3108 ms
Epoch: 156 Train Loss: 0.033 Train Acc: 0.990 Eval Loss: 0.472 Eval Acc: 0.887 Elapsed Time: 3040 ms
Epoch: 157 Train Loss: 0.030 Train Acc: 0.991 Eval Loss: 0.469 Eval Acc: 0.887 Elapsed Time: 3161 ms
Epoch: 158 Train Loss: 0.030 Train Acc: 0.990 Eval Loss: 0.473 Eval Acc: 0.887 Elapsed Time: 3036 ms
Epoch: 159 Train Loss: 0.028 Train Acc: 0.992 Eval Loss: 0.475 Eval Acc: 0.886 Elapsed Time: 3107 ms
Epoch: 160 Train Loss: 0.028 Train Acc: 0.991 Eval Loss: 0.477 Eval Acc: 0.888 Elapsed Time: 3120 ms
Epoch: 161 Train Loss: 0.027 Train Acc: 0.992 Eval Loss: 0.476 Eval Acc: 0.888 Elapsed Time: 3111 ms
Epoch: 162 Train Loss: 0.025 Train Acc: 0.993 Eval Loss: 0.476 Eval Acc: 0.887 Elapsed Time: 3046 ms
Epoch: 163 Train Loss: 0.025 Train Acc: 0.992 Eval Loss: 0.482 Eval Acc: 0.885 Elapsed Time: 3031 ms
Epoch: 164 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.481 Eval Acc: 0.887 Elapsed Time: 3170 ms
Epoch: 165 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.483 Eval Acc: 0.887 Elapsed Time: 3002 ms
Epoch: 166 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.484 Eval Acc: 0.885 Elapsed Time: 3028 ms
Epoch: 167 Train Loss: 0.024 Train Acc: 0.992 Eval Loss: 0.488 Eval Acc: 0.885 Elapsed Time: 2984 ms
Epoch: 168 Train Loss: 0.024 Train Acc: 0.993 Eval Loss: 0.487 Eval Acc: 0.886 Elapsed Time: 3026 ms
Epoch: 169 Train Loss: 0.021 Train Acc: 0.994 Eval Loss: 0.487 Eval Acc: 0.887 Elapsed Time: 3095 ms
Epoch: 170 Train Loss: 0.022 Train Acc: 0.993 Eval Loss: 0.485 Eval Acc: 0.885 Elapsed Time: 3008 ms
Epoch: 171 Train Loss: 0.021 Train Acc: 0.993 Eval Loss: 0.487 Eval Acc: 0.887 Elapsed Time: 3074 ms
Epoch: 172 Train Loss: 0.022 Train Acc: 0.993 Eval Loss: 0.489 Eval Acc: 0.888 Elapsed Time: 2955 ms
Epoch: 173 Train Loss: 0.022 Train Acc: 0.993 Eval Loss: 0.495 Eval Acc: 0.886 Elapsed Time: 2974 ms
Epoch: 174 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.491 Eval Acc: 0.888 Elapsed Time: 2937 ms
Epoch: 175 Train Loss: 0.020 Train Acc: 0.994 Eval Loss: 0.494 Eval Acc: 0.887 Elapsed Time: 3112 ms
Epoch: 176 Train Loss: 0.020 Train Acc: 0.994 Eval Loss: 0.496 Eval Acc: 0.884 Elapsed Time: 2988 ms
Epoch: 177 Train Loss: 0.020 Train Acc: 0.994 Eval Loss: 0.495 Eval Acc: 0.887 Elapsed Time: 3103 ms
Epoch: 178 Train Loss: 0.020 Train Acc: 0.994 Eval Loss: 0.497 Eval Acc: 0.887 Elapsed Time: 2988 ms
Epoch: 179 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.494 Eval Acc: 0.887 Elapsed Time: 2972 ms
Epoch: 180 Train Loss: 0.019 Train Acc: 0.995 Eval Loss: 0.498 Eval Acc: 0.886 Elapsed Time: 2950 ms
Epoch: 181 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.502 Eval Acc: 0.886 Elapsed Time: 3122 ms
Epoch: 182 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.498 Eval Acc: 0.888 Elapsed Time: 3013 ms
Epoch: 183 Train Loss: 0.018 Train Acc: 0.994 Eval Loss: 0.499 Eval Acc: 0.887 Elapsed Time: 3075 ms
Epoch: 184 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.501 Eval Acc: 0.888 Elapsed Time: 3015 ms
Epoch: 185 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.504 Eval Acc: 0.887 Elapsed Time: 3010 ms
Epoch: 186 Train Loss: 0.018 Train Acc: 0.995 Eval Loss: 0.507 Eval Acc: 0.887 Elapsed Time: 3035 ms
Epoch: 187 Train Loss: 0.019 Train Acc: 0.994 Eval Loss: 0.505 Eval Acc: 0.887 Elapsed Time: 3066 ms
Epoch: 188 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.505 Eval Acc: 0.887 Elapsed Time: 3033 ms
Epoch: 189 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.504 Eval Acc: 0.888 Elapsed Time: 2978 ms
Epoch: 190 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.513 Eval Acc: 0.886 Elapsed Time: 3031 ms
Epoch: 191 Train Loss: 0.017 Train Acc: 0.995 Eval Loss: 0.512 Eval Acc: 0.886 Elapsed Time: 3001 ms
Epoch: 192 Train Loss: 0.016 Train Acc: 0.996 Eval Loss: 0.510 Eval Acc: 0.888 Elapsed Time: 3025 ms
Epoch: 193 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.514 Eval Acc: 0.886 Elapsed Time: 3061 ms
Epoch: 194 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.516 Eval Acc: 0.886 Elapsed Time: 2997 ms
Epoch: 195 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.515 Eval Acc: 0.886 Elapsed Time: 3053 ms
Epoch: 196 Train Loss: 0.016 Train Acc: 0.996 Eval Loss: 0.515 Eval Acc: 0.888 Elapsed Time: 2972 ms
Epoch: 197 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.516 Eval Acc: 0.889 Elapsed Time: 3064 ms
Epoch: 198 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.517 Eval Acc: 0.887 Elapsed Time: 3007 ms
Epoch: 199 Train Loss: 0.016 Train Acc: 0.995 Eval Loss: 0.518 Eval Acc: 0.887 Elapsed Time: 3014 ms
Epoch: 200 Train Loss: 0.015 Train Acc: 0.996 Eval Loss: 0.521 Eval Acc: 0.887 Elapsed Time: 3034 ms
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
