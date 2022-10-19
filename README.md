# Exposing and Exploiting Fine-Grained Block Structures for Fast and Accurate Sparse Training

Abstract: _Sparse training is a popular technique to reduce the overhead of training large models. Although previous work has shown promising results for nonstructured sparse models, it is still unclear whether a sparse model with structural constraints can be trained from scratch to high accuracy. In this work, we study the dynamic sparse training for a class of sparse models with shuffled block structures. Compared to nonstructured models, such fine-grained structured models are more hardware-friendly and can effectively accelerate the training process. We propose an algorithm that keeps adapting the sparse model while maintaining the active parameters in shuffled blocks. We conduct experiments on a variety of networks and datasets and obtain positive results. In particular, on ImageNet, we achieve dense accuracy for ResNet50 and ResNet18 at 0.5 sparsity. On CIFAR10/100, we show that dense accuracy can be recovered at 0.6 sparsity for various models. At higher sparsity, our algorithm can still match the accuracy of nonstructured sparse training in most cases, while reducing the training time by up to 5x due to the fine-grained block structures in the models._

## Requirements

The library requires Python 3.7.8, PyTorch v1.10.1, and CUDA v11.6.

## Datasets
Our code will automatically download the CIFAR10 & CIFAR100 datasets when you first choose these two datasets.

For Imagenet-2012, you can download by https://image-net.org/challenges/LSVRC/2012/

For Tiny-Imagenet, you can use
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

## Training On CIFAR10/100
We provide the training codes for DSB(Dynamic Shuffled Block) and SSB(Static Shuffled Block).

To train a **RigL model**, we just need to use --block_size 0.
* To train WideResNet22-2 on CIFAR10 with block_size=16 and sparsity=0.9, run the following command:
```
python main.py --data_set cifar10 --data_dir YOUR_DATA_PATH --model wrn-22-2 --lr 0.1 --batch_size 128 --sparsity 0.9 --epochs 160 --T_end 75000 --T 100 --block_size 16
```
* To train VGG-16 on CIFAR100 with block_size=16 and sparsity=0.9, run the following command:
```
python main.py --data_set cifar100 --data_dir YOUR_DATA_PATH --model vgg-d --lr 0.1 --batch_size 128 --sparsity 0.9 --epochs 160 --T_end 75000 --T 100 --block_size 16
```

## Training On ImageNet/Tiny-ImageNet
* To train ResNet-50 on ImageNet with block_size=16 and sparsity=0.5, run the following command:
```
python train_imagenet.py --data_set imagenet --sparsity 0.5 --model ResNet50 --block_size 16 --batch_size 256 --T_end 400000 --T 800
```

* To train ResNet-50 on Tiny-ImageNet with block_size=16 and sparsity=0.75, run the following command:
```
python train_imagenet.py --data_set tinyimagenet --sparsity 0.75 --model ResNet50 --block_size 16 --batch_size 256 --T_end 400000 --T 800
```

### Options:
* --model (str) - type of networks
```
 CIFAR-10/100：
	alexnet-s
	alexnet-b
	vgg-c
	vgg-d
	vgg-like
	wrn-22-2
	wrn-28-2
	wrn-22-8
	wrn-16-8
	wrn-16-10
	ResNet-18
        ResNet-34
 ImageNet/Tiny-ImageNet:
        ResNet-18
        ResNet-34
        ResNet-50
```
* --data_set (str) - dataset for training. Choose from: cifar10, cifar100
* --sparsity (float) - sparsity level for model weights (default 0.9)
* --static_update (str) - using SSB for training. If n/no (as default setting), using DSB.
