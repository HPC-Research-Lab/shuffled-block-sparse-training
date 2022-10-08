import argparse
import time
from cmath import pi
import os
from numpy.core.shape_base import block

import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np

from models import *
from convolution import *
from preprocess import *
from utils import *
from loss import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import datetime

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--model', metavar='model', default='wrn-22-2')
parser.add_argument('--epochs', default=105, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('--cuda', type=str, default='0',
                    help='Avaiable GPU ID')
parser.add_argument('--sparsity', default=0.9, type=float,
                    help='forward and backward sparsity')
parser.add_argument('--T_end', default=75000, type=int,
                    help='Stop updating weight mask in T_end iterations')
parser.add_argument('--T', default=100, type=int,
                    help='Every T iterations, update the weight mask')
parser.add_argument('--alpha', default=0.3, type=float,
                    help='alpha for updating schedule')
parser.add_argument('--block_size', type=int, default=0)
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--static_update', type=str, default='n')

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['ResNet50'] = ()
models['ResNet34'] = ()
models['ResNet18'] = ()
models['alexnet-s'] = (AlexNet, ['s', 100])
models['alexnet-b'] = (AlexNet, ['b', 100])
models['vgg-c'] = (VGG16, ['C', 100])
models['vgg-d'] = (VGG16, ['D', 100])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-22-2'] = (WideResNet, [22, 2, 100, 0.3])
models['wrn-28-2'] = (WideResNet, [28, 2, 100, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 100, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 100, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 100, 0.3])

global args, best_prec1

best_prec1 = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def googleAI_ERK(model, density, erk_power_scale: float = 1.0):
    """Given the method, returns the sparsity of individual layers as a dict.
    It ensures that the non-custom layers have a total parameter count as the one
    with uniform sparsities. In other words for the layers which are not in the
    custom_sparsity_map the following equation should be satisfied.
    # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
    Args:
      module:
      density: float, between 0 and 1.
      erk_power_scale: float, if given used to take power of the ratio. Use
        scale<1 to make the erdos_renyi softer.
    Returns:
      density_dict, dict of where keys() are equal to all_masks and individiual
        masks are mapped to the their densities.
    """
    # Obtain masks
    total_params = 0
    masks = {}
    for name, layer in model.named_modules():
        if isinstance(layer, SparseConv2d) or isinstance(layer, SparseLinear):
            masks[name] = torch.zeros_like(layer.weight, dtype=torch.float32, requires_grad=False)
            total_params += layer.weight.numel()

    # We have to enforce custom sparsities and then find the correct scaling
    # factor.

    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                raw_probabilities[name] = (
                                                  np.sum(mask.shape) / np.prod(mask.shape)
                                          ) ** erk_power_scale
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in masks.items():
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
        total_nonzero += density_dict[name] * mask.numel()

    return density_dict

def train(train_loader, model, criterion, optimizer, epoch, print_freq, device, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print('\nEpoch: %d' % epoch)
    model.train()
    # interations in training, i.e., t in rigl algorithm
    iterations = len(train_loader) * epoch
    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model.forward(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update weight mask every 100 iterations, update active and frozen mask every 10 iterations.
        if iterations < args.T_end and (iterations + 1) % args.T == 0:
            for name, layer in model.named_modules():
                if isinstance (layer, SparseConv2d):
                    layer.update_conv_weight_mask(iterations, args.alpha, args.T_end, block_size=args.block_size)
                if isinstance (layer, SparseLinear):
                    layer.update_linear_weight_mask1(iterations, args.alpha, args.T_end, block_size=args.block_size)


        #check sparsity
        nnz = 0
        tot = 0
        for name, layer in model.named_modules():
            if isinstance (layer, SparseConv2d):
                nnz += layer.weight_mask.sum().item()
                tot += layer.weight.numel()
            if isinstance (layer, SparseLinear):
                nnz += layer.weight_mask.sum().item()
                tot += layer.weight.numel()

        print(f'sparsity: {1 - nnz / tot}')
        writer.add_scalars(f'sparsity', {'[sparsity]':1 - nnz / tot}, epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        iterations += 1
    return


def validate(val_loader, model, criterion, print_freq, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    return top1.avg, top5.avg

if __name__ == '__main__':
    args = parser.parse_args()
    writer = SummaryWriter()
    class_number = 10
    if args.data_set == 'cifar10':
        class_number = 10
        train_loader, val_loader = data_loader(args.data_dir, args.batch_size)
    elif args.data_set == 'cifar100':
        class_number = 100
        train_loader, val_loader = data_loader_cifar100(args.data_dir, args.batch_size)
    elif args.data_set=='tinyimagenet':
        class_number = 200
        train_loader, val_loader, _ = tiny_imagenet_dataloaders(args.data_dir, args.batch_size)
    elif args.data_set=='imagenet':
        class_number = 1000
        train_loader, val_loader = imagenet_dataloaders(args.data_dir, args.batch_size)
    else:
        assert 'dataset set error'

    device = int(args.cuda)

    if args.model not in models:
        print('You need to select an existing model via the --model argument. Available models include: ')
        for key in models:
            print('\t{0}'.format(key))
        raise Exception('You need to select a model')
    elif args.model == 'ResNet18' or args.model == 'resnet18':
        model = ResNet18(c=class_number).to(device)
    elif args.model == 'ResNet34' or args.model == 'resnet34':
        model = ResNet34(c=class_number).to(device)
    elif args.model == 'ResNet50' or args.model == 'resnet50':
        model = ResNet50(c=class_number).to(device)
    else:
        cls, cls_args = models[args.model]
        model = cls(*(cls_args)).to(device)

    # Initialize the weight_mask
    ERK_density = googleAI_ERK(model, 1 - args.sparsity)
    for name, layer in model.named_modules():
        if isinstance (layer, SparseConv2d):
            layer.init_conv_weight_mask(1 - ERK_density[name], block_size=args.block_size)
        if isinstance (layer, SparseLinear):
            layer.init_linear_weight_mask(1 - ERK_density[name], block_size=args.block_size)

    #check sparsity
    nnz = 0
    tot = 0
    for name, layer in model.named_modules():
        if isinstance (layer, SparseConv2d):
            nnz += layer.weight_mask.sum().item()
            tot += layer.weight.numel()
        if isinstance (layer, SparseLinear):
            nnz += layer.weight_mask.sum().item()
            tot += layer.weight.numel()

    print(f'sparsity: {1 - nnz / tot}')

    criterion = LabelSmoothSoftmaxCEV1()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)
    # train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers)
    step_scheduler = MultiStepLR(optimizer, milestones=[30,70,90], gamma=0.1)
    warmup_scheduler = LearningRateWarmUP(optimizer = optimizer, warmup_epochs = 5, target_lr = args.lr, after_scheduler = step_scheduler)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    # step_scheduler = MultiStepLR(optimizer, milestones=[75,95], gamma=0.1)
    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, device, writer)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, device)
        writer.add_scalars(f'test_accuracy', {'[prec1]':prec1}, epoch)

        # print current model sparsity
        layer_idx = 0
        for name, layer in model.named_modules():
            if isinstance(layer, SparseConv2d) or isinstance(layer, SparseLinear):
                print("Layer: ", layer_idx, " Sparsity: ", 1 - torch.sum(layer.weight_mask)/layer.weight_num)
                layer_idx += 1

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.model + '.pth')
        # step_scheduler.step(epoch)
        warmup_scheduler.step(epoch)
    writer.close()