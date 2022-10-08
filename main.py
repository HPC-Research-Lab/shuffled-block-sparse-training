import argparse
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
from preprocess import *
from utils import *
import logging
# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--model', metavar='model', default='wrn-22-2')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
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


global args, best_acc

best_acc = 0  # best test accuracy
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

def train(train_loader, model, criterion, optimizer, epoch, device, writer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    # interations in training, i.e., t in rigl algorithm
    iterations = len(train_loader) * epoch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # find the matched convolutional layer, update weight mask, and give the corresponding mask to each layer

        if (iterations + 1) % args.T == 0 and iterations < args.T_end // 2 and args.static_update == 'n' or args.static_update == 'no':
            layer_id = 0
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


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        iterations += 1
    return


def test(test_loader, model, criterion, device):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    return acc


if __name__ == '__main__':
    args = parser.parse_args()

    class_number = 10
    if args.data_set == 'cifar10':
        class_number = 10
        train_loader, test_loader = data_loader(args.data_dir, args.batch_size)
    elif args.data_set == 'cifar100':
        class_number = 100
        train_loader, test_loader = data_loader_cifar100(args.data_dir, args.batch_size)
    elif args.data_set=='tinyimagenet':
        class_number = 200
        train_loader, test_loader, _ = tiny_imagenet_dataloaders(args.data_dir, args.batch_size)
    elif args.data_set=='imagenet':
        class_number = 1000
        train_loader, test_loader = imagenet_dataloaders(args.data_dir, args.batch_size)
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

    print(f'total sparsity of the model: {1 - nnz / tot}')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)

    # writer = SummaryWriter()
    writer = None

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader, model, criterion, optimizer, epoch, device, writer)

        acc = test(test_loader, model, criterion, device)
        # writer.add_scalar(f'test_accuracy', acc, epoch)

        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        print('best accuracy:',best_acc)
        scheduler.step()
    # writer.close()