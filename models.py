import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution import *
from linear import *


def myconv(in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 1, bias: bool=False):
    return SparseConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1):
    return SparseConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return SparseConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AlexNet(nn.Module):
    """AlexNet with batch normalization and without pooling.

    This is an adapted version of AlexNet as taken from
    SNIP: Single-shot Network Pruning based on Connection Sensitivity,
    https://arxiv.org/abs/1810.02340

    There are two different version of AlexNet:
    AlexNet-s (small): Has hidden layers with size 1024
    AlexNet-b (big):   Has hidden layers with size 2048

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config='s', num_classes=1000, save_features=False):
        super(AlexNet, self).__init__()
        self.save_features = save_features
        self.feats = []
        self.densities = []

        factor = 1 if config=='s' else 2
        self.features = nn.Sequential(
            myconv(3, 96, 11, 2, 2, True),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            myconv(96, 256, 5, 2, 2, True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            myconv(256, 384, 3, 2, 1, True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            myconv(384, 384, 3, 2, 1, True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            myconv(384, 256, 3, 2, 1, True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            SparseLinear(256, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            SparseLinear(1024*factor, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            SparseLinear(1024*factor, num_classes),
        )

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                if isinstance(layer, nn.Conv2d):
                    self.densities.append((layer.weight.data != 0.0).sum().item()/layer.weight.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class LeNet_300_100(nn.Module):
    """Simple NN with hidden layers [300, 100]

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """
    def __init__(self, save_features=None, bench_model=False):
        super(LeNet_300_100, self).__init__()
        self.fc1 = SparseLinear(28*28, 300, bias=True)
        self.fc2 = SparseLinear(300, 100, bias=True)
        self.fc3 = SparseLinear(100, 10, bias=True)
        self.mask = None

    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return F.log_softmax(x3, dim=1)


class MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False):
        super(MLP_CIFAR10, self).__init__()

        self.fc1 = SparseLinear(3*32*32, 1024)
        self.fc2 = SparseLinear(1024, 512)
        self.fc3 = SparseLinear(512, 10)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        x1 = F.relu(self.fc2(x0))
        return F.log_softmax(self.fc3(x1), dim=1)


class LeNet_5_Caffe(nn.Module):
    """LeNet-5 without padding in the first layer.
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False):
        super().__init__()
        self.conv1 = myconv(1, 20, 5, padding=0, bias=True)
        self.conv2 = myconv(20, 50, 5, bias=True)
        self.fc3 = SparseLinear(50 * 4 * 4, 500)
        self.fc4 = SparseLinear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'like': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'C': [
        64, 64, 'M', 128, 128, 'M', 256, 256, (1, 256), 'M', 512, 512, (1, 512), 'M',
        512, 512, (1, 512), 'M' # tuples indicate (kernel size, output channels)
    ]
}


class VGG16(nn.Module):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like
    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm
    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config, num_classes=10, save_features=False):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)
        self.feats = []
        self.densities = []
        self.save_features = save_features

        if config == 'C' or config == 'D':
            self.classifier = nn.Sequential(
                SparseLinear((512 if config == 'D' else 2048), 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                SparseLinear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                SparseLinear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                SparseLinear(512, 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                SparseLinear(512, num_classes),
            )

    @staticmethod
    def make_layers(config, batch_norm=False):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                kernel_size = 3
                if isinstance(v, tuple):
                    kernel_size, v = v
                    conv2d = conv1x1(in_channels, v)
                    in_channels = v
                else:
                    conv2d = conv3x3(in_channels, v)
                    in_channels = v
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                    self.densities.append((x.data != 0.0).sum().item()/x.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

class WideResNet(nn.Module):
    """
    Wide Residual Network with varying depth and width.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146

    :param depth: No of layers
    :type depth: int
    :param widen_factor: Factor to increase channel width by
    :type widen_factor: int
    :param num_classes: No of output labels
    :type num_classes: int
    :param dropRate: Dropout Probability
    :type dropRate: float
    :param small_dense_density: Equivalent parameter density of Small-Dense model
    :type small_dense_density: float
    """

    def __init__(
            self,
            depth: int = 22,
            widen_factor: int = 2,
            num_classes: int = 10,
            dropRate: float = 0.3,
            small_dense_density: float = 1.0,
    ):
        super(WideResNet, self).__init__()

        small_dense_multiplier = np.sqrt(small_dense_density)
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(c * small_dense_multiplier) for c in nChannels]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = WRNBasicBlock

        # 1st conv before any network block
        self.net0_block0_conv0 = SparseConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False, net=0, block=0, conv=0)
        self.net1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, net=1,)
        self.net2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, net=2,)
        self.net3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, net=3,)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = SparseLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.feats = []
        self.densities = []

        for m in self.modules():
            if isinstance(m, SparseConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SparseLinear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.net0_block0_conv0(x)
        out = self.net1(out)
        out = self.net2(out)
        out = self.net3(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class WRNBasicBlock(nn.Module):
    """
    Wide Residual Network basic block

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146

    :param in_planes: input channels
    :type in_planes: int
    :param out_planes: output channels
    :type out_planes: int
    :param stride: the stride of the first block of this layer
    :type stride: int
    :param dropRate: Dropout Probability
    :type dropRate: float
    """

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int,
            dropRate: float = 0.0,
            net: int = 1,
            block: int = 0,
    ):
        super(WRNBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv0 = SparseConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, net=net, block=block, conv=0)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv1 = SparseConv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, net=net, block=block, conv=1)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.conv2 = (
                (not self.equalInOut) and SparseConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, net=net, block=block, conv=2) or None
        )
        self.feats = []
        self.densities = []
        self.in_planes = in_planes

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out0 = self.conv0(x)
        else:
            out = self.relu1(self.bn1(x))
            out0 = self.conv0(out)

        out = self.relu2(self.bn2(out0))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.conv1(out)

        return torch.add(x if self.equalInOut else self.conv2(x), out)


class NetworkBlock(nn.Module):
    """
    Wide Residual Network network block which holds basic blocks.

    For more info, see the paper:
        Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
        https://arxiv.org/abs/1605.07146

    :param nb_layers: Number of blocks
    :type nb_layers: int
    :param in_planes: input channels
    :type in_planes: int
    :param out_planes: output channels
    :type out_planes: int
    :param block: Block type, BasicBlock only
    :type block: BasicBlock
    :param stride: the stride of the first block of this layer
    :type stride: int
    :param dropRate: Dropout Probability
    :type dropRate: float
    """

    def __init__(
            self,
            nb_layers: int,
            in_planes: int,
            out_planes: int,
            block: WRNBasicBlock,
            stride: int,
            dropRate: float = 0.0,
            net: int = 1,
    ):

        super(NetworkBlock, self).__init__()
        self.feats = []
        self.densities = []
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


'''
ResNet Definition
'''


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes,stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion*planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = SparseLinear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out


def ResNet18(c=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], c)

def ResNet34(c=10):
    return ResNet(BasicBlock, [3,4,6,3],c)

def ResNet50(c=10):
    model = ResNet(Bottleneck, [3, 4, 6, 3],c)
    return model

def ResNet101(c=10):
    return ResNet(Bottleneck, [3,4,23,3],c)

def ResNet152(c=10):
    return ResNet(Bottleneck, [3,8,36,3],c)

def wide_resnet22_2():
    model = ResNet(BasicBlock, [2, 2, 2, 4])
    return model
