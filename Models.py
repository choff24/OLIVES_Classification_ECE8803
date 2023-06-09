import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

# This is a library of all models used throughout the work on this project. It includes 2d models as well as the final
# 3d models used

# Normal autoencoder for 2d data
class AE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(224 * 224, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )

        self.decoder = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# One version of convolutional autoencoder used
class CAE1(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(100352, 250),
            nn.Sigmoid()
        )

        self.conv_decode = nn.Sequential(
            nn.Linear(250, 100352),
            nn.ReLU(),
            nn.Unflatten(1, (32, 56, 56)),
            #nn.Unflatten(1, (4, 28, 28)),
            #nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

        )

    def forward(self, x):

        x = self.conv_encode(x)
        x = self.conv_decode(x)

        return x

    def get_representation(self, x):
        x = self.conv_encode(x)

        return x

# Another version of convolutional autoencoder used
class CAE2(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),

        )

        self.conv_decode = nn.Sequential(
            #nn.Linear(676, 25088),
            #nn.LeakyReLU(),
            nn.Unflatten(1, (8, 56, 56)),
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        )

    def forward(self, x):

        x = self.conv_encode(x)
        x = self.conv_decode(x)

        return x

    def get_representation(self, x):
        x = self.conv_encode(x)

        return x

# Final 3D convolutional autoencoder used in the project
class CAE_3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_encode = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Sigmoid()

        )

        self.conv_decode = nn.Sequential(

            nn.Unflatten(1, (4, 6, 28, 28)),
            nn.ConvTranspose3d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        )

    def forward(self, x):

        x = self.conv_encode(x)
        x = self.conv_decode(x)

        return x

    def get_representation(self, x):
        x = self.conv_encode(x)

        return x


# ANN classifier used in the project, experimented with some very deep ANNs, so this model allows me to set the depth
# and number of nuerons on initialization
class ClassifierANN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(ClassifierANN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        #self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=0.4)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(self.batch_norm)
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(0.2))

        for layer in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(self.batch_norm)
            self.layers.append(self.activation)
            self.layers.append(self.dropout)

        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


# This is the start of the 3D ResNet implementation taken from the github repo found here:
# https://github.com/xmuyzz/3D-CNN-PyTorch

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
