import torch.optim as optim
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt


class GoogLeNet(nn.Module):
    def __init__(self, aux_class, training_enable):
        super(GoogLeNet, self).__init__()
        self.aux_class = aux_class
        self.training_enable = training_enable

        self.conv_layer1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_layer2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_layer1 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)

        self.inception_layer2 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_layer3 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)

        self.inception_layer4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)

        self.inception_layer5 = InceptionBlock(512, 128, 128, 256, 24, 64, 64)

        self.inception_layer6 = InceptionBlock(512, 112, 144, 288, 32, 64, 64)

        self.inception_layer7 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_layer8 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)

        self.inception_layer9 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool_layer = nn.AvgPool2d(kernel_size=7, stride=1)

        self.dropout = nn.Dropout(p=0.4)

        self.fc_layer = nn.Linear(1024, 2)  # fully connected layer

        # Auxiliary classifiers only used during training
        if self.aux_class and self.training_enable:
            self.aux1 = InceptionAux(512, 2)
            self.aux2 = InceptionAux(528, 2)

    # data passes through layers in a forward propagation
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.maxpool_layer(x)
        x = self.conv_layer2(x)
        x = self.maxpool_layer(x)
        x = self.inception_layer1(x)
        x = self.inception_layer2(x)
        x = self.maxpool_layer(x)
        x = self.inception_layer3(x)

        if self.aux_class and self.training_enable:  # Auxiliary Softmax classifier 1
            aux_1 = self.aux1(x)

        x = self.inception_layer4(x)
        x = self.inception_layer5(x)
        x = self.inception_layer6(x)

        if self.aux_class and self.training_enable:  # Auxiliary Softmax classifier 2
            aux_2 = self.aux2(x)

        x = self.inception_layer7(x)
        x = self.maxpool_layer(x)
        x = self.inception_layer8(x)
        x = self.inception_layer9(x)
        x = self.avgpool_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc_layer(x)

        if self.aux_class and self.training_enable:
            return aux_1, aux_2, x
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


# filters and max-pooling operate simultaneously in the layer. The outputs are then concatenated to be input for the next layer
# extra 1x1 convolution added before 3x3 and 5x5 convolution branches to reduce memory usage
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, kern_1x1, kern_1x1_2, kern_3x3, kern_1x1_3,
                 kern_5x5, kern_1x1_4):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, kern_1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, kern_1x1_2, kernel_size=1, stride=1, padding=0),
            ConvBlock(kern_1x1_2, kern_3x3, kernel_size=3, stride=1, padding=1),
            )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, kern_1x1_3, kernel_size=1, stride=1, padding=0),
            ConvBlock(kern_1x1_3, kern_5x5, kernel_size=5, stride=1, padding=2)
            )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, kern_1x1_4, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):

        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x),
                          self.branch4(x)], 1)


# Auxiliary Classifiers help reduce the vanishing gradient problem
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.aux_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.aux_conv = ConvBlock(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.aux_avgpool(x)
        x = self.aux_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
