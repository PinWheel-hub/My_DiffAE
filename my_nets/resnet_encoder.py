import torch
import torch.nn as nn
from torchvision import models
import torch.utils.checkpoint

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18Encoder(nn.Module):
    def __init__(self, use_checkpoint=False):
        super(ResNet18Encoder, self).__init__()
        # 加载预训练的ResNet18模型
        self.use_checkpoint = use_checkpoint
        self.features = models.resnet18(pretrained=True)
        
        # 移除ResNet18的最后一个全连接层和平均池化层
        self.features = nn.Sequential(*list(self.features.children())[:-2])
        
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 512, 1),
            nn.Flatten(),
        )

    def forward(self, x):
        if not self.use_checkpoint:
            x.requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(self._forward, (x), 
                                              preserve_rng_state=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = self.features(x)
        x = self.out(x)
        return x

if __name__ == '__main__':

    # 实例化模型
    encoder = ResNet18Encoder()

    # 假设输入数据x
    batch_size = 4
    x = torch.randn([batch_size, 3, 224, 224])

    # 获取latent表示
    latent = encoder(x)

    print(latent.shape)
    print(encoder.training)  # 现在会输出False
