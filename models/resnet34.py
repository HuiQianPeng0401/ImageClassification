#coding:utf8

from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F

"""
1. `from .basic_module import BasicModule`：
    - 这一行导入了自定义的 `BasicModule` 类，该类是您自己定义的神经网络模型的基类。
2. `from torch import nn`：
    - 这一行导入了PyTorch中的 `nn` 模块，该模块包含了神经网络的各种层和函数。
3. `from torch.nn import functional as F`：
    - 这一行导入了PyTorch中的 `functional` 模块，并将其重命名为 `F`。
    - `functional` 模块包含了一些与层无关的函数，例如激活函数、损失函数等。
4. `class ResidualBlock(nn.Module):`：
    - 这是一个Python类的定义，它继承自PyTorch中的 `nn.Module` 类。
    - `ResidualBlock` 是您自己定义的一个残差块（Residual Block）模型。
5. `def __init__(self, inchannel, outchannel, stride=1, shortcut=None):`：
    - 这是类的构造函数（初始化方法）。
    - 它接受四个参数：输入通道数 `inchannel`、输出通道数 `outchannel`、步长 `stride` 和跳跃连接（shortcut）。
6. `self.left = nn.Sequential(...)`：
    - 这一行定义了一个包含多个层的序列（Sequential）。
    - `self.left` 是残差块的左侧部分，包括两个卷积层、Batch Normalization 和 ReLU 激活函数。
7. `self.right = shortcut`：
    - 这一行设置了残差块的右侧部分，即跳跃连接（shortcut）。
    - 如果没有提供跳跃连接，`self.right` 将为 `None`。
8. `def forward(self, x):`：
    - 这是模型的前向传播方法。
    - 输入 `x` 经过左侧部分的卷积层，得到 `out`。
    - 如果提供了跳跃连接，将其加到 `out` 上，得到最终的输出。
9. `return F.relu(out)`：
    - 这一行应用了ReLU激活函数，将 `out` 中的负值变为零。
总之，这段代码定义了一个残差块（Residual Block）模型，它由左侧部分和可选的跳跃连接组成。在前向传播中，输入经过左侧部分的卷积层，然后与跳跃连接相加，最终应用ReLU激活函数。
"""

class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right=shortcut

    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)


class ResNet34(BasicModule):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """
    def __init__(self,num_classes=2):
        super(ResNet34,self).__init__()
        self.model_name='resnet34'
        # 前几层：图像转换
        self.pre=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1=self._make_layer(64,64,3,1,is_shortcut=False)
        self.layer2=self._make_layer(64,128,4,2)
        self.layer3=self._make_layer(128,256,6,2)
        self.layer4=self._make_layer(256,512,3,2)
        # 分类用的全连接
        self.fc=nn.Linear(512,num_classes)
    def _make_layer(self,inchannel,outchannel,block_num,stride,is_shortcut=True):
        """
        构建layer，包含多个residual block
        """
        if is_shortcut:
            shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            shortcut=None
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.pre(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=F.avg_pool2d(x,7)
        x=x.view(x.size(0),-1)
        return self.fc(x)