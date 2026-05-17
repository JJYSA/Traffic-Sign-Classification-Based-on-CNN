import torch
# 神经网络层、损失函数、优化器等
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
#transforms模块提供多种图像变换方法，用于预处理图像数据。

from torch.utils.data import DataLoader
# 绘图工具：用于显示图片
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用Windows自带的黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号变成方框的问题
# 进度条工具：训练时显示进度
from tqdm import tqdm
import numpy as np

print("PyTorch版本:", torch.__version__)

# ===================== 2. 配置运行设备（自动选择GPU/CPU） =====================
# 如果电脑有NVIDIA显卡，用CUDA加速训练；否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"模型将在: {device} 上运行")

# ===================== 3. 设置训练超参数 =====================
BATCH_SIZE = 64    # 每次训练喂入模型的图片数量
EPOCHS = 1         # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# ===================== 4. 数据预处理 =====================
# PIL/image格式图片需要转换成张量image + 标准化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图片/ numpy数组 转换为 [0,1] 范围的张量 (通道, 高, 宽)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差，用于标准化数据
])
#这段代码本身并不指定处理图像的位置，transform只是定义了对图像的预处理操作。

data_dir = r"D:\GitHub\Traffic-Sign-Classification-Based-on-CNN\data\traffic_detector_dataset"
train_dataset = datasets.ImageFolder(f'{data_dir}\\train', transform=transform)
val_dataset = datasets.ImageFolder(f'{data_dir}\\val', transform=transform)
#看数据是三七分明明应该是训练集和测试集吧,这个测试集怎么顶着一个验证集的名字validation啊
"""
OOP真是好。torchvision.datasets.ImageFolder 是一个类（class）,它用于从磁盘加载图像数据。这里train_dataset就是基于这个类创建的一个对象（object）
torchvision.datasets.ImageFolder 的父类是 torchvision.datasets.DatasetFolder 。而 DatasetFolder 又继承自 torch.utils.data.Dataset 。
torch.utils.data.Dataset 是所有自定义数据集类的基类，DatasetFolder 在此基础上，提供了按文件夹结构加载数据的通用功能。 ImageFolder 进一步针对图像数据，做了特定优化，如默认支持图像文件格式读取 。

由 torchvision.datasets.ImageFolder 创建的对象，在深度学习流程中有诸多用途：
数据访问：支持索引操作，如 image, label = train_dataset[i]，可获取第 i 个图像及其标签，便于模型训练与评估。
数据迭代：能用于循环，遍历整个数据集，如 for image, label in train_dataset:，为模型提供训练样本。
与数据加载器配合：可作为参数传入 torch.utils.data.DataLoader，实现数据批量加载、打乱及多线程读取，提升训练效率。
"""
"""
测试代码：
image, label = train_dataset[50]
print(label)
结果发现没有找到任何文件
原来ImageFolder需要数据按类别存放，但是这个数据集直接用文件名标记类别没分
我真没招了，手动分类放了。
json也扔一块，ImageFolder自己会忽略，它只认图片。
太好了分完BUG解决了。
"""

# ===================== 5. 加载数据集=====================
# 数据加载器DataLoader能够批量加载数据，打乱顺序，加速训练
# 转换顺序：数据->张量->数据集对象dataset->数据加载器DataLoader，悲痛啊中间不能跳步一步到位
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True      # shuffle洗牌。训练集必须打乱，防止模型记忆顺序
)

test_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False     # 测试集不需要打乱
)

# ===================== 6. 构建卷积神经网络模型 =====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：输入通道（灰度图），输出通道16，卷积核3x3
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        # 池化层：最大池化，缩小图片尺寸（2x2）
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2：输入通道16，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        # 全连接层1：特征展平后输入，输出128维
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        # 全连接层2：输出8维
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        """前向传播：定义数据在模型中的流动路径"""
        # 卷积1 + 激活函数 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        # 卷积2 + 激活函数 + 池化
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平：将多维特征图展平为一维向量 (batch_size, 特征数)
        x = torch.flatten(x, 1)
        # 全连接层1 + 激活函数
        x = torch.relu(self.fc1(x))
        # 全连接层2：输出10个分类的概率
        x = self.fc2(x)
        return x

# 实例化模型，并移动到设备（GPU）
model = CNN().to(device)
print("\n 模型结构：")
print(model)

# ===================== 7. 定义损失函数和优化器 =====================
# 交叉熵损失：分类任务的标准损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器：自动调整学习率，更新模型参数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
