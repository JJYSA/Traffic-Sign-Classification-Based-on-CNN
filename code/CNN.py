import torch
# 神经网络层、损失函数、优化器等
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用Windows自带的黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号变成方框的问题
from tqdm import tqdm
import numpy as np

print("PyTorch版本:", torch.__version__)

# ===================== 2. 配置运行设备（自动选择GPU/CPU） =====================
# 如果电脑有NVIDIA显卡，用CUDA加速训练；否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"模型将在: {device} 上运行")

# ===================== 3. 设置训练超参数 =====================
BATCH_SIZE = 64    # 每次训练喂入模型的图片数量
#好家伙一个类别总共才21张训练图片
EPOCHS = 2         # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# ===================== 4. 数据预处理 =====================
# PIL/image格式图片需要转换成张量image + 标准化
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 随机旋转 ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度/对比度
    #数据增强
    transforms.ToTensor(),  # 将PIL图片/ numpy数组 转换为 [0,1] 范围的张量 (通道, 高, 宽)
    # 通用RGB归一化，不能照抄MNIST(?)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
#这段代码本身并不指定处理图像的位置，transform只是定义了对图像的预处理操作。

'''
这一步要做吗？
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#把图片像素从 [0, 1] 缩放到 [-1, 1]
'''

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
    shuffle=True
)
"""
检查确认数据标签对应正确
测试代码：
print("数据集类别：", train_dataset.classes)
print("类别映射：", train_dataset.class_to_idx)
print("训练集图片数量：", len(train_dataset))
才168张，太少了，得数据增强
"""

# ===================== 6. 构建卷积神经网络模型 =====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 1)
        #如果in_feature用三个数相乘，一定要加padding，不然矩阵维数对不上
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 7, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 7, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 7, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 7, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        # 全连接层：特征展平后输入
        self.fc1 = nn.Linear(58368, 1024)
        #原本236800输入也太离谱了
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 8)
        #原本的全连接层跨度太离谱了。但是加全连接层会明显增加训练时间。而且loss还是不动啊。
        '''
        该用多少层?
        借鉴AlexNet 
        输入：224×224×3
        1. 卷积层总数：5 层 Conv
        2. 全连接层总数：3 层 FC
        '''
    def forward(self, x):
        """前向传播：定义数据在模型中的流动路径"""
        # 卷积 + 激活函数 + 池化
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.pool5(torch.relu(self.conv5(x)))
        # 展平：将多维特征图展平为一维向量 (batch_size, 特征数)
        x = torch.flatten(x, 1)
        # 全连接层1 + 激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # 全连接层输出8个分类的概率
        x = self.fc4(x)
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

# ===================== 8. 模型训练 =====================
def train_model(model, loader, criterion, optimizer, epochs, device):
    print("\n开始训练...")
    # 遍历每一轮训练
    for epoch in range(EPOCHS):
        running_loss = 0.0  # 记录每轮的损失值
        # 遍历批次数据，带进度条
        pbar = tqdm(loader, desc=f"第{epoch+1}/{EPOCHS}轮")
        for images, labels in pbar:
            # 将数据移动到设备
            images, labels = images.to(device), labels.to(device)
            # 1. 梯度清零：防止上一批次的梯度累积
            optimizer.zero_grad()
            # 2. 前向传播：模型输出预测结果
            outputs = model(images)
            # 3. 计算损失：预测值和真实值的误差
            loss = criterion(outputs, labels)
            # 4. 反向传播：计算梯度
            loss.backward()
            # 5. 更新参数：优化器调整模型权重
            optimizer.step()

            # 累计损失
            running_loss += loss.item()
            # 更新进度条显示损失
            pbar.set_postfix({"损失值": f"{loss.item():.3f}"})

        # 打印每轮的平均损失
        avg_loss = running_loss / len(loader)
        print(f"第{epoch+1}轮训练完成，平均损失: {avg_loss:.3f}")

    print("\n训练完成！")
    return model

# 开始训练
model = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

# 保存训练好的模型
torch.save(model.state_dict(), "model.pth")
print("模型已保存为: model.pth")

# ===================== 9. 模型推理 + 可视化测试集结果 =====================
def infer_and_show_test_images(model, loader, device):
    """
    加载训练好的模型，对测试集图片推理
    显示6张图片 + 打印推理结果
    """
    print("\n开始推理测试集图片...")
    # 加载模型（如果需要重新加载，取消注释）
    # model.load_state_dict(torch.load("mnist_digit_model.pth"))
    # 切换为评估模式：关闭dropout、batchnorm等训练专用层
    model.eval()

    # 获取一个批次的6张测试图片
    data_iter = iter(loader)
    images, true_labels = next(data_iter)
    images = images.to(device)

    # 关闭梯度计算：推理时不需要计算梯度，加速+节省内存
    with torch.no_grad():
        outputs = model(images)
        # 获取预测结果：取输出概率最大的类别
        pred_labels = torch.argmax(outputs, dim=1)

    # 转换为numpy数组，方便绘图
    images = images.cpu().numpy()
    true_labels = true_labels.numpy()
    pred_labels = pred_labels.cpu().numpy()

    # ===================== 可视化6张测试集图片 + 推理结果 =====================
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        # 显示图片
        img = images[i].squeeze()
        # 调整维度顺序
        img = img.transpose((1, 2, 0))
        #img是numpy.ndarray类型
        '''#反归一化，归一化，转了[-1, 1]后得做这一步逆运算，否则会变得很昏暗
        img = (img * 0.5) + 0.5
        '''
        plt.imshow(img)
        #因为plt.imshow() 期望的图像数据形状是 (height, width, channels)，而原始传入的图像img数据形状是 (channels, height, width)
        # 标题：真实值/预测值
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        plt.title(f"真:{true_labels[i]}\n预:{pred_labels[i]}", color=color)

    plt.tight_layout()
    plt.suptitle('测试集推理结果（绿色=正确，红色=错误）', fontsize=16)
    plt.show()

    # ===================== 打印推理结果=====================
    print("测试集6张图片推理结果：\n")
    correct = 0
    for i in range(6):
        res = f"第{i+1}张 → 真实: {true_labels[i]}, 推理: {pred_labels[i]}"
        if true_labels[i] == pred_labels[i]:
            res += " ✅ 正确"
            correct +=1
        else:
            res += " ❌ 错误"
        print(res)
    print(f"\n6张图片准确率: {correct/6*100:.2f}%")

# 调用推理函数
infer_and_show_test_images(model, test_loader, device)