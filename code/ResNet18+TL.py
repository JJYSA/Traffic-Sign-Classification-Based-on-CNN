import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

# ===================== 超参数 =====================
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.0001
data_dir = r"D:\GitHub\Traffic-Sign-Classification-Based-on-CNN\data\traffic_detector_dataset"

# ===================== 数据增强（温和，不破坏特征） =====================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ===================== 加载数据集 =====================
train_dataset = datasets.ImageFolder(f'{data_dir}\\train', transform=train_transform)
test_dataset = datasets.ImageFolder(f'{data_dir}\\val', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 迁移学习：ResNet18 小样本神器 =====================
model = models.resnet18(pretrained=True)

# 冻结主干网络，只训练最后一层（防止过拟合）
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层为8分类
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 8)
model = model.to(device)

# ===================== 优化器 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# ===================== 训练 =====================
def train():
    best_acc = 0
    best_model = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"第{epoch + 1}轮")
        for imgs, labs in pbar:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # 测试集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                pred = torch.argmax(model(imgs), dim=1)
                correct += (pred == labs).sum().item()
                total += labs.size(0)

        acc = correct / total
        print(f"训练损失:{avg_train:.3f} | 测试准确率:{acc:.2%}")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "best_transfer_model.pth")
    print(f"\n最佳测试准确率: {best_acc:.2%}")


# ===================== 6张图可视化 =====================
def show6():
    model.eval()
    imgs, labs = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        pred = torch.argmax(model(imgs), dim=1)

    imgs = imgs.cpu().numpy()
    labs = labs.numpy()
    pred = pred.cpu().numpy()

    plt.figure(figsize=(12, 8))
    corr = 0
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        img = imgs[i].transpose(1, 2, 0)
        img = (img * 0.5 + 0.5).clip(0, 1)
        plt.imshow(img)
        color = "green" if labs[i] == pred[i] else "red"
        if labs[i] == pred[i]: corr += 1
        plt.title(f"真:{labs[i]}\n预:{pred[i]}", color=color)

    plt.tight_layout()
    plt.show()
    print(f"6张图准确率: {corr / 6:.2%}")


# ===================== 执行 =====================
train()
show6()