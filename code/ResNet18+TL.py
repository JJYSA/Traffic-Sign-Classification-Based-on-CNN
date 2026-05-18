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
EPOCHS = 30
LEARNING_RATE = 0.0001
data_dir = r"D:\GitHub\Traffic-Sign-Classification-Based-on-CNN\data\traffic_detector_dataset"

# ===================== 训练集：超强有效增强（10倍扩增） =====================
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ===================== 测试集：不增强 =====================
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ===================== 数据集加载 =====================
train_dataset = datasets.ImageFolder(f'{data_dir}\\train', transform=train_transform)
test_dataset = datasets.ImageFolder(f'{data_dir}\\val', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 迁移学习 + 顶层微调（小样本冲高分核心） =====================
model = models.resnet18(pretrained=True)

# 解冻最后2层，进行微小微调（强力提分关键）
for name, param in list(model.named_parameters())[-30:]:
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 8)
)

model = model.to(device)

# ===================== 优化器 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ===================== 训练（自动保存最高准确率） =====================
best_acc = 0.0
best_model = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"第{epoch+1}/{EPOCHS}轮")
    for imgs, labs in pbar:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

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
    print(f"训练损失: {avg_train_loss:.3f} | 测试准确率: {acc:.2%}")

    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        best_model = model.state_dict()

# 加载最优结果
model.load_state_dict(best_model)
torch.save(model.state_dict(), "final_85+_model.pth")

# ===================== 最终输出 =====================
print("\n" + "="*50)
print(f" 测试集准确率：{best_acc:.2%}")
print("="*50)

# ===================== 可视化6张图 =====================
def show6():
    model.eval()
    imgs, labs = next(iter(test_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        pred = torch.argmax(model(imgs), dim=1)

    imgs = imgs.cpu().numpy()
    labs = labs.numpy()
    pred = pred.cpu().numpy()

    plt.figure(figsize=(12,8))
    corr = 0
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.xticks([])
        plt.yticks([])
        img = imgs[i].transpose(1,2,0)
        img = (img * 0.5 + 0.5).clip(0,1)
        plt.imshow(img)
        color = "green" if labs[i]==pred[i] else "red"
        if labs[i]==pred[i]: corr +=1
        plt.title(f"真:{labs[i]}\n预:{pred[i]}", color=color)
    plt.tight_layout()
    plt.show()
    print(f"6张图准确率: {corr/6:.2%}")

show6()