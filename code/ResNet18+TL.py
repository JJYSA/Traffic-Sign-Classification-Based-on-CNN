import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

RESULT_SAVE_PATH = r"D:\GitHub\Traffic-Sign-Classification-Based-on-CNN\results"
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

# ===================== 超参数 =====================
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0001
data_dir = r"D:\GitHub\Traffic-Sign-Classification-Based-on-CNN\data\traffic_detector_dataset"

train_loss_list = []
test_acc_list = []

# ===================== 训练集：数据增强 =====================
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
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

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

# ===================== 训练（保存最高准确率） =====================
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
            imgs = imgs.to(device)
            labs = labs.to(device)
            pred = torch.argmax(model(imgs), dim=1)
            correct += (pred == labs).sum().item()
            total += labs.size(0)

    acc = correct / total
    print(f"训练损失: {avg_train_loss:.3f} | 测试准确率: {acc:.2%}")

    train_loss_list.append(avg_train_loss)
    test_acc_list.append(acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model.state_dict()

model.load_state_dict(best_model)
torch.save(model.state_dict(), "model.pth")

print("\n" + "="*50)
print(f" 测试集准确率：{best_acc:.2%}")
print("="*50)

# ===================== 保存训练图表 & 数据 =====================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_loss_list, color='#e74c3c', linewidth=2, label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), test_acc_list, color='#2980b9', linewidth=2, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('测试集准确率曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_SAVE_PATH, "loss & accuracy.png"), dpi=300)
plt.close()

# 保存CSV
with open(os.path.join(RESULT_SAVE_PATH, "log.csv"), 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["轮数", "训练损失", "测试准确率"])
    for i in range(EPOCHS):
        writer.writerow([i+1, round(train_loss_list[i],4), round(test_acc_list[i],4)])

# 保存最优结果
with open(os.path.join(RESULT_SAVE_PATH, "best.txt"), 'w', encoding='utf-8') as f:
    f.write(f"最佳测试集准确率: {best_acc:.2%}\n")
    f.write(f"总轮数: {EPOCHS}\n")

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

show6()