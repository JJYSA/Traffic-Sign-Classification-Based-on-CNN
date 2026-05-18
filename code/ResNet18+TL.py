import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
import random
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用Windows自带的黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号变成方框的问题

# ===================== 配置运行设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

RESULT_SAVE_PATH = r"D:\GitHub\Traffic-Sign-Classification-Based-on-ResNet18-TransferLearning\results"  #请替换您自己的路径
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

# ===================== 超参数 =====================
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0001

data_dir = r"D:\GitHub\Traffic-Sign-Classification-Based-on-ResNet18-TransferLearning\data\traffic_detector_dataset"

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

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
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ===================== 数据集加载 =====================
train_dataset = datasets.ImageFolder(f'{data_dir}\\train', transform=train_transform)
test_dataset = datasets.ImageFolder(f'{data_dir}\\val', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = train_dataset.classes

# ===================== 迁移学习 =====================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 解冻最后1个大模块（layer4）进行微调，即最后17组参数
for name, param in list(model.named_parameters())[-17:]:
    param.requires_grad = True

"""
确认ResNet18网络结构
调试代码：
print("==== ResNet18 所有参数（共{}组）====".format(len(list(model.named_parameters()))))
for idx, (name, param) in enumerate(model.named_parameters()):
    print(idx, name)
"""

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 8)
)

model = model.to(device)

# ===================== 训练 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_acc = 0.0
best_model = None

for epoch in range(EPOCHS):
    # -------- 训练 --------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"第{epoch + 1}/{EPOCHS}轮")
    for imgs, labs in pbar:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        train_correct += (pred == labs).sum().item()
        train_total += labs.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / train_total

    # -------- 验证 --------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labs).item()
            pred = torch.argmax(outputs, dim=1)
            val_correct += (pred == labs).sum().item()
            val_total += labs.size(0)

    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = val_correct / val_total

    # -------- 记录 & 打印 --------
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    train_acc_list.append(avg_train_acc)
    val_acc_list.append(avg_val_acc)

    print(f"训练损失:{avg_train_loss:.3f} | 验证损失:{avg_val_loss:.3f} | 训练准确率:{avg_train_acc:.2%} | 验证准确率:{avg_val_acc:.2%}")

    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        best_model = model.state_dict()

#保存最优模型
model.load_state_dict(best_model)
torch.save(model.state_dict(), "best_model.pth")
print(f"\n最高测试准确率: {best_acc:.2%}")

#保存曲线 & 日志
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(train_loss_list, label='训练损失', c='red')
plt.plot(val_loss_list, label='验证损失', c='blue')
plt.title('损失曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(train_acc_list, label='训练准确率', c='red')
plt.plot(val_acc_list, label='验证准确率', c='blue')
plt.title('准确率曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_SAVE_PATH, "curve.png"), dpi=300)
plt.close()

with open(os.path.join(RESULT_SAVE_PATH, "log.csv"), "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["轮数", "训练损失", "验证损失", "训练准确率", "验证准确率"])
    for i in range(EPOCHS):
        writer.writerow([i + 1, round(train_loss_list[i], 4), round(val_loss_list[i], 4), round(train_acc_list[i], 4),
                         round(val_acc_list[i], 4)])

# ===================== 预测演示=====================
def show_random_val_prediction():
    model.eval()
    indices = random.sample(range(len(test_dataset)), 6)

    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        img = img.to(device)

        with torch.no_grad():
            output = model(img.unsqueeze(0))
            pred = torch.argmax(output).item()

        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = img_np * 0.5 + 0.5
        img_np = img_np.clip(0, 1)

        true_class = class_names[label]
        pred_class = class_names[pred]

        plt.subplot(2, 3, i + 1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"真实: {true_class}\n预测: {pred_class}", fontsize=12)

    plt.suptitle("预测演示", fontsize=16)
    plt.tight_layout()
    plt.show()


# 运行预测演示
show_random_val_prediction()