"""
train_rhythm_rnn.py
节奏型分类器 - 识别 beam 连接的音符组合

节奏型类别:
0: beam_8_8      两个八分音符 (1条beam)
1: beam_16_16    两个十六分音符 (2条beam)
2: beam_16_16_8  前十六后八 (部分双beam)
3: beam_8_16_16  前八后十六 (部分双beam)
4: beam_16x4     四个十六分音符 (2条beam)
5: beam_triplet  三连音 (1条beam + "3"标记)

输入: 整个节奏型的所有笔画合并
输出: 节奏型类别
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import math

RHYTHM_CLASSES = [
    'beam_8_8',       # 0: 两个八分
    'beam_16_16',     # 1: 两个十六分
    'beam_16_16_8',   # 2: 前十六后八
    'beam_8_16_16',   # 3: 前八后十六
    'beam_16x4',      # 4: 四个十六分
    'beam_triplet',   # 5: 三连音
]

MAX_SEQ_LEN = 512
INPUT_DIM = 3  # x, y, eos


class RhythmDataset(Dataset):
    """节奏型数据集 - 合并所有笔画"""
    def __init__(self, samples, augment=False):
        self.samples = samples  # [(points, label_idx), ...]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        points, label = self.samples[idx]
        points = self.normalize(points)
        if self.augment:
            points = self.augment_points(points)
        points = self.pad_fixed(points, MAX_SEQ_LEN)
        return torch.tensor(points, dtype=torch.float32), label

    def normalize(self, points):
        """归一化到 [0,1]"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w, h = max_x - min_x, max_y - min_y
        scale = max(w, h, 1.0)
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        return [[(p[0] - cx) / scale + 0.5, (p[1] - cy) / scale + 0.5, p[2]] for p in points]

    def pad_fixed(self, points, max_len):
        """固定长度填充"""
        if len(points) >= max_len:
            return points[:max_len]
        last = points[-1]
        return points + [last[:] for _ in range(max_len - len(points))]

    def augment_points(self, points):
        """数据增强: 旋转、缩放、平移"""
        scale = np.random.uniform(0.85, 1.15)
        tx, ty = np.random.uniform(-0.08, 0.08), np.random.uniform(-0.08, 0.08)
        # 节奏型允许更大旋转 (模拟不同斜率)
        angle = np.random.uniform(-15, 15) * math.pi / 180
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        result = []
        for p in points:
            x, y = p[0] - 0.5, p[1] - 0.5
            x_rot, y_rot = x * cos_a - y * sin_a, x * sin_a + y * cos_a
            result.append([x_rot * scale + 0.5 + tx, y_rot * scale + 0.5 + ty, p[2]])
        return result


class RhythmRNN(nn.Module):
    """Bi-GRU 节奏型分类器"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=6, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden_cat)


def extract_rhythms(data_path='collected/iPad.jsonl'):
    """从数据提取节奏型样本"""
    rhythms = []
    class_to_idx = {cls: i for i, cls in enumerate(RHYTHM_CLASSES)}
    stats = {cls: 0 for cls in RHYTHM_CLASSES}

    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            label = sample['label']
            strokes = sample['strokes']

            # 只处理节奏型标签
            if label not in class_to_idx:
                continue

            # 合并所有笔画
            points = []
            for stroke in strokes:
                for i, pt in enumerate(stroke):
                    x, y = pt[0], pt[1]
                    eos = 1.0 if i == len(stroke) - 1 else 0.0
                    points.append([x, y, eos])

            if len(points) < 5:  # 节奏型至少要有一定长度
                continue

            rhythm_idx = class_to_idx[label]
            rhythms.append((points, rhythm_idx))
            stats[label] += 1

    print(f"提取节奏型样本: {len(rhythms)}")
    print("各类别数量:")
    for cls, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cls:15s}: {count:4d}")

    return rhythms


def split_data(samples, train_ratio=0.8, val_ratio=0.1):
    """划分数据集"""
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n = len(indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        [samples[i] for i in indices[:train_end]],
        [samples[i] for i in indices[train_end:val_end]],
        [samples[i] for i in indices[val_end:]]
    )


def train_model(model, train_loader, val_loader, epochs=80, device='cuda', lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc, best_epoch = 0, 0

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0

        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                outputs = model(seq)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1:3d}/{epochs}: Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), 'rhythm_best.pt')
            print(f"  >>> New best!")

    return best_val_acc, best_epoch


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            outputs = model(seq)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    total_acc = 100. * (all_preds == all_labels).sum() / len(all_labels)

    print(f"\n测试准确率: {total_acc:.2f}%")
    print("\n各类别准确率:")

    num_classes = len(RHYTHM_CLASSES)
    for i, cls in enumerate(RHYTHM_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = 100. * (all_preds[mask] == i).sum() / mask.sum()
            print(f"  {cls:15s}: {acc:6.2f}% ({(all_preds[mask] == i).sum()}/{mask.sum()})")

    return {'test_accuracy': total_acc}


def main():
    print("=" * 60)
    print("节奏型分类器 RNN (6类)")
    print("beam_8_8, beam_16_16, beam_16_16_8, beam_8_16_16, beam_16x4, beam_triplet")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 提取节奏型
    rhythms = extract_rhythms('collected/iPad.jsonl')

    if len(rhythms) == 0:
        print("\n⚠️  没有节奏型数据！")
        print("请先采集以下标签的数据:")
        for cls in RHYTHM_CLASSES:
            print(f"  - {cls}")
        print("\n采集后重新运行此脚本。")
        return

    # 划分数据
    train_samples, val_samples, test_samples = split_data(rhythms)
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # 数据集
    train_set = RhythmDataset(train_samples, augment=True)
    val_set = RhythmDataset(val_samples)
    test_set = RhythmDataset(test_samples)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # 模型
    model = RhythmRNN(
        input_size=INPUT_DIM,
        hidden_size=128,
        num_layers=2,
        num_classes=len(RHYTHM_CLASSES),
        dropout=0.2
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")

    # 训练
    start = time.time()
    best_acc, best_epoch = train_model(
        model, train_loader, val_loader,
        epochs=80, device=device, lr=1e-3
    )
    print(f"\n训练时间: {time.time()-start:.1f}s")
    print(f"最佳验证: {best_acc:.2f}% (epoch {best_epoch})")

    # 测试
    model.load_state_dict(torch.load('rhythm_best.pt'))
    results = evaluate(model, test_loader, device)

    # 保存结果
    results.update({
        'val_accuracy': best_acc,
        'best_epoch': best_epoch,
        'parameters': params,
        'num_classes': len(RHYTHM_CLASSES),
        'class_names': RHYTHM_CLASSES,
    })

    with open('rhythm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n保存: rhythm_best.pt, rhythm_results.json")


if __name__ == "__main__":
    main()
