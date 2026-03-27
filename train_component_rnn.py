"""
train_component_rnn.py
组件分类器 - 11类基础形状
修复: 不用 pack_padded_sequence，固定512长度+重复填充

组件类别:
0: notehead_filled (实心符头)
1: notehead_hollow (空心符头)
2: stem (符干)
3: flag (旗帜)
4: dot (附点)
5: rest_quarter (四分休止符)
6: rest_eighth (八分休止符)
7: clef (谱号)
8: sharp (升号)
9: flat (降号)
10: natural (还原号)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import math

# 11 类组件 (升降还原号分开，休止符分开)
COMPONENT_CLASSES = [
    'notehead_filled',  # 0: 实心符头 (四分/八分/十六分音符)
    'notehead_hollow',  # 1: 空心符头 (二分/全音符)
    'stem',             # 2: 符干
    'flag',             # 3: 旗帜
    'dot',              # 4: 附点
    'rest_quarter',     # 5: 四分休止符
    'rest_eighth',      # 6: 八分休止符
    'clef',             # 7: 谱号
    'sharp',            # 8: 升号 ♯
    'flat',             # 9: 降号 ♭
    'natural',          # 10: 还原号 ♮
]

# 从原始标签拆分组件的映射
# 格式: 原标签 -> [(stroke_idx, component_class), ...]
LABEL_TO_COMPONENTS = {
    'quarter-note-up':   [(0, 'notehead_filled'), (1, 'stem')],
    'quarter-note-down': [(0, 'notehead_filled'), (1, 'stem')],
    'eighth-note-up':    [(0, 'notehead_filled'), (1, 'stem'), (2, 'flag')],
    'eighth-note-down':  [(0, 'notehead_filled'), (1, 'stem'), (2, 'flag')],
    'half-note-up':      [(0, 'notehead_hollow'), (1, 'stem')],
    'half-note-down':    [(0, 'notehead_hollow'), (1, 'stem')],
    'whole-note':        [(0, 'notehead_hollow')],
    'rest-quarter':      [(0, 'rest_quarter')],
    'rest-eighth':       [(0, 'rest_eighth')],
    'treble-clef':       [(0, 'clef')],
    'sharp':             [('all', 'sharp')],      # 合并所有笔画
    'flat':              [('all', 'flat')],       # 合并所有笔画
    'natural':           [('all', 'natural')],    # 合并所有笔画
    'dot':               [(0, 'dot')],
}

MAX_SEQ_LEN = 512
INPUT_DIM = 3  # x, y, eos (去掉 force 噪声)


class ComponentDataset(Dataset):
    """单个笔画组件数据集"""
    def __init__(self, samples, augment=False):
        self.samples = samples  # [(points, label_idx), ...]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        points, label = self.samples[idx]

        # 归一化
        points = self.normalize(points)

        # 数据增强
        if self.augment:
            points = self.augment_points(points)

        # 固定长度填充 (跟 Swift 端一致！)
        points = self.pad_fixed(points, MAX_SEQ_LEN)

        return torch.tensor(points, dtype=torch.float32), label

    def normalize(self, points):
        """归一化到 [0,1] - 跟 Swift 端完全一致"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        w = max_x - min_x
        h = max_y - min_y
        scale = max(w, h, 1.0)

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        normalized = []
        for p in points:
            x_norm = (p[0] - cx) / scale + 0.5
            y_norm = (p[1] - cy) / scale + 0.5
            eos = p[2]  # 保持 eos 标记
            normalized.append([x_norm, y_norm, eos])

        return normalized

    def pad_fixed(self, points, max_len):
        """固定长度填充 - 用最后一个点重复 (跟 Swift 端一致！)"""
        if len(points) >= max_len:
            return points[:max_len]

        # 用最后一个点重复填充
        last = points[-1]
        padding = [last[:] for _ in range(max_len - len(points))]
        return points + padding

    def augment_points(self, points):
        """数据增强: 轻微旋转、缩放、平移"""
        scale = np.random.uniform(0.9, 1.1)
        tx = np.random.uniform(-0.05, 0.05)
        ty = np.random.uniform(-0.05, 0.05)
        angle = np.random.uniform(-8, 8) * math.pi / 180
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        result = []
        for p in points:
            x, y = p[0] - 0.5, p[1] - 0.5
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            new_x = x_rot * scale + 0.5 + tx
            new_y = y_rot * scale + 0.5 + ty
            result.append([new_x, new_y, p[2]])
        return result


class ComponentRNN(nn.Module):
    """轻量 Bi-GRU 组件分类器"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=8, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        不用 pack_padded_sequence！直接处理固定长度序列
        这样训练和推理一致
        """
        # x: [batch, seq_len, input_size]
        output, hidden = self.gru(x)
        # hidden: [num_layers*2, batch, hidden_size]
        # 取最后一层的双向 hidden
        hidden_fwd = hidden[-2]  # 前向最后一层
        hidden_bwd = hidden[-1]  # 后向最后一层
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        return self.classifier(hidden_cat)


def extract_components(data_path='collected/iPad.jsonl'):
    """从原始数据提取组件样本"""
    components = []  # [(points, component_class_idx), ...]

    class_to_idx = {cls: i for i, cls in enumerate(COMPONENT_CLASSES)}
    stats = {cls: 0 for cls in COMPONENT_CLASSES}

    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            label = sample['label']
            strokes = sample['strokes']

            if label not in LABEL_TO_COMPONENTS:
                continue

            mapping = LABEL_TO_COMPONENTS[label]

            for stroke_idx, component_name in mapping:
                # 'all' 表示合并所有笔画
                if stroke_idx == 'all':
                    # 合并所有笔画
                    points = []
                    for stroke in strokes:
                        for i, pt in enumerate(stroke):
                            x, y = pt[0], pt[1]
                            eos = 1.0 if i == len(stroke) - 1 else 0.0
                            points.append([x, y, eos])
                else:
                    # 单独一笔
                    if stroke_idx >= len(strokes):
                        continue
                    stroke = strokes[stroke_idx]
                    points = []
                    for i, pt in enumerate(stroke):
                        x, y = pt[0], pt[1]
                        eos = 1.0 if i == len(stroke) - 1 else 0.0
                        points.append([x, y, eos])

                if len(points) < 3:  # 太短的笔画跳过
                    continue

                component_idx = class_to_idx[component_name]
                components.append((points, component_idx))
                stats[component_name] += 1

    print(f"提取组件样本: {len(components)}")
    print("各类别数量:")
    for cls, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cls:18s}: {count:4d}")

    return components


def split_data(components, train_ratio=0.8, val_ratio=0.1):
    """划分数据集"""
    np.random.seed(42)
    indices = np.random.permutation(len(components))

    n = len(indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = [components[i] for i in indices[:train_end]]
    val = [components[i] for i in indices[train_end:val_end]]
    test = [components[i] for i in indices[val_end:]]

    return train, val, test


def train_model(model, train_loader, val_loader, epochs=60, device='cuda', lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc, best_epoch = 0, 0

    for epoch in range(epochs):
        # 训练
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

        # 验证
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
            torch.save(model.state_dict(), 'component_best.pt')
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

    num_classes = len(COMPONENT_CLASSES)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    print(f"\n测试准确率: {total_acc:.2f}%")
    print("\n各类别准确率:")
    for i, cls in enumerate(COMPONENT_CLASSES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {cls:18s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")

    # 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    return {
        'test_accuracy': total_acc,
        'class_correct': class_correct,
        'class_total': class_total,
        'confusion_matrix': confusion.tolist()
    }


def main():
    print("=" * 60)
    print("组件分类器 RNN (11类)")
    print("升降还原号分开，休止符分开")
    print("修复: 固定512长度 + 重复填充 (跟Swift一致)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 提取组件
    components = extract_components('collected/iPad.jsonl')

    # 划分数据
    train_samples, val_samples, test_samples = split_data(components)
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # 数据集
    train_set = ComponentDataset(train_samples, augment=True)
    val_set = ComponentDataset(val_samples, augment=False)
    test_set = ComponentDataset(test_samples, augment=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # 模型 (轻量版)
    model = ComponentRNN(
        input_size=INPUT_DIM,
        hidden_size=64,
        num_layers=2,
        num_classes=len(COMPONENT_CLASSES),
        dropout=0.2
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")

    # 训练
    start = time.time()
    best_acc, best_epoch = train_model(
        model, train_loader, val_loader,
        epochs=60, device=device, lr=1e-3
    )
    print(f"\n训练时间: {time.time()-start:.1f}s")
    print(f"最佳验证: {best_acc:.2f}% (epoch {best_epoch})")

    # 测试
    model.load_state_dict(torch.load('component_best.pt'))
    results = evaluate(model, test_loader, device)

    # 保存结果
    results.update({
        'val_accuracy': best_acc,
        'best_epoch': best_epoch,
        'parameters': params,
        'num_classes': len(COMPONENT_CLASSES),
        'class_names': COMPONENT_CLASSES,
        'input_dim': INPUT_DIM,
        'max_seq_len': MAX_SEQ_LEN,
        'padding': 'repeat_last',  # 关键: 记录填充方式
    })

    with open('component_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n保存: component_best.pt, component_results.json")
    print("\n关键修复:")
    print("  - 不用 pack_padded_sequence")
    print("  - 固定 512 长度 + 最后点重复填充")
    print("  - 输入 3 维 [x, y, eos] (去掉 force)")
    print("  - 跟 Swift 推理端完全一致!")


if __name__ == "__main__":
    main()
