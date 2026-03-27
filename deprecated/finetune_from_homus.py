"""
finetune_from_homus.py
用 HOMUS RNN 98%+ 作为基座，用 iPad Apple Pencil 数据微调
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import json
import time
import math
import os

# 14 类 (去掉 barline-single)
CLASS_NAMES = [
    'quarter-note-up', 'quarter-note-down', 'eighth-note-up', 'eighth-note-down',
    'half-note-up', 'half-note-down', 'whole-note', 'rest-quarter', 'rest-eighth',
    'treble-clef', 'sharp', 'flat', 'natural', 'dot'
]

# 原始 class_id 到新 class_id 的映射 (跳过 13=barline-single)
OLD_TO_NEW = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 14:13}


class PencilStrokeDataset(Dataset):
    """iPad 数据集 - 只用 (x, y, eos)，不用 force"""
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        strokes = sample['strokes']
        label = OLD_TO_NEW[sample['class_id']]

        # 只用 x, y, eos (3维)，和 HOMUS 基座一致
        points = []
        for stroke in strokes:
            for i, pt in enumerate(stroke):
                x, y = pt[0], pt[1]
                eos = 1.0 if i == len(stroke) - 1 else 0.0
                points.append([x, y, eos])  # 3 维，不要 force

        points = self.normalize(points)
        if self.augment:
            points = self.augment_points(points)

        return torch.tensor(points, dtype=torch.float32), label

    def normalize(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        return [[(p[0] - cx) / scale + 0.5, (p[1] - cy) / scale + 0.5, p[2]] for p in points]

    def augment_points(self, points):
        scale = np.random.uniform(0.85, 1.15)
        tx = np.random.uniform(-0.1, 0.1)
        ty = np.random.uniform(-0.1, 0.1)
        angle = np.random.uniform(-10, 10) * math.pi / 180
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        result = []
        for p in points:
            x, y = p[0] - 0.5, p[1] - 0.5
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            result.append([x_rot * scale + 0.5 + tx, y_rot * scale + 0.5 + ty, p[2]])
        return result


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels), lengths


class StrokeLSTM(nn.Module):
    """和 HOMUS 基座一样的结构"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden)


def load_data(data_path='collected/iPad.jsonl'):
    samples = []
    with open(data_path) as f:
        for line in f:
            s = json.loads(line)
            if s['class_id'] != 13:  # 跳过 barline-single
                samples.append(s)

    print(f"Loaded {len(samples)} samples (excluded barline-single)")

    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n = len(indices)
    train_end, val_end = int(n * 0.8), int(n * 0.9)

    return ([samples[i] for i in indices[:train_end]],
            [samples[i] for i in indices[train_end:val_end]],
            [samples[i] for i in indices[val_end:]])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for seq, labels, lengths in loader:
        seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(seq, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seq, labels, lengths in loader:
            seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
            _, pred = model(seq, lengths).max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return 100. * correct / total, np.array(all_preds), np.array(all_labels)


def main():
    print("=" * 60)
    print("Finetune HOMUS RNN -> iPad Apple Pencil Data")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 加载 HOMUS 基座模型 (32 类)
    base_model_path = "../MusicSymbolTrainer/exports/rnn_best.pt"
    if not os.path.exists(base_model_path):
        # 尝试其他路径
        alt_paths = [
            "../MusicSymbolTrainer/rnn_best.pt",
            "../../MusicSymbolTrainer/exports/rnn_best.pt",
            "../../MusicSymbolTrainer/rnn_best.pt"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                base_model_path = p
                break
        else:
            print("ERROR: Cannot find HOMUS base model (rnn_best.pt)")
            print("Please ensure the model exists at MusicSymbolTrainer/exports/rnn_best.pt")
            return

    print(f"Base model: {base_model_path}")

    # 加载基座模型
    base_model = StrokeLSTM(input_size=3, hidden_size=128, num_layers=2, num_classes=32)
    base_model.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    print("Loaded HOMUS base model (32 classes)")

    # 创建新模型 (14 类)
    model = StrokeLSTM(input_size=3, hidden_size=128, num_layers=2, num_classes=14)

    # 复制 LSTM 权重
    model.lstm.load_state_dict(base_model.lstm.state_dict())
    print("Copied LSTM weights from base model")

    # 复制 classifier 第一层 (256 维)
    model.classifier[0].load_state_dict(base_model.classifier[0].state_dict())
    print("Copied classifier[0] (256-dim hidden layer)")

    # 最后一层随机初始化 (256 -> 14)
    print("Last layer (256->14) randomly initialized")

    # 冻结 LSTM
    for param in model.lstm.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable (classifier only)")

    # 加载数据
    train_samples, val_samples, test_samples = load_data()
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_set = PencilStrokeDataset(train_samples, augment=True)
    val_set = PencilStrokeDataset(val_samples)
    test_set = PencilStrokeDataset(test_samples)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: 冻结 LSTM，只训练分类头 (20 epochs)
    print("\n" + "=" * 60)
    print("Phase 1: Frozen LSTM, train classifier only (20 epochs)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_acc, best_epoch = 0, 0
    start = time.time()

    for epoch in range(20):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:2d}/20: Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), 'pencil_finetuned.pt')
            print(f"  >>> New best!")

    phase1_time = time.time() - start
    print(f"\nPhase 1 done: {phase1_time:.1f}s, best val={best_val_acc:.2f}% (epoch {best_epoch})")

    # Phase 2: 解冻 LSTM，全模型微调 (10 epochs) - 如果 Phase 1 不够好
    if best_val_acc < 99.0:
        print("\n" + "=" * 60)
        print("Phase 2: Unfrozen LSTM, full model finetune (10 epochs)")
        print("=" * 60)

        # 解冻 LSTM
        for param in model.lstm.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        # 加载 Phase 1 最佳模型
        model.load_state_dict(torch.load('pencil_finetuned.pt'))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        start = time.time()
        for epoch in range(10):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, _, _ = evaluate(model, val_loader, device)
            scheduler.step()

            print(f"Epoch {epoch+1:2d}/10: Train={train_acc:.2f}%, Val={val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, 20 + epoch + 1
                torch.save(model.state_dict(), 'pencil_finetuned.pt')
                print(f"  >>> New best!")

        phase2_time = time.time() - start
        print(f"\nPhase 2 done: {phase2_time:.1f}s, best val={best_val_acc:.2f}% (epoch {best_epoch})")

    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(torch.load('pencil_finetuned.pt'))
    test_acc, preds, labels = evaluate(model, test_loader, device)

    # 每类准确率
    num_classes = 14
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for pred, label in zip(preds, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class[CLASS_NAMES[i]] = {
                'accuracy': 100. * class_correct[i] / class_total[i],
                'correct': class_correct[i],
                'total': class_total[i]
            }

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nPer-class:")
    for name, s in sorted(per_class.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {name:20s}: {s['accuracy']:6.2f}% ({s['correct']}/{s['total']})")

    # 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(preds, labels):
        confusion[label][pred] += 1

    # 保存结果
    results = {
        'method': 'finetune_from_homus',
        'base_model': base_model_path,
        'val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'best_epoch': best_epoch,
        'per_class': per_class,
        'confusion_matrix': confusion.tolist(),
        'parameters': total_params,
        'num_classes': 14,
        'input_size': 3,
        'class_names': CLASS_NAMES
    }

    with open('pencil_finetuned_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSaved: pencil_finetuned.pt, pencil_finetuned_results.json")

    # 对比从零训练
    print("\n" + "=" * 60)
    print("Comparison with scratch training:")
    print("=" * 60)
    scratch_results = 'pencil_results.json'
    if os.path.exists(scratch_results):
        with open(scratch_results) as f:
            scratch = json.load(f)
        print(f"  From scratch: val={scratch['val_accuracy']:.2f}%, test={scratch['test_accuracy']:.2f}%")
        print(f"  Finetuned:    val={best_val_acc:.2f}%, test={test_acc:.2f}%")
        diff = test_acc - scratch['test_accuracy']
        print(f"  Improvement:  {diff:+.2f}%")


if __name__ == "__main__":
    main()
