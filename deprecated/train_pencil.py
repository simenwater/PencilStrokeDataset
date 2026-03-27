"""
train_pencil.py
用 iPad Apple Pencil 手写数据训练 14 类笔画序列分类器
(去掉 barline-single，用规则检测)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import json
import time
import math

# 14 类 (去掉 barline-single)
CLASS_NAMES = [
    'quarter-note-up', 'quarter-note-down', 'eighth-note-up', 'eighth-note-down',
    'half-note-up', 'half-note-down', 'whole-note', 'rest-quarter', 'rest-eighth',
    'treble-clef', 'sharp', 'flat', 'natural', 'dot'
]

# 原始 class_id 到新 class_id 的映射 (跳过 13=barline-single)
OLD_TO_NEW = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 14:13}


class PencilStrokeDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        strokes = sample['strokes']
        label = OLD_TO_NEW[sample['class_id']]  # 映射到新 ID

        points = []
        for stroke in strokes:
            for i, pt in enumerate(stroke):
                x, y = pt[0], pt[1]
                force = pt[2] if len(pt) > 2 else 0.5
                eos = 1.0 if i == len(stroke) - 1 else 0.0
                points.append([x, y, force, eos])

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
        return [[(p[0] - cx) / scale + 0.5, (p[1] - cy) / scale + 0.5, p[2], p[3]] for p in points]

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
            result.append([x_rot * scale + 0.5 + tx, y_rot * scale + 0.5 + ty, p[2], p[3]])
        return result


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels), lengths


class StrokeLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_classes=14, dropout=0.3):
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
            # 跳过 barline-single (class_id=13)
            if s['class_id'] != 13:
                samples.append(s)

    print(f"Loaded {len(samples)} samples (excluded barline-single)")

    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n = len(indices)
    train_end, val_end = int(n * 0.8), int(n * 0.9)

    return ([samples[i] for i in indices[:train_end]],
            [samples[i] for i in indices[train_end:val_end]],
            [samples[i] for i in indices[val_end:]])


def train_model(model, train_loader, val_loader, epochs=50, device='cuda', lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc, best_epoch = 0, 0

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0

        for seq, labels, lengths in train_loader:
            seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq, lengths), labels)
            loss.backward()
            optimizer.step()

            _, pred = model(seq, lengths).max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seq, labels, lengths in val_loader:
                seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
                _, pred = model(seq, lengths).max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1:3d}/{epochs}: Train={train_acc:.2f}%, Val={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), 'pencil_best.pt')
            print(f"  >>> New best!")

    return best_val_acc, best_epoch


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for seq, labels, lengths in test_loader:
            seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
            _, pred = model(seq, lengths).max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    total_acc = 100. * (all_preds == all_labels).sum() / len(all_labels)

    num_classes = 14
    class_correct, class_total = [0] * num_classes, [0] * num_classes
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class[CLASS_NAMES[i]] = {
                'accuracy': 100. * class_correct[i] / class_total[i],
                'correct': class_correct[i], 'total': class_total[i]
            }

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    print(f"\nTest Accuracy: {total_acc:.2f}%")
    print("\nPer-class:")
    for name, s in sorted(per_class.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {name:20s}: {s['accuracy']:6.2f}% ({s['correct']}/{s['total']})")

    print("\nWorst classes:")
    for name, s in sorted(per_class.items(), key=lambda x: x[1]['accuracy'])[:5]:
        cid = CLASS_NAMES.index(name)
        conf = [(CLASS_NAMES[j], confusion[cid][j]) for j in range(num_classes) if j != cid and confusion[cid][j] > 0]
        conf.sort(key=lambda x: -x[1])
        print(f"  {name}: {s['accuracy']:.2f}% -> {', '.join([f'{c[0]}({c[1]})' for c in conf[:3]])}")

    return {'test_accuracy': total_acc, 'per_class': per_class, 'confusion_matrix': confusion.tolist()}


def main():
    print("=" * 50)
    print("Apple Pencil Stroke Classifier (14 classes)")
    print("(excluded barline-single)")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_samples, val_samples, test_samples = load_data()
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_set = PencilStrokeDataset(train_samples, augment=True)
    val_set = PencilStrokeDataset(val_samples)
    test_set = PencilStrokeDataset(test_samples)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)

    model = StrokeLSTM(input_size=4, hidden_size=128, num_layers=2, num_classes=14, dropout=0.3)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    start = time.time()
    best_acc, best_epoch = train_model(model, train_loader, val_loader, epochs=50, device=device)
    print(f"\nTraining time: {time.time()-start:.1f}s")
    print(f"Best val: {best_acc:.2f}% (epoch {best_epoch})")

    model.load_state_dict(torch.load('pencil_best.pt'))
    results = evaluate(model, test_loader, device)
    results.update({
        'val_accuracy': best_acc,
        'best_epoch': best_epoch,
        'parameters': params,
        'num_classes': 14,
        'excluded': ['barline-single'],
        'class_names': CLASS_NAMES
    })

    with open('pencil_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: pencil_best.pt, pencil_results.json")


if __name__ == "__main__":
    main()
