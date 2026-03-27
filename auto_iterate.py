"""
组件分类器自动迭代优化
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import math

MAX_SEQ_LEN = 512
INPUT_DIM = 3

COMPONENT_CLASSES = [
    'notehead_filled', 'notehead_hollow', 'stem', 'flag',
    'dot', 'rest_quarter', 'rest_eighth', 'clef',
    'sharp', 'flat', 'natural',
]

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


class ComponentDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
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
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w, h = max_x - min_x, max_y - min_y
        scale = max(w, h, 1.0)
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        return [[(p[0] - cx) / scale + 0.5, (p[1] - cy) / scale + 0.5, p[2]] for p in points]

    def pad_fixed(self, points, max_len):
        if len(points) >= max_len:
            return points[:max_len]
        last = points[-1]
        return points + [last[:] for _ in range(max_len - len(points))]

    def augment_points(self, points):
        scale = np.random.uniform(0.85, 1.15)
        tx, ty = np.random.uniform(-0.08, 0.08), np.random.uniform(-0.08, 0.08)
        angle = np.random.uniform(-12, 12) * math.pi / 180
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        result = []
        for p in points:
            x, y = p[0] - 0.5, p[1] - 0.5
            x_rot, y_rot = x * cos_a - y * sin_a, x * sin_a + y * cos_a
            result.append([x_rot * scale + 0.5 + tx, y_rot * scale + 0.5 + ty, p[2]])
        return result


class ComponentRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=8, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden_cat)


def extract_components(data_path='collected/iPad.jsonl'):
    components = []
    class_to_idx = {cls: i for i, cls in enumerate(COMPONENT_CLASSES)}
    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            label = sample['label']
            strokes = sample['strokes']
            if label not in LABEL_TO_COMPONENTS:
                continue
            for stroke_idx, component_name in LABEL_TO_COMPONENTS[label]:
                # 'all' 表示合并所有笔画
                if stroke_idx == 'all':
                    points = []
                    for stroke in strokes:
                        for i, pt in enumerate(stroke):
                            eos = 1.0 if i == len(stroke) - 1 else 0.0
                            points.append([pt[0], pt[1], eos])
                else:
                    if stroke_idx >= len(strokes):
                        continue
                    stroke = strokes[stroke_idx]
                    points = [[pt[0], pt[1], 1.0 if i == len(stroke)-1 else 0.0] for i, pt in enumerate(stroke)]
                if len(points) >= 3:
                    components.append((points, class_to_idx[component_name]))
    return components


def split_data(components):
    np.random.seed(42)
    indices = np.random.permutation(len(components))
    n = len(indices)
    train_end, val_end = int(n * 0.8), int(n * 0.9)
    return ([components[i] for i in indices[:train_end]],
            [components[i] for i in indices[train_end:val_end]],
            [components[i] for i in indices[val_end:]])


def train_config(config, train_samples, val_samples, test_samples, device):
    train_set = ComponentDataset(train_samples, augment=True)
    val_set = ComponentDataset(val_samples)
    test_set = ComponentDataset(test_samples)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    model = ComponentRNN(
        input_size=INPUT_DIM, hidden_size=config['hidden'], num_layers=config['layers'],
        num_classes=len(COMPONENT_CLASSES), dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, best_test = 0, 0
    best_state = None

    for epoch in range(config['epochs']):
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                _, pred = model(seq).max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Test
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for seq, labels in test_loader:
                    seq, labels = seq.to(device), labels.to(device)
                    _, pred = model(seq).max(1)
                    test_total += labels.size(0)
                    test_correct += pred.eq(labels).sum().item()
            best_test = 100. * test_correct / test_total

    params = sum(p.numel() for p in model.parameters())
    return best_val, best_test, params, best_state, config


def main():
    print('=' * 60)
    print('组件分类器 自动迭代优化')
    print('=' * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    components = extract_components()
    train_samples, val_samples, test_samples = split_data(components)
    print(f'数据: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}')
    print()

    configs = [
        {'name': 'baseline',      'hidden': 64,  'layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 60,  'batch_size': 32},
        {'name': 'larger',        'hidden': 128, 'layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 80,  'batch_size': 32},
        {'name': 'deeper',        'hidden': 64,  'layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'epochs': 80,  'batch_size': 32},
        {'name': 'larger_deeper', 'hidden': 128, 'layers': 3, 'dropout': 0.3, 'lr': 8e-4, 'epochs': 100, 'batch_size': 32},
        {'name': 'wide',          'hidden': 192, 'layers': 2, 'dropout': 0.3, 'lr': 8e-4, 'epochs': 100, 'batch_size': 32},
    ]

    results = []
    best_overall = None
    best_test_acc = 0

    for cfg in configs:
        print(f"训练 {cfg['name']}: hidden={cfg['hidden']}, layers={cfg['layers']}, epochs={cfg['epochs']}")
        val_acc, test_acc, params, state, config = train_config(cfg, train_samples, val_samples, test_samples, device)
        results.append((cfg['name'], val_acc, test_acc, params))
        print(f"  -> Val={val_acc:.2f}%, Test={test_acc:.2f}%, Params={params:,}")
        print()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_overall = (state, config)

    print('=' * 60)
    print('迭代结果汇总')
    print('=' * 60)
    print(f"{'配置':<15} {'Val%':>8} {'Test%':>8} {'参数量':>10}")
    print('-' * 45)
    for name, val, test, params in sorted(results, key=lambda x: -x[2]):
        print(f'{name:<15} {val:>8.2f} {test:>8.2f} {params:>10,}')

    # 保存最佳模型
    if best_overall:
        state, config = best_overall
        torch.save(state, 'component_best.pt')
        print(f"\n最佳模型已保存: component_best.pt")
        print(f"配置: hidden={config['hidden']}, layers={config['layers']}")

        # 保存配置
        with open('component_results.json', 'w') as f:
            json.dump({
                'best_config': config['name'],
                'hidden_size': config['hidden'],
                'num_layers': config['layers'],
                'test_accuracy': best_test_acc,
                'all_results': [{'name': n, 'val': v, 'test': t, 'params': p} for n, v, t, p in results]
            }, f, indent=2)


if __name__ == '__main__':
    main()
