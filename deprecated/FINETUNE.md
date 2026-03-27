# 微调指令

## 目标

用 HOMUS 训练的 98% RNN 做基座，用 iPad 手写数据微调，让模型适应 Apple Pencil 笔迹风格。

## 步骤

### 1. 准备基座模型

从 MusicSymbolTrainer 拿 HOMUS RNN 权重：
```python
# 基座模型路径（MusicSymbolTrainer 项目里的）
BASE_MODEL = "exports/rnn_best.pt"  # HOMUS Bi-LSTM, 97%+, 605K params
```

### 2. 准备 iPad 数据

```python
import json

samples = []
with open("collected/iPad.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))
# ~679 samples, 15 classes
```

### 3. 类别映射

HOMUS 有 32 类，iPad 数据只有 15 类。微调时冻结 LSTM 层，只替换分类头：

```python
import torch
import torch.nn as nn

class StrokeLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x, lengths=None):
        output, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden)

# 加载 HOMUS 基座（32 类）
base = StrokeLSTM(input_size=3, num_classes=32)
base.load_state_dict(torch.load(BASE_MODEL, map_location="cpu"))

# 替换分类头为 15 类
new_model = StrokeLSTM(input_size=3, num_classes=15)
# 复制 LSTM 权重（冻结）
new_model.lstm.load_state_dict(base.lstm.state_dict())
# 复制 classifier 第一层（256 维，通用特征）
new_model.classifier[0].load_state_dict(base.classifier[0].state_dict())
# 最后一层（256→15）随机初始化，需要训练
```

### 4. 数据预处理

注意：HOMUS RNN 的 input_size=3 (x, y, eos)，不是 4。
iPad 数据有 force，微调时**去掉 force，只用 x, y, eos**，保持跟基座一致。

```python
def sample_to_sequence(sample):
    points = []
    for stroke in sample["strokes"]:
        for i, pt in enumerate(stroke):
            x, y = pt[0], pt[1]
            eos = 1.0 if i == len(stroke) - 1 else 0.0
            points.append([x, y, eos])  # 3 维，不要 force
    return normalize(points)

def normalize(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    scale = max(max(xs)-min(xs), max(ys)-min(ys), 1)
    return [[(p[0]-cx)/scale + 0.5, (p[1]-cy)/scale + 0.5, p[2]] for p in points]
```

### 5. 微调训练

```python
# 冻结 LSTM，只训练分类头
for param in new_model.lstm.parameters():
    param.requires_grad = False

# 小 LR，少 epoch
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, new_model.parameters()),
    lr=1e-4  # 比从零训练小 10 倍
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 只需 20 epoch（基座已经很好了）
for epoch in range(20):
    train(...)
    val_acc = evaluate(...)
    if val_acc > best:
        torch.save(new_model.state_dict(), "pencil_finetuned.pt")

# 如果效果不够好，解冻 LSTM 再跑 10 epoch（全模型微调）
for param in new_model.lstm.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-5)  # 更小的 LR
for epoch in range(10):
    train(...)
```

### 6. 15 类映射

iPad 数据的 15 类：
| ID | Label |
|----|-------|
| 0 | quarter-note-up |
| 1 | quarter-note-down |
| 2 | eighth-note-up |
| 3 | eighth-note-down |
| 4 | half-note-up |
| 5 | half-note-down |
| 6 | whole-note |
| 7 | rest-quarter |
| 8 | rest-eighth |
| 9 | treble-clef |
| 10 | sharp |
| 11 | flat |
| 12 | natural |
| 13 | barline-single |
| 14 | dot |

### 7. 导出

```python
results = {
    "method": "finetune_from_homus",
    "base_model": "rnn_best.pt (HOMUS 97%+)",
    "val_accuracy": best_val_acc,
    "test_accuracy": test_acc,
    "per_class": {...},
    "parameters": 602127,
}
json.dump(results, open("pencil_finetuned_results.json", "w"), indent=2)
```

保存 `pencil_finetuned.pt` 和 `pencil_finetuned_results.json`，push 到 GitHub。

### 关键点

- **input_size=3**（不是 4），跟 HOMUS 基座一致
- **先冻 LSTM 只训分类头**（20 epoch），不够再解冻全模型（10 epoch）
- **LR 要小**：冻结时 1e-4，解冻时 1e-5
- **数据增强照常**：缩放 ±15%、平移 ±10%、旋转 ±10°
