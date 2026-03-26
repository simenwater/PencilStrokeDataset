# 训练指令（给 Win Claude 看的）

## 你要做什么

用 `collected/iPad.jsonl` 里的 Apple Pencil 手写数据，训练一个 **15 类笔画序列分类器**。

## 数据格式

`collected/iPad.jsonl` 每行一个 JSON：
```json
{"label": "quarter-note-up", "class_id": 0, "strokes": [[[x,y,force], [x,y,force], ...], [[x,y,force], ...]], "device": "iPad", "time": "..."}
```

- `strokes`: 多笔画，每笔是 `[[x, y, force], ...]` 坐标序列
- `label`: 类别名
- `class_id`: 类别 ID（0-14）
- `force`: Apple Pencil 压力 0~1（可选特征，也可以只用 x,y）

## 15 个类别

| ID | Label | 说明 |
|----|-------|------|
| 0 | quarter-note-up | 四分音符 符干↑ |
| 1 | quarter-note-down | 四分音符 符干↓ |
| 2 | eighth-note-up | 八分音符 ↑ |
| 3 | eighth-note-down | 八分音符 ↓ |
| 4 | half-note-up | 二分音符 ↑ |
| 5 | half-note-down | 二分音符 ↓ |
| 6 | whole-note | 全音符 |
| 7 | rest-quarter | 四分休止符 |
| 8 | rest-eighth | 八分休止符 |
| 9 | treble-clef | 高音谱号 |
| 10 | sharp | 升号 |
| 11 | flat | 降号 |
| 12 | natural | 还原号 |
| 13 | barline-single | 小节线 |
| 14 | dot | 附点 |

## 训练步骤

### 1. 加载数据
```python
import json

samples = []
with open("collected/iPad.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")
# 预期: ~567 samples
```

### 2. 数据预处理
```python
# 每个 sample 的 strokes → 展平为点序列 [x, y, eos]
# 跟 MusicSymbolTrainer/train_rnn.py 的 StrokeDataset 一样的格式
# 但多了 force，可以用 [x, y, force, eos] 4维输入

def sample_to_sequence(sample):
    points = []
    for stroke in sample["strokes"]:
        for i, pt in enumerate(stroke):
            x, y = pt[0], pt[1]
            force = pt[2] if len(pt) > 2 else 0.5
            eos = 1.0 if i == len(stroke) - 1 else 0.0
            points.append([x, y, force, eos])
    return points
```

### 3. 归一化
坐标是 iPad 屏幕原始像素值，需要归一化：
```python
# 居中到 0.5（跟 HOMUS 一样）
def normalize(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    scale = max(max(xs)-min(xs), max(ys)-min(ys), 1)
    return [[(p[0]-cx)/scale + 0.5, (p[1]-cy)/scale + 0.5, p[2], p[3]] for p in points]
```

### 4. 模型
用 Bi-LSTM（跟 MusicSymbolTrainer/train_rnn.py 一样的架构），但：
- **input_size = 4**（x, y, force, eos）不是 3
- **num_classes = 15** 不是 32
- 其他超参保持不变（hidden_size=128, num_layers=2, dropout=0.3）

### 5. 数据划分
```python
# 按 80/10/10 划分，打乱顺序，固定种子
import numpy as np
np.random.seed(42)
indices = np.random.permutation(len(samples))
n = len(indices)
train = [samples[i] for i in indices[:int(n*0.8)]]   # ~454
val   = [samples[i] for i in indices[int(n*0.8):int(n*0.9)]]  # ~57
test  = [samples[i] for i in indices[int(n*0.9):]]   # ~56
```

### 6. 训练
- optimizer: AdamW, lr=1e-3
- epochs: 50
- batch_size: 32（数据少，batch 小一点）
- label_smoothing: 0.1
- 数据增强: 随机缩放 ±15%、随机平移 ±10%、随机旋转 ±10°
- 每 epoch 打印 train_acc 和 val_acc
- 保存 val_acc 最高的模型为 pencil_best.pt

### 7. 测试集评估
训练完后用 test set 评估：
- 总体 accuracy
- 每类 accuracy（confusion matrix）
- 打印最差的 5 个类和它们混淆为什么

### 8. 导出
训练完后保存:
- `pencil_best.pt` → 模型权重
- `pencil_results.json` → 准确率 + per_class accuracy + confusion matrix
- push 到 GitHub

## 目标
- val_accuracy > 95%（567 样本 + 数据增强应该能到）
- 重点看 confusion matrix：哪些类在互混

## 不要做的事
- 不要用 HOMUS 数据混合训练（这是纯 iPad 数据，风格不同）
- 不要改 collected/ 目录下的数据文件
- 不要用 CNN 图像模型（这是笔画序列数据，用 RNN）
