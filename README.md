# PencilStrokeDataset

Apple Pencil 手写音乐符号笔触数据集，用于训练 DaSegno 手写识别模型。

---

## ⚠️ 重要：使用新方案！

### 新方案 (2026-03-27) ✅ 推荐

**组件分类器** - 识别 8 类基础形状，然后用乐理规则组合

| 文件 | 说明 |
|------|------|
| `component_best.pt` | **最新模型** (100%, 432K params) |
| `train_component_rnn.py` | 新训练脚本 |
| `auto_iterate.py` | 自动迭代优化脚本 |
| `component_results.json` | 训练结果 |

**关键修复:**
- ✅ 不用 `pack_padded_sequence`
- ✅ 固定 512 长度 + 最后点重复填充
- ✅ 输入 3 维 `[x, y, eos]` (去掉 force 噪声)
- ✅ 训练/推理完全一致

**8 类组件:**
| ID | 类别 | 说明 |
|----|------|------|
| 0 | notehead_filled | 实心符头 ● |
| 1 | notehead_hollow | 空心符头 ○ |
| 2 | stem | 符干 \| |
| 3 | flag | 旗帜 |
| 4 | dot | 附点 • |
| 5 | rest | 休止符 |
| 6 | clef | 谱号 |
| 7 | accidental | 升降号 |

### 旧方案 ❌ 已废弃

| 文件 | 问题 |
|------|------|
| `pencil_best.pt` | 用 pack_padded_sequence，推理时 padding 不一致 → **10% 识别率** |
| `pencil_finetuned.pt` | 同上问题 |
| `train_pencil.py` | 旧训练脚本，有 padding bug |
| `finetune_from_homus.py` | 旧 finetune 脚本 |
| `FINETUNE.md` | 旧 finetune 说明 |

---

## Swift 端适配

### 归一化 (必须与训练一致)
```swift
let w = maxX - minX, h = maxY - minY
let scale = max(w, h, 1)
let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2

// 归一化到 [0, 1]
let x_norm = (x - cx) / scale + 0.5
let y_norm = (y - cy) / scale + 0.5
let eos = isLastPoint ? 1.0 : 0.0  // 每笔最后一点
```

### 填充 (必须与训练一致)
```swift
// 用最后一个点重复填充到 512 长度
for i in count..<512 {
    input[i] = lastPoint  // 不是零填充！
}
```

### 输入格式
```swift
shape: [1, 512, 3]  // batch=1, seq_len=512, features=3(x,y,eos)
```

### 乐理组合规则 (代码写死)
```swift
func determineNote(notehead: String, hasStem: Bool, flagCount: Int) -> String {
    if notehead == "hollow" && !hasStem { return "whole" }
    if notehead == "hollow" && hasStem { return "half" }
    if notehead == "filled" && hasStem && flagCount == 0 { return "quarter" }
    if notehead == "filled" && hasStem && flagCount == 1 { return "eighth" }
    if notehead == "filled" && hasStem && flagCount == 2 { return "sixteenth" }
    return "unknown"
}
```

---

## 数据格式

```json
{
  "label": "quarter-note-up",
  "class_id": 0,
  "strokes": [
    [[234.5, 180.3, 0.456], [235.1, 181.0, 0.512], ...],
    [[234.8, 120.0, 0.380], ...]
  ],
  "device": "iPad",
  "time": "2026-03-26T19:46:31"
}
```

每个点：`[x, y, force]`
- x, y: 像素坐标
- force: Apple Pencil 压力 0~1

---

## 目录结构

```
PencilStrokeDataset/
├── collected/
│   └── iPad.jsonl          # 采集的数据 (567+ 样本)
├── xcode_inference/        # Swift 代码分析
│   ├── PREPROCESSING_ANALYSIS.md
│   └── NEW_PLAN.md
│
├── component_best.pt       # ✅ 新模型 (使用这个!)
├── train_component_rnn.py  # ✅ 新训练脚本
├── auto_iterate.py         # ✅ 迭代优化
│
├── pencil_best.pt          # ❌ 旧模型 (有bug)
├── train_pencil.py         # ❌ 旧脚本
└── ...
```

---

## 采集工具

`MusicSymbolDemo` iPad app
