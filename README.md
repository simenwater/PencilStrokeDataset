# PencilStrokeDataset

Apple Pencil 手写音乐符号笔触数据集，用于训练 DaSegno 手写识别模型。

## 数据格式

```json
{
  "writer": "jiangbolong",
  "device": "iPad Pro",
  "session": "2026-03-27_0130",
  "point_format": ["x", "y", "force"],
  "samples": [
    {
      "label": "quarter-note-up",
      "class_id": 1,
      "strokes": [
        [[234.5, 180.3, 0.456], [235.1, 181.0, 0.512], ...],
        [[234.8, 120.0, 0.380], ...]
      ]
    }
  ]
}
```

每个点：`[x, y, force]`
- x, y: 像素坐标（原始值，归一化由训练端处理）
- force: Apple Pencil 压力 0~1（反映线条粗细）

## 类别

### 第一阶段（15 类）
| ID | Name | Description |
|----|------|-------------|
| 0 | quarter-note-up | 四分音符 ↑ |
| 1 | quarter-note-down | 四分音符 ↓ |
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

### 后续阶段
完整 69 类见 MusicSymbolDemo 项目。

## 采集工具

`MusicSymbolDemo` iPad app（`~/Desktop/Music/MusicSymbolDemo/`）

## 训练

Win 端 `MusicSymbolTrainer` 项目加载此数据训练。
