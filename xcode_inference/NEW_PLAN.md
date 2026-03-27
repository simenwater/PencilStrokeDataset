# 新方案：组件识别 + 乐理规则组合

## 之前的问题

训练用 `pack_padded_sequence` 跳过 padding，CoreML 导出后没有 packing，推理时模型吃到 462 个填充数据，识别率从 94% 掉到 10%。

**解决办法**：训练时不用 `pack_padded_sequence`，用固定 512 长度 + 最后一个点重复填充，跟 Swift 端一致。

## 新方向：不再识别完整音符，改为识别组件

### 为什么

整体识别"四分音符 vs 二分音符"很难——模型要同时判断符头空心/实心、符干方向、有没有旗帜。
拆成组件后每个判断都很简单。

### 组件分类器（模型负责）

只需要识别 **6-8 类基础形状**：

| ID | 类别 | 形状 |
|----|------|------|
| 0 | notehead_filled | 实心圆 ● |
| 1 | notehead_hollow | 空心圆 ○ |
| 2 | stem | 竖线 \| |
| 3 | flag | 旗帜弯钩 |
| 4 | dot | 小点 • |
| 5 | rest | 休止符（各种） |
| 6 | clef | 谱号 |
| 7 | accidental | 升降号 |

### 乐理组合规则（代码写死，不用模型）

```python
def determine_note(notehead, has_stem, flag_count, has_dot):
    if notehead == "hollow" and not has_stem:
        return "whole"
    elif notehead == "hollow" and has_stem:
        return "half"
    elif notehead == "filled" and has_stem and flag_count == 0:
        return "quarter"
    elif notehead == "filled" and has_stem and flag_count == 1:
        return "eighth"
    elif notehead == "filled" and has_stem and flag_count == 2:
        return "sixteenth"
    elif notehead == "filled" and has_stem and flag_count == 3:
        return "thirty_second"
    # has_dot → duration * 1.5
```

## 执行步骤

### 步骤 1：准备组件数据集

从 HOMUS 笔画数据拆分组件：
```python
# HOMUS 每个样本有多个 stroke
# Quarter-Note 通常是 2 笔: stroke[0]=符头, stroke[1]=符干
# Eighth-Note 通常是 2-3 笔: stroke[0]=符头, stroke[1]=符干, stroke[2]=旗帜

# 拆分逻辑：
# 1. 对每个 HOMUS 样本，分析每个 stroke 的几何特征
# 2. 宽高比 < 0.5 → stem（竖线）
# 3. 宽高比 ~ 1.0 且封闭 → notehead
# 4. 短弧线 → flag
# 5. 用这些自动标注训练组件分类器
```

从 MUSCIMA++ 直接用（已经是组件级标注）。
从 iPad 数据也可以拆（每个样本的每笔对应一个组件）。

### 步骤 2：训练组件分类器

```python
# RNN 或小 CNN，input_size=3 (x, y, eos)
# 只有 8 类，比 15 类或 32 类简单得多
# 每类需要的样本也更少

model = StrokeLSTM(input_size=3, num_classes=8)
# 不用 pack_padded_sequence！
# 训练时就用固定 512 长度 + 填充（跟推理一致）
```

### 步骤 3：修复 padding 问题

训练时的 forward 改为：
```python
def forward(self, x, lengths=None):
    # 不用 pack_padded_sequence，直接跑
    output, (hidden, cell) = self.lstm(x)
    hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
    return self.classifier(hidden)
```

训练数据预处理：
```python
def pad_sequence_fixed(points, max_len=512):
    """跟 Swift 端完全一致的填充方式"""
    if len(points) >= max_len:
        return points[:max_len]
    # 用最后一个点重复填充
    last = points[-1]
    padding = [last] * (max_len - len(points))
    return points + padding
```

### 步骤 4：导出 + iPad 验证

训练完后：
1. 直接 `torch.jit.trace` 导出（不需要去掉 pack，因为本来就没用）
2. push `component_best.pt` 到 GitHub
3. Mac 端导出 CoreML
4. iPad 上验证

### 步骤 5：iPad 端组合

```
用户画一小节（多笔）
  ↓ 按空间分组
  ↓ 每笔送组件分类器
  ↓ "filled_notehead" + "stem" + "flag"
  ↓ 乐理规则 → 八分音符
  ↓ 五线谱 Y 坐标 → 音高
  ↓ 渲染
```

## 总结

| | 旧方案 | 新方案 |
|---|---|---|
| 模型任务 | 识别完整音符（15类） | 识别基础形状（8类） |
| 组合逻辑 | 模型猜 | 代码规则 |
| padding | 训练/推理不一致 | 一致（都用固定512+重复填充） |
| 难度 | 高（形状相似的音符互混） | 低（圆vs线vs弯钩很好分） |
| 准确率预期 | ~90% | **97%+** |

## 数据来源优先级

1. **MUSCIMA++** — 已有组件标注，直接用
2. **HOMUS** — 按笔画拆分组件，自动标注
3. **iPad 数据** — 每笔对应一个组件
