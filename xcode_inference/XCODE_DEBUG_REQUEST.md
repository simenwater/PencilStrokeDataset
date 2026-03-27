# Xcode 推理代码调试请求

## 问题描述
- Python 训练准确率: **94.44%**
- Xcode 实际识别率: **~10%**
- 结论: 预处理代码不一致

## 需要 Mac 上传的文件

请将以下 Swift 代码文件上传到 `xcode_inference/` 文件夹:

### 1. 笔画数据收集代码
- 获取 Apple Pencil 的 x, y, force 坐标的代码
- 文件可能叫: `CanvasView.swift`, `DrawingView.swift`, `StrokeCollector.swift` 等

### 2. 预处理/归一化代码
- 把原始坐标转换成模型输入的代码
- 关键看:
  - 坐标归一化方式 (中心化? 缩放到[0,1]?)
  - EOS (end-of-stroke) 标记怎么加的
  - 输入格式是 `[x, y, eos]` 还是 `[x, y, force, eos]`?

### 3. CoreML 模型调用代码
- 加载和调用 .mlmodel 的代码
- 输入 tensor 的 shape 是什么?

## Python 训练时的预处理 (参考对比)

```python
# train_pencil.py 中的归一化方式:

def normalize_points(points):
    # 1. 计算边界框中心
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    # 2. 计算缩放因子
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    scale = max(width, height, 1)

    # 3. 归一化到 [0, 1]
    normalized = []
    for p in points:
        x_norm = (p[0] - cx) / scale + 0.5
        y_norm = (p[1] - cy) / scale + 0.5
        force = p[2] if len(p) > 2 else 0.5
        eos = p[3] if len(p) > 3 else 0.0
        normalized.append([x_norm, y_norm, force, eos])

    return normalized

# EOS 标记: 每个笔画最后一个点 eos=1.0, 其他点 eos=0.0
# 输入格式: [x, y, force, eos] 共4维
```

## 上传命令

```bash
cd PencilStrokeDataset
mkdir -p xcode_inference
# 复制相关 Swift 文件
cp /path/to/YourApp/*.swift xcode_inference/
git add xcode_inference/
git commit -m "Add Xcode inference code for debugging preprocessing mismatch"
git push
```

## 上传后

Windows Claude 会:
1. 对比 Python 和 Swift 的预处理代码
2. 找出不一致的地方
3. 给出修复建议

---
**目标: 让 Xcode 里的识别率从 10% 提升到 90%+**
