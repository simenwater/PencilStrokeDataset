# Swift 端预处理分析

## 关键文件

| 文件 | 作用 |
|------|------|
| StrokeClassifier.swift | 模型加载 + 推理 + 归一化 |
| DrawingCanvas.swift | PencilKit 画布封装 |
| RecordingView.swift | 数据采集（保存到服务器的代码） |

## 推理路径 (StrokeClassifier.swift)

### 模型加载 (第 34-52 行)
```swift
pencilRNN = try? PencilStrokeRNN(configuration: .init()).model  // 15 类
homusRNN = try? MusicSymbolRNN(configuration: .init()).model    // 32 类 fallback
```

### 推理入口 (第 96-114 行)
`classifyMultiStroke()` → 调用 `classifyWithPencilRNN()`

### 归一化 (第 260-290 行) ← 最可能出问题的地方
```swift
// 居中到 0.5
let w = maxX - minX, h = maxY - minY
let scale = max(w, h, 1)
let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2

// 归一化
input[i][0] = Float((x - cx) / scale + 0.5)  // x
input[i][1] = Float((y - cy) / scale + 0.5)  // y
input[i][2] = Float(force)                    // force (原始值 0~1)
input[i][3] = eos ? 1.0 : 0.0                // end of stroke
```

### 填充策略 (第 292-300 行)
```swift
// 用最后一个点重复填充到 512 长度
for i in count..<maxSeqLen {
    input[i] = lastPoint  // 不是零填充
}
```

### MLMultiArray shape
```swift
shape: [1, 512, 4]  // batch=1, seq_len=512, features=4(x,y,force,eos)
```

## 数据采集路径 (RecordingView.swift)

### 保存到服务器的数据 (第 276-290 行)
```swift
// 保存原始像素坐标，不归一化
points.append([
    round(Double(p.location.x) * 100) / 100,  // 原始 x 像素
    round(Double(p.location.y) * 100) / 100,  // 原始 y 像素
    round(Double(p.force) * 1000) / 1000,      // force 0~1
])
```

## 可能的不一致

1. **采集时存的是原始像素坐标**（几百的数值），推理时归一化到 0~1
2. **训练时 Python 归一化**是从 JSON 读原始坐标 → normalize → 训练
3. **推理时 Swift 归一化**是从 PKStroke 读原始坐标 → normalize → 推理
4. 如果归一化逻辑一致，应该没问题。但 **scale 计算** 和 **居中方式** 可能有细微差异

### 特别注意
- Python: `scale = max(width, height, 1)` — 用 1 作为最小值
- Swift: `let scale = max(w, h, 1)` — CGFloat 的 1 vs Python 的 int 1
- 这两个应该是一致的

### 可能的 bug
- **填充方式不同**：Python 训练用 `pack_padded_sequence` 完全不看填充，Swift 用最后一个点填充 512 长度。Bi-LSTM 反向传播会处理这些填充数据，导致结果不同。
- **这是最大的嫌疑**：训练时 padding 被忽略，推理时 padding 被处理。
