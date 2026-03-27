import PencilKit
import CoreGraphics
import UIKit
import CoreML
import Vision

// MARK: - 分类结果
struct StrokeClassification {
    let label: String           // 分类名称
    let confidence: Double      // 置信度 0-1
    let method: String          // "geometric" 或 "coreml"
    let features: StrokeFeatures // 用于调试显示
}

// MARK: - 笔画几何特征
struct StrokeFeatures {
    let pointCount: Int
    let boundingBox: CGRect
    let aspectRatio: Double     // 宽高比 (>1 = 横向, <1 = 纵向)
    let linearity: Double       // 0=弯曲 1=直线
    let curvature: Double       // 平均曲率
    let enclosedArea: Double    // 封闭面积比 (1=完全封闭的圆)
    let totalLength: Double     // 笔画总长度
    let avgSpeed: Double        // 平均画速 (points/sec)
    let dominantDirection: String // "horizontal", "vertical", "diagonal", "circular"
}

// MARK: - 几何规则分类器
class StrokeClassifier {

    private var pencilRNN: MLModel?     // iPad 手写数据训练的 15 类模型
    private var homusRNN: MLModel?      // HOMUS 训练的 32 类模型 (fallback)
    private var cnnModel: VNCoreMLModel?
    private let maxSeqLen = 512

    init() {
        // 优先加载 iPad 手写模型（你自己的笔迹训练的）
        if let model = try? PencilStrokeRNN(configuration: .init()).model {
            self.pencilRNN = model
            print("[StrokeClassifier] PencilStroke RNN loaded (15 classes, your handwriting)")
        }

        // HOMUS RNN fallback
        if let model = try? MusicSymbolRNN(configuration: .init()).model {
            self.homusRNN = model
            print("[StrokeClassifier] HOMUS RNN loaded (32 classes, fallback)")
        }

        // CNN fallback
        if let model = try? MusicSymbolClassifier(configuration: .init()).model,
           let vnModel = try? VNCoreMLModel(for: model) {
            self.cnnModel = vnModel
            print("[StrokeClassifier] CNN loaded (image, fallback)")
        }
    }

    /// 对单个 PKStroke 进行分类
    /// 优先 PencilRNN（你的手写模型）→ HOMUS RNN → CNN → 几何规则
    func classify(_ stroke: PKStroke) -> StrokeClassification {
        let features = extractFeatures(stroke)

        // 1. iPad 手写 RNN（15 类，你自己的笔迹）
        if let result = classifyWithPencilRNN([stroke]) {
            return StrokeClassification(
                label: result.label, confidence: result.confidence,
                method: "pencil", features: features)
        }

        // 2. HOMUS RNN fallback
        if let result = classifyWithRNN(stroke) {
            return StrokeClassification(
                label: result.label, confidence: result.confidence,
                method: "homus", features: features)
        }

        // 3. CNN fallback
        if let result = classifyWithCNN(stroke) {
            return StrokeClassification(
                label: result.label, confidence: result.confidence,
                method: "cnn", features: features)
        }

        // 4. 几何规则
        let (label, confidence) = classifyByGeometry(features)
        return StrokeClassification(
            label: label, confidence: confidence,
            method: "geo", features: features)
    }

    /// 对一组 strokes 进行空间分组 → 每组整体识别
    /// 返回的 classifications 数量 = 分组数（不是笔画数）
    func classifyGroup(_ strokes: [PKStroke]) -> [StrokeClassification] {
        let groups = groupNearbyStrokes(strokes, threshold: 50)
        return groups.map { classifyMultiStroke($0) }
    }

    /// 多笔画合并识别
    func classifyMultiStroke(_ strokes: [PKStroke]) -> StrokeClassification {
        let features = extractFeatures(strokes[0])

        // 优先 PencilRNN（4 维，带 force）
        if let result = classifyWithPencilRNN(strokes) {
            return StrokeClassification(
                label: result.label, confidence: result.confidence,
                method: "pencil", features: features)
        }

        // HOMUS RNN fallback（3 维）
        if let result = classifyMultiStrokeWithRNN(strokes) {
            return StrokeClassification(
                label: result.label, confidence: result.confidence,
                method: "homus", features: features)
        }

        return classify(strokes[0])
    }

    // MARK: - 笔画空间分组

    /// 把空间上接近的笔画归为一组（一个音符通常是 2-3 笔紧挨着的）
    func groupNearbyStrokes(_ strokes: [PKStroke], threshold: CGFloat) -> [[PKStroke]] {
        guard !strokes.isEmpty else { return [] }

        var groups: [[Int]] = []  // 每组存笔画索引
        var assigned = Set<Int>()

        for i in 0..<strokes.count {
            guard !assigned.contains(i) else { continue }

            var group = [i]
            assigned.insert(i)

            // 找所有跟这组距离近的笔画
            var changed = true
            while changed {
                changed = false
                for j in 0..<strokes.count {
                    guard !assigned.contains(j) else { continue }
                    // 检查 j 是否跟 group 里任意笔画接近
                    for k in group {
                        if strokeDistance(strokes[k], strokes[j]) < threshold {
                            group.append(j)
                            assigned.insert(j)
                            changed = true
                            break
                        }
                    }
                }
            }

            groups.append(group)
        }

        return groups.map { indices in indices.map { strokes[$0] } }
    }

    /// 两个笔画的最近距离（用边界框中心）
    private func strokeDistance(_ a: PKStroke, _ b: PKStroke) -> CGFloat {
        let ac = CGPoint(x: a.renderBounds.midX, y: a.renderBounds.midY)
        let bc = CGPoint(x: b.renderBounds.midX, y: b.renderBounds.midY)
        return hypot(ac.x - bc.x, ac.y - bc.y)
    }

    // MARK: - 多笔画 RNN 推理

    private func classifyMultiStrokeWithRNN(_ strokes: [PKStroke]) -> (label: String, confidence: Double)? {
        guard let model = homusRNN else { return nil }

        // 收集所有点，计算全局边界
        var allPoints: [(CGFloat, CGFloat, Bool)] = []
        var minX = CGFloat.greatestFiniteMagnitude, minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude, maxY = -CGFloat.greatestFiniteMagnitude

        for stroke in strokes {
            let path = stroke.path
            for i in 0..<path.count {
                let p = path[i].location
                minX = min(minX, p.x); minY = min(minY, p.y)
                maxX = max(maxX, p.x); maxY = max(maxY, p.y)
                allPoints.append((p.x, p.y, i == path.count - 1))
            }
        }

        guard allPoints.count >= 2 else { return nil }

        // ── 归一化：跟 HOMUS prepare.py normalize_strokes() 完全一致 ──
        // cx = (min_x + max_x) / 2;  nx = (x - cx) / scale + 0.5
        let w = maxX - minX, h = maxY - minY
        let scale = max(w, h, 1)
        let cx = (minX + maxX) / 2
        let cy = (minY + maxY) / 2

        guard let inputArray = try? MLMultiArray(shape: [1, NSNumber(value: maxSeqLen), 3], dataType: .float32) else {
            return nil
        }

        let pointCount = min(allPoints.count, maxSeqLen)

        for i in 0..<pointCount {
            let (x, y, eos) = allPoints[i]
            let nx = Float((x - cx) / scale + 0.5)  // 居中到 0.5
            let ny = Float((y - cy) / scale + 0.5)
            inputArray[[0, NSNumber(value: i), 0] as [NSNumber]] = NSNumber(value: nx)
            inputArray[[0, NSNumber(value: i), 1] as [NSNumber]] = NSNumber(value: ny)
            inputArray[[0, NSNumber(value: i), 2] as [NSNumber]] = NSNumber(value: eos ? Float(1.0) : Float(0.0))
        }

        // ── 填充：用最后一个点重复填充（不是零，防止双向 LSTM 被零干扰）──
        if pointCount < maxSeqLen {
            let (lastX, lastY, _) = allPoints[pointCount - 1]
            let lastNx = Float((lastX - cx) / scale + 0.5)
            let lastNy = Float((lastY - cy) / scale + 0.5)
            for i in pointCount..<maxSeqLen {
                inputArray[[0, NSNumber(value: i), 0] as [NSNumber]] = NSNumber(value: lastNx)
                inputArray[[0, NSNumber(value: i), 1] as [NSNumber]] = NSNumber(value: lastNy)
                inputArray[[0, NSNumber(value: i), 2] as [NSNumber]] = NSNumber(value: Float(1.0)) // eos
            }
        }

        let inputFeatures = try? MLDictionaryFeatureProvider(dictionary: ["stroke_sequence": inputArray])
        guard let inputFeatures, let output = try? model.prediction(from: inputFeatures) else {
            return nil
        }

        if let classLabel = output.featureValue(for: "classLabel")?.stringValue {
            var confidence = 0.5
            for name in output.featureNames {
                if let dict = output.featureValue(for: name)?.dictionaryValue as? [String: Double],
                   let prob = dict[classLabel] {
                    confidence = prob
                    break
                }
            }
            return (classLabel, confidence)
        }

        return nil
    }

    // MARK: - PencilStroke RNN（iPad 手写模型，4 维输入）

    private func classifyWithPencilRNN(_ strokes: [PKStroke]) -> (label: String, confidence: Double)? {
        guard let model = pencilRNN else { return nil }

        var allPoints: [(CGFloat, CGFloat, CGFloat, Bool)] = []  // x, y, force, eos
        var minX = CGFloat.greatestFiniteMagnitude, minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude, maxY = -CGFloat.greatestFiniteMagnitude

        for stroke in strokes {
            let path = stroke.path
            for i in 0..<path.count {
                let p = path[i]
                minX = min(minX, p.location.x); minY = min(minY, p.location.y)
                maxX = max(maxX, p.location.x); maxY = max(maxY, p.location.y)
                allPoints.append((p.location.x, p.location.y, CGFloat(p.force), i == path.count - 1))
            }
        }

        guard allPoints.count >= 2 else { return nil }

        // 归一化（居中到 0.5，跟 train_pencil.py 一致）
        let w = maxX - minX, h = maxY - minY
        let scale = max(w, h, 1)
        let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2

        guard let input = try? MLMultiArray(shape: [1, NSNumber(value: maxSeqLen), 4], dataType: .float32) else { return nil }

        let count = min(allPoints.count, maxSeqLen)
        for i in 0..<count {
            let (x, y, force, eos) = allPoints[i]
            input[[0, NSNumber(value: i), 0] as [NSNumber]] = NSNumber(value: Float((x - cx) / scale + 0.5))
            input[[0, NSNumber(value: i), 1] as [NSNumber]] = NSNumber(value: Float((y - cy) / scale + 0.5))
            input[[0, NSNumber(value: i), 2] as [NSNumber]] = NSNumber(value: Float(force))
            input[[0, NSNumber(value: i), 3] as [NSNumber]] = NSNumber(value: eos ? Float(1.0) : Float(0.0))
        }

        // 填充（用最后一个点重复）
        if count < maxSeqLen {
            let (lx, ly, lf, _) = allPoints[count - 1]
            let nlx = Float((lx - cx) / scale + 0.5), nly = Float((ly - cy) / scale + 0.5)
            for i in count..<maxSeqLen {
                input[[0, NSNumber(value: i), 0] as [NSNumber]] = NSNumber(value: nlx)
                input[[0, NSNumber(value: i), 1] as [NSNumber]] = NSNumber(value: nly)
                input[[0, NSNumber(value: i), 2] as [NSNumber]] = NSNumber(value: Float(lf))
                input[[0, NSNumber(value: i), 3] as [NSNumber]] = NSNumber(value: Float(1.0))
            }
        }

        guard let features = try? MLDictionaryFeatureProvider(dictionary: ["stroke_sequence": input]),
              let output = try? model.prediction(from: features),
              let label = output.featureValue(for: "classLabel")?.stringValue else { return nil }

        // 安全读取置信度（输出 key 可能是 var_60 或其他名字）
        var confidence = 0.5
        for name in output.featureNames {
            if let dict = output.featureValue(for: name)?.dictionaryValue as? [String: Double],
               let prob = dict[label] {
                confidence = prob
                break
            }
        }

        return (label, confidence)
    }

    // MARK: - HOMUS RNN fallback（3 维输入）

    private func classifyWithRNN(_ stroke: PKStroke) -> (label: String, confidence: Double)? {
        return classifyMultiStrokeWithRNN([stroke])
    }

    // MARK: - CNN 分类（渲染图片 → Vision 推理）

    private func classifyWithCNN(_ stroke: PKStroke) -> (label: String, confidence: Double)? {
        guard let vnModel = cnnModel else { return nil }

        let image = renderStrokeToImage(stroke, size: 64)
        guard let cgImage = image.cgImage else { return nil }

        var result: (String, Double)?
        let request = VNCoreMLRequest(model: vnModel) { req, _ in
            guard let observations = req.results as? [VNClassificationObservation],
                  let top = observations.first else { return }
            result = (top.identifier, Double(top.confidence))
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        return result
    }

    // MARK: - 特征提取

    func extractFeatures(_ stroke: PKStroke) -> StrokeFeatures {
        let points = stroke.path
        let count = points.count
        guard count >= 2 else {
            return StrokeFeatures(
                pointCount: count, boundingBox: .zero, aspectRatio: 1,
                linearity: 0, curvature: 0, enclosedArea: 0,
                totalLength: 0, avgSpeed: 0, dominantDirection: "unknown"
            )
        }

        // 边界框
        var minX = CGFloat.greatestFiniteMagnitude
        var minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude
        var maxY = -CGFloat.greatestFiniteMagnitude

        var totalLength: Double = 0
        var totalCurvature: Double = 0
        var curvatureCount = 0

        for i in 0..<count {
            let p = points[i].location
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)

            if i > 0 {
                let prev = points[i - 1].location
                let dx = p.x - prev.x
                let dy = p.y - prev.y
                totalLength += sqrt(dx * dx + dy * dy)
            }

            // 三点曲率
            if i > 0 && i < count - 1 {
                let p0 = points[i - 1].location
                let p1 = points[i].location
                let p2 = points[i + 1].location
                let curv = pointCurvature(p0, p1, p2)
                totalCurvature += curv
                curvatureCount += 1
            }
        }

        let bbox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        let w = max(bbox.width, 1)
        let h = max(bbox.height, 1)
        let aspectRatio = Double(w / h)

        // 线性度：笔画总长度 vs 起点到终点直线距离
        let start = points[0].location
        let end = points[count - 1].location
        let directDist = hypot(end.x - start.x, end.y - start.y)
        let linearity = totalLength > 0 ? min(Double(directDist) / totalLength, 1.0) : 0

        // 平均曲率
        let avgCurvature = curvatureCount > 0 ? totalCurvature / Double(curvatureCount) : 0

        // 封闭度：起点终点距离 / 笔画总长度 (越小越封闭)
        let closedness = totalLength > 0 ? Double(directDist) / totalLength : 1
        let enclosedArea = 1.0 - closedness  // 0=开放直线, 1=完全封闭

        // 画速
        let duration = points[count - 1].timeOffset - points[0].timeOffset
        let avgSpeed = duration > 0 ? Double(count) / duration : 0

        // 主方向
        let dominantDirection: String
        if enclosedArea > 0.6 {
            dominantDirection = "circular"
        } else if aspectRatio > 2.5 {
            dominantDirection = "horizontal"
        } else if aspectRatio < 0.4 {
            dominantDirection = "vertical"
        } else {
            dominantDirection = "diagonal"
        }

        return StrokeFeatures(
            pointCount: count,
            boundingBox: bbox,
            aspectRatio: aspectRatio,
            linearity: linearity,
            curvature: avgCurvature,
            enclosedArea: enclosedArea,
            totalLength: totalLength,
            avgSpeed: avgSpeed,
            dominantDirection: dominantDirection
        )
    }

    // MARK: - 几何规则分类

    private func classifyByGeometry(_ f: StrokeFeatures) -> (String, Double) {
        let ar = f.aspectRatio
        let lin = f.linearity

        // ── 附点 (dot): 极小的点 ──
        if f.totalLength < 15 && f.boundingBox.width < 20 && f.boundingBox.height < 20 {
            return ("dot", 0.90)
        }

        // ── 符头 (notehead): 近圆形/椭圆，封闭度高 ──
        if f.enclosedArea > 0.5 && ar > 0.4 && ar < 2.5 && f.totalLength > 20 {
            // 区分空心 vs 实心需要看笔画面积覆盖率，这里先统一
            if f.enclosedArea > 0.75 {
                return ("notehead (closed)", 0.85)
            }
            return ("notehead", 0.75)
        }

        // ── 符干 (stem): 近垂直直线 ──
        if ar < 0.35 && lin > 0.85 && f.totalLength > 30 {
            return ("stem", 0.90)
        }

        // ── 小节线 (barline): 长垂直直线 ──
        if ar < 0.15 && lin > 0.90 && f.totalLength > 80 {
            return ("barline", 0.85)
        }

        // ── 符梁 (beam): 近水平直线，有一定粗度 ──
        if ar > 3.0 && lin > 0.80 && f.totalLength > 25 {
            return ("beam", 0.85)
        }

        // ── 五线谱线 (staff line): 很长的水平线 ──
        if ar > 8.0 && lin > 0.90 {
            return ("staff_line", 0.90)
        }

        // ── 符尾 (flag): 短弧线，在一端 ──
        if f.curvature > 0.02 && f.totalLength > 15 && f.totalLength < 80
           && ar > 0.3 && ar < 3.0 && lin < 0.7 {
            return ("flag", 0.70)
        }

        // ── 连线 (slur/tie): 长弧线 ──
        if f.curvature > 0.005 && f.totalLength > 60 && lin < 0.8 && lin > 0.3
           && ar > 1.5 {
            return ("slur/tie", 0.70)
        }

        // ── 渐强/渐弱 (hairpin): 长三角形状 ──
        if ar > 3.0 && lin > 0.5 && lin < 0.85 {
            return ("hairpin", 0.60)
        }

        // ── 升号 (sharp): 多笔画交叉，但单笔可能是其中一部分 ──
        if f.dominantDirection == "diagonal" && f.totalLength > 20 && f.totalLength < 60 {
            return ("accidental_stroke", 0.50)
        }

        // ── 无法确定 ──
        return ("unknown", 0.30)
    }

    // MARK: - 辅助函数

    /// 三点曲率 (Menger curvature)
    private func pointCurvature(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint) -> Double {
        let ax = p1.x - p0.x, ay = p1.y - p0.y
        let bx = p2.x - p1.x, by = p2.y - p1.y
        let cross = abs(ax * by - ay * bx)
        let a = hypot(ax, ay)
        let b = hypot(bx, by)
        let c = hypot(p2.x - p0.x, p2.y - p0.y)
        let denom = a * b * c
        return denom > 0 ? Double(2 * cross / denom) : 0
    }
}

// MARK: - CoreML 集成占位（Win 导出模型后启用）

extension StrokeClassifier {
    /// 将 PKStroke 渲染为 64×64 灰度图（黑底白字，与 HOMUS 训练数据一致）
    func renderStrokeToImage(_ stroke: PKStroke, size: Int = 64) -> UIImage {
        let path = stroke.path
        guard path.count >= 2 else { return UIImage() }

        // 计算边界
        var minX = CGFloat.greatestFiniteMagnitude, minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude, maxY = -CGFloat.greatestFiniteMagnitude
        for i in 0..<path.count {
            let p = path[i].location
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }
        let bboxW = max(maxX - minX, 1)
        let bboxH = max(maxY - minY, 1)

        let targetSize = CGSize(width: size, height: size)
        let scale = min(CGFloat(size) / bboxW, CGFloat(size) / bboxH) * 0.80
        let offsetX = (CGFloat(size) - bboxW * scale) / 2 - minX * scale
        let offsetY = (CGFloat(size) - bboxH * scale) / 2 - minY * scale

        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { ctx in
            let gc = ctx.cgContext
            // 黑色背景
            gc.setFillColor(UIColor.black.cgColor)
            gc.fill(CGRect(origin: .zero, size: targetSize))

            // 白色笔画
            gc.setStrokeColor(UIColor.white.cgColor)
            gc.setLineWidth(max(2, 3 * scale))
            gc.setLineCap(.round)
            gc.setLineJoin(.round)

            gc.translateBy(x: offsetX, y: offsetY)
            gc.scaleBy(x: scale, y: scale)

            gc.beginPath()
            gc.move(to: path[0].location)
            for i in 1..<path.count {
                gc.addLine(to: path[i].location)
            }
            gc.strokePath()
        }
    }

    /// 提取 PKStroke 的点序列（供 RNN 模型使用）
    func extractPointSequence(_ stroke: PKStroke, maxPoints: Int = 128) -> [[Double]] {
        let path = stroke.path
        let count = path.count
        guard count >= 2 else { return [] }

        // 归一化到 [0, 1]
        var minX = CGFloat.greatestFiniteMagnitude
        var minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude
        var maxY = -CGFloat.greatestFiniteMagnitude

        for i in 0..<count {
            let p = path[i].location
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }

        let rangeX = max(maxX - minX, 1)
        let rangeY = max(maxY - minY, 1)
        let scale = max(rangeX, rangeY)

        // 均匀采样到 maxPoints
        var sequence: [[Double]] = []
        let step = max(1, count / maxPoints)
        for i in stride(from: 0, to: count, by: step) {
            let p = path[i]
            let nx = Double((p.location.x - minX) / scale)
            let ny = Double((p.location.y - minY) / scale)
            let t = p.timeOffset
            let force = Double(p.force)
            sequence.append([nx, ny, t, force])
        }

        return sequence
    }
}
