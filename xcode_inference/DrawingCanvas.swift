import SwiftUI
import PencilKit

struct DrawingCanvas: UIViewRepresentable {
    @Binding var drawing: PKDrawing
    var onStrokeAdded: (PKStroke) -> Void

    func makeUIView(context: Context) -> PKCanvasView {
        let canvas = PKCanvasView()
        canvas.drawingPolicy = .anyInput  // 手指也能画（方便模拟器测试）
        canvas.tool = PKInkingTool(.pen, color: .black, width: 3)
        canvas.backgroundColor = .clear
        canvas.isOpaque = false
        canvas.delegate = context.coordinator

        // 显示工具选择器
        let toolPicker = PKToolPicker()
        toolPicker.setVisible(true, forFirstResponder: canvas)
        toolPicker.addObserver(canvas)
        canvas.becomeFirstResponder()
        context.coordinator.toolPicker = toolPicker

        return canvas
    }

    func updateUIView(_ canvas: PKCanvasView, context: Context) {
        if canvas.drawing.strokes.count != drawing.strokes.count {
            canvas.drawing = drawing
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PKCanvasViewDelegate {
        let parent: DrawingCanvas
        var toolPicker: PKToolPicker?
        private var lastStrokeCount = 0

        init(_ parent: DrawingCanvas) {
            self.parent = parent
        }

        func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            let currentCount = canvasView.drawing.strokes.count
            parent.drawing = canvasView.drawing

            // 新增了笔画
            if currentCount > lastStrokeCount {
                let newStroke = canvasView.drawing.strokes[currentCount - 1]
                parent.onStrokeAdded(newStroke)
            }
            lastStrokeCount = currentCount
        }
    }
}
