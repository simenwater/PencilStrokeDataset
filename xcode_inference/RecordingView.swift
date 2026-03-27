import SwiftUI
import PencilKit

// MARK: - 数据持久化

class StrokeDataStore: ObservableObject {
    static let shared = StrokeDataStore()

    @Published var samples: [String: [SavedSample]] = [:]  // label → samples
    @Published var serverCounts: [String: Int] = [:]        // 服务器端计数

    private let fileURL: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        fileURL = docs.appendingPathComponent("stroke_dataset.json")
        load()
        syncFromServer()
    }

    var totalCount: Int {
        // 优先用服务器数据
        if !serverCounts.isEmpty {
            return serverCounts.values.reduce(0, +)
        }
        return samples.values.reduce(0) { $0 + $1.count }
    }

    func count(for label: String) -> Int {
        // 优先用服务器数据
        if let serverCount = serverCounts[label] {
            return serverCount
        }
        return samples[label]?.count ?? 0
    }

    func syncFromServer() {
        CloudSync.shared.fetchClassCounts { [weak self] counts in
            guard !counts.isEmpty else { return }
            self?.serverCounts = counts
        }
    }

    func add(_ sample: SavedSample) {
        samples[sample.label, default: []].append(sample)
        save()
    }

    func deleteLast(for label: String) {
        guard samples[label]?.isEmpty == false else { return }
        samples[label]?.removeLast()
        save()
    }

    // MARK: - 持久化

    private func save() {
        let flat = samples.values.flatMap { $0 }
        let export = flat.map { s -> [String: Any] in
            ["label": s.label, "class_id": s.classId, "strokes": s.strokes]
        }
        let wrapper: [String: Any] = [
            "source": "MusicSymbolDemo",
            "device": UIDevice.current.model,
            "date": ISO8601DateFormatter().string(from: Date()),
            "sample_count": flat.count,
            "point_format": ["x", "y", "force"],
            "samples": export
        ]
        if let data = try? JSONSerialization.data(withJSONObject: wrapper, options: .prettyPrinted) {
            try? data.write(to: fileURL)
        }
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawSamples = dict["samples"] as? [[String: Any]] else { return }

        for raw in rawSamples {
            guard let label = raw["label"] as? String,
                  let classId = raw["class_id"] as? Int,
                  let strokes = raw["strokes"] as? [[[Double]]] else { continue }
            let sample = SavedSample(label: label, classId: classId, strokes: strokes)
            samples[label, default: []].append(sample)
        }
    }

    /// 导出用的 URL（直接就是持久化的文件）
    var exportURL: URL { fileURL }
}

struct SavedSample {
    let label: String
    let classId: Int
    let strokes: [[[Double]]]
}

// MARK: - 类别定义

struct SymbolClass: Identifiable {
    let id: Int
    let name: String
    let hint: String
    let glyph: String

    static let phase1: [SymbolClass] = [
        .init(id: 0,  name: "quarter-note-up",   hint: "四分音符 ↑",  glyph: "\u{E1D5}"),
        .init(id: 1,  name: "quarter-note-down", hint: "四分音符 ↓",  glyph: "\u{E1D6}"),
        .init(id: 2,  name: "eighth-note-up",    hint: "八分音符 ↑",  glyph: "\u{E1D7}"),
        .init(id: 3,  name: "eighth-note-down",  hint: "八分音符 ↓",  glyph: "\u{E1D8}"),
        .init(id: 4,  name: "half-note-up",      hint: "二分音符 ↑",  glyph: "\u{E1D3}"),
        .init(id: 5,  name: "half-note-down",    hint: "二分音符 ↓",  glyph: "\u{E1D4}"),
        .init(id: 6,  name: "whole-note",        hint: "全音符 ○",    glyph: "\u{E1D2}"),
        .init(id: 7,  name: "rest-quarter",      hint: "四分休止符",   glyph: "\u{E4E5}"),
        .init(id: 8,  name: "rest-eighth",       hint: "八分休止符",   glyph: "\u{E4E6}"),
        .init(id: 9,  name: "treble-clef",       hint: "高音谱号",    glyph: "\u{E050}"),
        .init(id: 10, name: "sharp",             hint: "升号 ♯",     glyph: "\u{E262}"),
        .init(id: 11, name: "flat",              hint: "降号 ♭",     glyph: "\u{E260}"),
        .init(id: 12, name: "natural",           hint: "还原号 ♮",   glyph: "\u{E261}"),
        .init(id: 13, name: "barline-single",    hint: "小节线 |",    glyph: ""),
        .init(id: 14, name: "dot",               hint: "附点 •",     glyph: "\u{E1E7}"),
    ]
}

// MARK: - 集合总览

struct RecordingView: View {
    @ObservedObject var store = StrokeDataStore.shared
    @State private var selectedClass: SymbolClass?
    @State private var showExport = false
    @State private var serverStatus = "Checking..."
    @State private var pushStatus = ""

    let columns = [GridItem(.adaptive(minimum: 100))]

    var body: some View {
        VStack(spacing: 0) {
            // 顶部统计 + 服务器状态
            VStack(spacing: 4) {
                HStack {
                    Text("Total: \(store.totalCount) samples")
                        .font(.headline)
                    Spacer()

                    Button {
                        pushToGit()
                    } label: {
                        Label("Push Git", systemImage: "arrow.up.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled(pushStatus == "Pushing...")

                    Button {
                        showExport = true
                    } label: {
                        Label("Export", systemImage: "square.and.arrow.up")
                    }
                    .disabled(store.totalCount == 0)
                }

                HStack {
                    Circle()
                        .fill(serverStatus.contains("Online") ? .green : .red)
                        .frame(width: 8, height: 8)
                    Text(serverStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !pushStatus.isEmpty {
                        Text(pushStatus)
                            .font(.caption2)
                            .foregroundStyle(pushStatus.contains("Done") ? .green : .orange)
                    }
                }
            }
            .padding()
            .onAppear {
                CloudSync.shared.checkStatus { serverStatus = $0 }
            }

            // 进度总览
            ProgressView(value: Double(store.totalCount), total: Double(SymbolClass.phase1.count * 30))
                .tint(.blue)
                .padding(.horizontal)

            // 类别网格
            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(SymbolClass.phase1) { cls in
                        ClassCard(cls: cls, count: store.count(for: cls.name))
                            .onTapGesture { selectedClass = cls }
                    }
                }
                .padding()
            }
        }
        .fullScreenCover(item: $selectedClass) { cls in
            DrawForClassView(symbolClass: cls, store: store) {
                selectedClass = nil
                store.syncFromServer()  // 回来时刷新服务器计数
            }
        }
        .sheet(isPresented: $showExport) {
            ShareSheet(url: store.exportURL)
        }
    }

    private func pushToGit() {
        pushStatus = "Pushing..."
        CloudSync.shared.triggerGitPush { success in
            pushStatus = success ? "Done! ✓" : "Failed ✗"
            // 3 秒后清除状态
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                pushStatus = ""
            }
        }
    }
}

struct ClassCard: View {
    let cls: SymbolClass
    let count: Int

    var body: some View {
        VStack(spacing: 6) {
            if !cls.glyph.isEmpty {
                Text(cls.glyph)
                    .font(.custom("Bravura", size: 36))
                    .frame(height: 44)
            } else {
                Image(systemName: "music.note")
                    .font(.title)
                    .frame(height: 44)
                    .foregroundStyle(.secondary)
            }

            Text(cls.hint)
                .font(.caption2)
                .lineLimit(1)

            // 数量 + 进度条
            Text("\(count)")
                .font(.system(.title3, design: .rounded, weight: .bold))
                .foregroundStyle(count >= 30 ? .green : count > 0 ? .blue : .secondary)

            ProgressView(value: Double(min(count, 30)), total: 30)
                .tint(count >= 30 ? .green : .blue)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(count >= 30 ? Color.green.opacity(0.05) : Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(count >= 30 ? Color.green.opacity(0.3) : Color.gray.opacity(0.2))
        )
    }
}

// MARK: - 单类别录制

struct DrawForClassView: View {
    let symbolClass: SymbolClass
    @ObservedObject var store: StrokeDataStore
    var onDismiss: () -> Void

    @State private var drawing = PKDrawing()
    @State private var sessionCount = 0  // 本次进入后新画的数量
    @State private var baseCount = 0     // 进入时服务器的数量

    var count: Int { baseCount + sessionCount }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // 提示
                HStack(spacing: 16) {
                    if !symbolClass.glyph.isEmpty {
                        Text(symbolClass.glyph)
                            .font(.custom("Bravura", size: 50))
                    }
                    VStack(alignment: .leading) {
                        Text(symbolClass.hint)
                            .font(.title2.bold())
                        Text("\(count) collected")
                            .font(.caption)
                            .foregroundStyle(count >= 30 ? .green : .secondary)
                    }
                    Spacer()
                }
                .padding()
                .onAppear {
                    baseCount = store.count(for: symbolClass.name)
                }

                Divider()

                // 画布
                ZStack {
                    Color.white
                    DrawingCanvas(drawing: $drawing) { _ in }
                }

                Divider()

                // 操作：Save 和 Undo 在左侧（左手方便），Clear 在右侧
                HStack(spacing: 16) {
                    Button {
                        saveCurrentDrawing()
                    } label: {
                        Label("Save (\(count + 1))", systemImage: "checkmark.circle.fill")
                            .font(.headline)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(drawing.strokes.isEmpty)

                    Button("Undo") {
                        store.deleteLast(for: symbolClass.name)
                    }
                    .foregroundStyle(.red)
                    .disabled(count == 0)

                    Spacer()

                    Button("Clear") {
                        drawing = PKDrawing()
                    }
                    .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle(symbolClass.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { onDismiss() }
                }
            }
        }
    }

    private func saveCurrentDrawing() {
        let strokes = extractStrokes(drawing.strokes)
        let sample = SavedSample(
            label: symbolClass.name,
            classId: symbolClass.id,
            strokes: strokes
        )
        store.add(sample)
        sessionCount += 1

        // 自动上传到服务器（后台，不阻塞）
        CloudSync.shared.uploadSample(
            label: symbolClass.name,
            classId: symbolClass.id,
            strokes: strokes
        )

        drawing = PKDrawing()
    }

    private func extractStrokes(_ strokes: [PKStroke]) -> [[[Double]]] {
        strokes.map { stroke in
            (0..<stroke.path.count).map { i in
                let p = stroke.path[i]
                return [
                    round(Double(p.location.x) * 100) / 100,
                    round(Double(p.location.y) * 100) / 100,
                    round(Double(p.force) * 1000) / 1000,
                ]
            }
        }
    }
}

// MARK: - 分享

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
