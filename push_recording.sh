#!/bin/bash
# 用法: ./push_recording.sh ~/Downloads/ipad_recordings_*.json
# 把 iPad AirDrop 过来的录制文件推送到 GitHub

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

if [ $# -eq 0 ]; then
    echo "用法: ./push_recording.sh <json文件路径>"
    echo "  例: ./push_recording.sh ~/Downloads/ipad_recordings_2026-03-27_0130.json"
    exit 1
fi

for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "[skip] 文件不存在: $f"
        continue
    fi

    BASENAME=$(basename "$f")
    cp "$f" "$REPO_DIR/data/$BASENAME"
    echo "[copied] $BASENAME → data/"

    # 显示样本数
    COUNT=$(python3 -c "import json; d=json.load(open('data/$BASENAME')); print(d.get('sample_count', '?'))" 2>/dev/null)
    echo "[info] $COUNT samples"
done

git add data/
git commit -m "Add iPad recording: $(ls -1 data/*.json | wc -l | tr -d ' ') files, $(date +%Y-%m-%d)"
git push origin main

echo ""
echo "[done] Pushed to https://github.com/simenwater/PencilStrokeDataset"
echo "Win 端: git pull 即可拿到数据"
