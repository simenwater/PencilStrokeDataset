"""
collect_server.py — 笔触数据采集 API
部署到 Contabo 服务器 (161.97.72.212)

启动: uvicorn collect_server:app --host 0.0.0.0 --port 8400
测试: curl http://161.97.72.212:8400/api/status
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import json

app = FastAPI(title="PencilStroke Collector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("collected")
DATA_DIR.mkdir(exist_ok=True)


class StrokeSample(BaseModel):
    writer: str = "anonymous"
    device: str = ""
    label: str
    class_id: int
    strokes: list  # [[[x, y, force], ...], ...]


class BatchUpload(BaseModel):
    writer: str = "anonymous"
    device: str = ""
    samples: list[StrokeSample]


@app.get("/api/status")
def status():
    """查看采集状态"""
    stats = {}
    total = 0
    writers = set()
    for f in DATA_DIR.glob("*.jsonl"):
        writer = f.stem
        writers.add(writer)
        count = sum(1 for _ in open(f))
        stats[writer] = count
        total += count

    # 按类统计
    class_counts = {}
    for f in DATA_DIR.glob("*.jsonl"):
        for line in open(f):
            sample = json.loads(line)
            label = sample.get("label", "?")
            class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "total_samples": total,
        "writers": len(writers),
        "per_writer": stats,
        "per_class": dict(sorted(class_counts.items())),
    }


@app.post("/api/upload")
def upload_single(sample: StrokeSample):
    """上传单个样本"""
    save_sample(sample.writer, sample)
    return {"status": "ok", "label": sample.label}


@app.post("/api/upload_batch")
def upload_batch(batch: BatchUpload):
    """批量上传（iPad 一次性同步所有新样本）"""
    count = 0
    for sample in batch.samples:
        if not sample.writer:
            sample.writer = batch.writer
        if not sample.device:
            sample.device = batch.device
        save_sample(sample.writer or batch.writer, sample)
        count += 1
    return {"status": "ok", "count": count}


@app.get("/api/export")
def export_all():
    """导出全部数据为训练用 JSON"""
    all_samples = []
    for f in DATA_DIR.glob("*.jsonl"):
        for line in open(f):
            all_samples.append(json.loads(line))

    return {
        "source": "PencilStroke Collector",
        "date": datetime.now().isoformat(),
        "sample_count": len(all_samples),
        "point_format": ["x", "y", "force"],
        "samples": all_samples,
    }


def save_sample(writer: str, sample: StrokeSample):
    """每个 writer 一个 JSONL 文件，追加写入"""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in writer)
    filepath = DATA_DIR / f"{safe_name}.jsonl"
    record = {
        "label": sample.label,
        "class_id": sample.class_id,
        "strokes": sample.strokes,
        "device": sample.device,
        "time": datetime.now().isoformat(),
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
