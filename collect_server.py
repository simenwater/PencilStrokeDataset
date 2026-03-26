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
import subprocess
import threading

app = FastAPI(title="PencilStroke Collector", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("collected")
DATA_DIR.mkdir(exist_ok=True)

# 自动推送计数器
_sample_counter = 0
_PUSH_EVERY = 10  # 每 10 个样本自动 push 一次


class StrokeSample(BaseModel):
    writer: str = "anonymous"
    device: str = ""
    label: str
    class_id: int
    strokes: list


class BatchUpload(BaseModel):
    writer: str = "anonymous"
    device: str = ""
    samples: list[StrokeSample]


@app.get("/api/status")
def status():
    stats = {}
    total = 0
    class_counts = {}
    for f in DATA_DIR.glob("*.jsonl"):
        writer = f.stem
        count = sum(1 for _ in open(f))
        stats[writer] = count
        total += count
        for line in open(f):
            sample = json.loads(line)
            label = sample.get("label", "?")
            class_counts[label] = class_counts.get(label, 0) + 1
    return {
        "total_samples": total,
        "writers": len(stats),
        "per_writer": stats,
        "per_class": dict(sorted(class_counts.items())),
    }


@app.post("/api/upload")
def upload_single(sample: StrokeSample):
    save_sample(sample.writer, sample)
    maybe_git_push()
    return {"status": "ok", "label": sample.label}


@app.post("/api/upload_batch")
def upload_batch(batch: BatchUpload):
    count = 0
    for sample in batch.samples:
        if not sample.writer:
            sample.writer = batch.writer
        if not sample.device:
            sample.device = batch.device
        save_sample(sample.writer or batch.writer, sample)
        count += 1
    maybe_git_push()
    return {"status": "ok", "count": count}


@app.get("/api/export")
def export_all():
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


@app.post("/api/push")
def force_push():
    """手动触发 git push"""
    result = git_push()
    return {"status": "ok" if result else "failed"}


def save_sample(writer: str, sample: StrokeSample):
    global _sample_counter
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
    _sample_counter += 1


def maybe_git_push():
    global _sample_counter
    if _sample_counter >= _PUSH_EVERY:
        _sample_counter = 0
        threading.Thread(target=git_push, daemon=True).start()


def git_push():
    try:
        # 统计当前数据
        total = sum(sum(1 for _ in open(f)) for f in DATA_DIR.glob("*.jsonl"))
        writers = len(list(DATA_DIR.glob("*.jsonl")))

        subprocess.run(["git", "add", "collected/"], check=True, cwd="/opt/pencilstroke")
        subprocess.run(
            ["git", "commit", "-m", f"Auto: {total} samples from {writers} writers @ {datetime.now():%Y-%m-%d %H:%M}"],
            check=True, cwd="/opt/pencilstroke"
        )
        subprocess.run(["git", "push", "origin", "main"], check=True, cwd="/opt/pencilstroke",
                        timeout=30)
        print(f"[git] Pushed: {total} samples")
        return True
    except Exception as e:
        print(f"[git] Push failed: {e}")
        return False
