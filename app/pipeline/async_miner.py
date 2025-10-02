
from __future__ import annotations
import os, time, threading, queue, requests, json, random
from typing import List, Dict, Optional

"""
Async miner client that:
- polls multiple validator endpoints concurrently
- respects a UID blacklist
- uses the local generation service to fulfill jobs
- fan-outs without waiting serially
This is an optional helper; keep your service contract unchanged.
"""

class MinerWorker(threading.Thread):
    def __init__(self, name: str, task_q: "queue.Queue[dict]", service_url: str):
        super().__init__(daemon=True, name=name)
        self.q = task_q
        self.service_url = service_url

    def run(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                prompt = job["prompt"]
                # choose endpoint based on job["type"]
                if job.get("video", False):
                    r = requests.post(f"{self.service_url}/generate_video/", data={"prompt": prompt}, timeout=60)
                    if r.status_code == 200:
                        job["on_result"](r.content, "video/mp4")
                else:
                    r = requests.post(f"{self.service_url}/generate/", data={"prompt": prompt}, timeout=60)
                    if r.status_code == 200:
                        job["on_result"](r.content, "application/zip")
            except Exception as e:
                job.get("on_error", lambda *_: None)(str(e))
            finally:
                self.q.task_done()

class AsyncMiner:
    def __init__(self, validators: List[Dict], service_url: str = "http://127.0.0.1:8093", blacklist: Optional[List[int]] = None, workers: int = 4):
        self.validators = [v for v in validators if int(v.get("uid", -1)) not in set(blacklist or [])]
        self.service_url = service_url
        self.q = queue.Queue(maxsize=workers*2)
        self.workers = [MinerWorker(f"w{i}", self.q, self.service_url) for i in range(workers)]
        for w in self.workers:
            w.start()

    def poll_once(self):
        # pull tasks from all validators without waiting on each other
        for v in self.validators:
            try:
                task = requests.get(v["pull_url"], timeout=5).json()
                if not task or "prompt" not in task:
                    continue
                def on_result(payload: bytes, mime: str, v=v, task=task):
                    # fan-out callback: send back immediately
                    try:
                        requests.post(v["push_url"], files={"file": ("res", payload, mime)}, data={"task_id": task.get("id","")}, timeout=10)
                    except Exception:
                        pass
                self.q.put({"prompt": task["prompt"], "video": bool(task.get("video")), "on_result": on_result})
            except Exception:
                continue

    def run_forever(self, interval_s: float = 0.5):
        while True:
            self.poll_once()
            time.sleep(interval_s)
