import time, requests, os, json

BASE = os.environ.get("BASE", "http://127.0.0.1:8093")

def bench(prompt="a red vintage robot toy"):
    t0 = time.time()
    r = requests.post(f"{BASE}/generate_video/", data={"prompt": prompt}, timeout=120)
    vid_ok = r.status_code == 200 and r.headers.get("content-type","").startswith("video/mp4")
    v_ms = (time.time() - t0) * 1000

    t1 = time.time()
    r2 = requests.post(f"{BASE}/generate/", data={"prompt": prompt}, timeout=120)
    zip_ok = r2.status_code in (200, 204)
    z_ms = (time.time() - t1) * 1000

    print(json.dumps({"video_ms": v_ms, "video_ok": vid_ok, "generate_ms": z_ms, "generate_ok": zip_ok}, indent=2))

if __name__ == "__main__":
    bench()