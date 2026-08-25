#!/usr/bin/env python3
"""
Resumable, self-healing downloader for Wan2.2-I2V-A14B-Diffusers.

Strategy: per-file download with curl (NOT hf_hub / hf_transfer).

Why curl instead of the hf_hub native downloader:
  - The hf_hub native downloader silently hangs forever on dead connections
    (we observed multiple multi-hour stalls with no exception raised), and
    any supervisor-level watchdog was too slow to notice because computing
    total bytes via rglob() chokes on the loaded filesystem.
  - curl has first-class support for every failure mode:
      * `-C -`            resume a partial file via HTTP range request
      * `--speed-limit`   abort the transfer if throughput collapses
        `--speed-time`    (i.e. the "dead connection" case) -> curl exits != 0
      * `--retry`         re-attempt transient errors
        `--retry-all-errors`
  - The outer Python loop simply re-runs curl for any file whose size does
    not yet match the expected size from the HF API, so we always resume.

Usage:
  python scripts/download_wan22_diffusers.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
# Canonical location: resolve_video_model_path() looks in
# models/image_to_video/<model_name>/, so we download straight there.
TARGET = Path(__file__).parent.parent / "models/image_to_video/wan22_i2v"

API_URL = f"https://huggingface.co/api/models/{REPO_ID}?blobs=true"
RESOLVE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/"

# Files hf_hub's snapshot_download was configured to ignore. The diffusers
# pipeline never needs them.
IGNORED_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".md", ".JPG")

MAX_FILE_ATTEMPTS = 20          # per-file curl retries before giving up
RETRY_DELAY_SECONDS = 20
CURL_TIMEOUT = 3600             # --max-time per curl invocation (seconds)
STALL_LIMIT = 8192              # bytes/sec below which curl aborts
STALL_SECONDS = 90              # sustained slow window before abort


def fetch_file_list() -> list[tuple[str, int]]:
    """Return [(relative_path, size_in_bytes), ...] from the HF API."""
    r = subprocess.run(
        ["curl", "-sS", "--max-time", "60", API_URL],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    files = []
    for s in data.get("siblings", []):
        if s.get("rfilename", "").endswith(IGNORED_SUFFIXES):
            continue
        files.append((s["rfilename"], s.get("size", 0)))
    return sorted(files)


def fetch_etag_size(url: str) -> int:
    """HEAD request to learn the server-side size for a file."""
    r = subprocess.run(
        ["curl", "-sSI", "--max-time", "30", url],
        capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return -1


def download_file(rel_path: str, expected_size: int, quiet: bool = False) -> bool:
    """Download one file with curl, resuming and retrying. True on success."""
    local = TARGET / rel_path
    local.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE_URL + rel_path

    for attempt in range(1, MAX_FILE_ATTEMPTS + 1):
        # Already complete (possibly by an earlier attempt)? Done.
        if local.is_file() and local.stat().st_size == expected_size:
            if not quiet:
                print(f"OK   {rel_path} ({expected_size/1e9:.2f} GB)", flush=True)
            return True

        if not quiet:
            have = local.stat().st_size if local.is_file() else 0
            percentage = (have / expected_size * 100) if expected_size > 0 else 0
            print(
                f"GET  {rel_path} [{have/1e9:.2f}/{expected_size/1e9:.2f} GB] ({percentage:.1f}%) "
                f"attempt {attempt}/{MAX_FILE_ATTEMPTS}",
                flush=True,
            )
        
        # Start timing for speed calculation
        start_time = time.time()
        start_size = local.stat().st_size if local.is_file() else 0
        
        cmd = [
            "curl", "-sS", "-L", "--fail", "-C", "-",
            "--connect-timeout", "30",
            "--max-time", str(CURL_TIMEOUT),
            "--speed-limit", str(STALL_LIMIT),
            "--speed-time", str(STALL_SECONDS),
            "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
            "-o", str(local),
            url,
        ]
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        end_time = time.time()
        end_size = local.stat().st_size if local.is_file() else 0
        
        # Calculate speed and update status if file was downloaded
        if not quiet:
            if rc == 0 and local.is_file() and local.stat().st_size == expected_size:
                print(f"OK   {rel_path} ({expected_size/1e9:.2f} GB)", flush=True)
            else:
                # Calculate transfer speed for partial download
                transferred = end_size - start_size
                duration = end_time - start_time
                if duration > 0:
                    speed_mbps = (transferred / 1024 / 1024) / duration
                    print(f"     Transferred: {transferred/1024/1024:.2f} MB in {duration:.1f}s "
                          f"at {speed_mbps:.2f} MB/s", flush=True)
        
        if rc == 0 and local.is_file() and local.stat().st_size == expected_size:
            if not quiet:
                print(f"OK   {rel_path} ({expected_size/1e9:.2f} GB)", flush=True)
            return True
        # Transient error / stall / partial write: curl already resumed as far
        # as it could; the loop re-invokes with `-C -` which continues.
        if rc != 0 and not quiet:
            print(f"     curl rc={rc}; will retry in {RETRY_DELAY_SECONDS}s", flush=True)
        time.sleep(RETRY_DELAY_SECONDS)

    return False


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Fetching file list from {API_URL} ...", flush=True)
    files = fetch_file_list()
    total = sum(sz for _, sz in files)
    print(f"{len(files)} files, {total/1e9:.1f} GB", flush=True)

    # Progress loop: keep sweeping the file list until every file matches.
    round_no = 0
    while True:
        round_no += 1
        print(f"\n===== Sweep {round_no} ({time.strftime('%H:%M:%S')}) =====", flush=True)
        pending = 0
        for rel_path, expected in files:
            ok = download_file(rel_path, expected, quiet=(round_no > 1))
            if not ok:
                pending += 1
        if pending == 0:
            print("\nWAN DOWNLOAD COMPLETE — all files present.", flush=True)
            return 0
        print(f"\n{pending} files still incomplete; resuming in "
              f"{RETRY_DELAY_SECONDS}s.", flush=True)
        time.sleep(RETRY_DELAY_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
