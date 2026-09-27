"""
Approach V2 Blocking: Live Terminal Monitor
Read-only monitoring script for generate_full_candidates.py
Continuously refreshes every 5 seconds without modifying or stopping the running generation process.
"""

import os
import re
import sys
import time
import glob
import psutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Path definitions
SRC_DIR = Path(__file__).resolve().parent
APPROACH_ROOT = SRC_DIR.parent
CHUNKS_DIR = APPROACH_ROOT / "candidates" / "chunks_full"
FINAL_PARQUET = APPROACH_ROOT / "candidates" / "train_candidates.parquet"
TOTAL_S1_TARGET = 2206821


def find_latest_log_file() -> Path:
    """Finds the most recent task log file for generate_full_candidates.py"""
    brain_dir = Path.home() / ".gemini" / "antigravity-ide" / "brain"
    log_candidates = list(brain_dir.glob("*/.system_generated/tasks/task-*.log"))
    
    matching_logs = []
    for log_path in log_candidates:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(500)
                if "FULL-SCALE CANDIDATE GENERATION" in content:
                    matching_logs.append((log_path.stat().st_mtime, log_path))
        except Exception:
            continue
            
    if matching_logs:
        matching_logs.sort(reverse=True)
        return matching_logs[0][1]
    return None


def format_seconds(seconds: float) -> str:
    """Formats seconds into human-readable Xh Ym or Xm Ys format."""
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def is_generator_running() -> bool:
    """Checks if generate_full_candidates.py is currently running in OS process table."""
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmd_str = " ".join(cmdline)
                if "generate_full_candidates.py" in cmd_str:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return False


def parse_log_metrics(log_file: Path) -> dict:
    """Parses progress metrics from the latest log lines."""
    metrics = {
        "s1_processed": 0,
        "pct": 0.0,
        "candidates": 0,
        "avg_cands": 0.0,
        "speed": 0.0,
        "eta_seconds": None,
        "completed": False,
        "failed": False
    }

    if not log_file or not log_file.exists():
        return metrics

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        batch_re = re.compile(
            r"\[Batch\s+(\d+)\s+-\s+([\d\.]+)%\]\s+Processed\s+([\d,]+)\s+/\s+[\d,]+\s+S1\s+\|\s+Candidates Generated:\s+([\d,]+)\s+\(Avg:\s+([\d\.]+)\)\s+\|\s+Speed:\s+([\d\.]+)\s+S1/s\s+\|\s+Batch Time:\s+[\d\.]+s\s+\|\s+ETA:\s+([\d\.]+)m"
        )

        for line in reversed(lines):
            if "APPROACH V2: FULL-SCALE CANDIDATE GENERATION COMPLETE" in line:
                metrics["completed"] = True
                metrics["pct"] = 100.0
            
            m = batch_re.search(line)
            if m and metrics["s1_processed"] == 0:
                metrics["pct"] = float(m.group(2))
                metrics["s1_processed"] = int(m.group(3).replace(",", ""))
                metrics["candidates"] = int(m.group(4).replace(",", ""))
                metrics["avg_cands"] = float(m.group(5))
                metrics["speed"] = float(m.group(6))
                metrics["eta_seconds"] = float(m.group(7)) * 60.0

    except Exception:
        pass

    return metrics


def main():
    print("Initializing Approach V2 Candidate Generation Monitor...", flush=True)
    time.sleep(1)

    log_file = find_latest_log_file()
    start_time = log_file.stat().st_mtime if log_file else time.time()

    while True:
        # Re-check log file if needed
        if not log_file or not log_file.exists():
            log_file = find_latest_log_file()
            if log_file:
                start_time = log_file.stat().st_mtime

        metrics = parse_log_metrics(log_file)

        # Check process status
        running = is_generator_running()
        if metrics["completed"]:
            status_str = "COMPLETED"
        elif running:
            status_str = "RUNNING"
        else:
            if metrics["s1_processed"] >= TOTAL_S1_TARGET:
                status_str = "COMPLETED"
            elif FINAL_PARQUET.exists():
                status_str = "COMPLETED"
            else:
                status_str = "STOPPED / WAITING"

        # Check chunk files
        chunk_files = sorted(list(CHUNKS_DIR.glob("train_candidates_chunk_*.parquet")))
        num_chunks = len(chunk_files)
        latest_chunk_name = chunk_files[-1].name if chunk_files else "None"

        # Calculate timing
        elapsed_sec = time.time() - start_time
        elapsed_str = format_seconds(elapsed_sec)
        eta_str = format_seconds(metrics["eta_seconds"]) if metrics["eta_seconds"] is not None else "N/A"

        # Format output string
        output = [
            "=" * 80,
            "APPROACH V2: CANDIDATE GENERATION LIVE MONITOR (READ-ONLY)",
            "=" * 80,
            f"S1 Progress        : {metrics['s1_processed']:,} / {TOTAL_S1_TARGET:,} ({metrics['pct']:.2f}%)",
            f"Candidates         : {metrics['candidates']:,}",
            f"Avg Candidates/S1  : {metrics['avg_cands']:.2f}",
            f"Speed              : {metrics['speed']:.1f} S1/s",
            f"Elapsed            : {elapsed_str}",
            f"ETA                : {eta_str}",
            f"Chunks Written     : {num_chunks}",
            f"Latest Chunk       : {latest_chunk_name}",
            f"Process Status     : {status_str}",
            "=" * 80,
            "(Refreshes every 5 seconds. Press Ctrl+C to exit monitor)\n"
        ]

        # Clear screen and display updated block in-place
        os.system("cls" if os.name == "nt" else "clear")
        print("\n".join(output), flush=True)

        if status_str == "COMPLETED":
            print("✅ Full-scale candidate generation completed successfully!", flush=True)
            break

        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
