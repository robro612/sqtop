"""Headless smoke test against your real jobs: drives every TUI screen, checks --json, and checks that
no monitoring steps are left behind. Needs at least one running job.

    uv run python tests/smoke.py [--size 170x60] [--out smoke-out]

Screenshots (SVG) of each screen are written to --out for eyeballing.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time

from sqtop.agent import collect_json, parse_window
from sqtop.core import STEP_NAME, Monitor
from sqtop.tui import SqTop


def leftover_steps():
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-s", "-h", "-o", "%j"], capture_output=True, text=True).stdout
    return sum(1 for ln in out.split() if ln.startswith(STEP_NAME))


async def drive(m, size, out):
    app = SqTop(m)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(9)
        shots = [("main", [])]
        app.save_screenshot(f"{out}/1_main.svg")
        steps = [
            ("2_selected", ["down", "down"], 1),
            ("3_metrics", ["enter"], 4),
            ("4_log", ["escape", "l"], 2),
            ("5_env", ["escape", "e"], 6),
            ("6_compact_sorted", ["escape", "o", "G", "c"], 1),
            ("7_help", ["h"], 0.5),
            ("8_zoomed", ["x", "f", "-", "-", "-"], 1),
        ]
        for name, keys, wait in steps:
            for k in keys:
                await pilot.press(k)
            await pilot.pause(wait)
            app.save_screenshot(f"{out}/{name}.svg")
            shots.append((name, keys))
        assert len(app.screen_stack) >= 2, f"screen stack collapsed: {app.screen_stack}"
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="170x60")
    ap.add_argument("--out", default="smoke-out")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)

    args = argparse.Namespace(gpu_ms=1000, proc_s=2.0, squeue_s=30.0, user=os.environ["USER"], no_history=False,
                              once=False, json=True, window=parse_window("30m"), jobs=None, log_lines=3, max_procs=15,
                              no_procs=False, pretty=False)
    t = time.time()
    doc = collect_json(args)
    running = [j for j in doc["jobs"] if j["state"] == "RUNNING"]
    print(f"--json: {len(doc['jobs'])} jobs ({len(running)} running), {len(json.dumps(doc))} bytes, {time.time() - t:.1f}s")
    assert not doc.get("errors"), doc.get("errors")
    if not running:
        sys.exit("no running jobs: nothing to drive the TUI against")
    for j in running:
        assert "gpus" in j or "cpu" in j, f"no data for {j['id']}: {j.get('errors')}"

    m = Monitor(args)
    threading.Thread(target=m.squeue_loop, daemon=True).start()
    try:
        shots = asyncio.run(drive(m, (w, h), a.out))
        print(f"TUI: {len(shots)} screens rendered -> {a.out}/")
    finally:
        m.shutdown()
    for _ in range(12):
        if leftover_steps() == 0:
            break
        time.sleep(5)
    n = leftover_steps()
    assert n == 0, f"{n} {STEP_NAME} steps left running"
    print("no monitoring steps left behind: ok")


if __name__ == "__main__":
    main()
