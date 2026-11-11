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


def sqtop_steps():
    """Step IDs of sqtop monitoring steps currently running in your jobs."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-s", "-h", "-o", "%i %j"], capture_output=True, text=True).stdout
    # when run via sbatch, steps in the test's own job are skipped: an sqtop you have open monitors it
    # too, and anything in it ends with the job anyway
    own = os.environ.get("SLURM_JOB_ID")
    return {ln.split()[0] for ln in out.splitlines() if len(ln.split()) == 2 and ln.split()[1].startswith(STEP_NAME)
            and not (own and ln.split()[0].startswith(own + "."))}


async def drive(m, size, out):
    app = SqTop(m)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(9)
        shots = [("main", [])]
        app.save_screenshot(f"{out}/1_main.svg")
        steps = [
            ("1b_procs_view_default_sort", ["v", "v"], 1),
            ("2_selected", ["down", "down"], 1),
            ("3_metrics", ["enter"], 4),
            ("4_log", ["escape", "l"], 2),
            ("5_env", ["escape", "e"], 6),
            ("6_compact_sorted", ["escape", "o", "G", "c"], 1),
            ("6b_procs_view", ["v", "a", "home"], 1),
            ("6c_proc_metrics", ["enter"], 3),
            ("6d_jobs_view", ["escape", "v"], 1),
            ("7_help", ["h"], 0.5),
            ("8_zoomed", ["x", "f", "+", "+", "+"], 1),
            ("8b_settings", ["s", "down", "down", "right", "right", "left", "down", "left"], 1),
            ("8c_settings_applied", ["escape"], 8),
            ("9_gpus", ["g"], 4),
            ("10_gpus_nodes", ["n", "down"], 0.5),
            ("11_gpu_type", ["n", "down", "down", "enter"], 1),
        ]
        for name, keys, wait in steps:
            for k in keys:
                await pilot.press(k)
            await pilot.pause(wait)
            app.save_screenshot(f"{out}/{name}.svg")
            shots.append((name, keys))
        assert (m.a.gpu_ms, m.a.proc_s) == (2000, 1), (m.a.gpu_ms, m.a.proc_s)
        assert len(app.screen_stack) >= 2, f"screen stack collapsed: {app.screen_stack}"
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="170x60")
    ap.add_argument("--out", default="smoke-out")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)

    args = argparse.Namespace(gpu_ms=1000, proc_s=2.0, squeue_s=30.0, cluster_s=60.0, user=os.environ["USER"], no_history=False,
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

    before = sqtop_steps()   # e.g. from an sqtop you have open: not ours to clean up
    m = Monitor(args)
    threading.Thread(target=m.squeue_loop, daemon=True).start()
    try:
        shots = asyncio.run(drive(m, (w, h), a.out))
        print(f"TUI: {len(shots)} screens rendered -> {a.out}/")
    finally:
        m.shutdown()
    for _ in range(12):
        if not sqtop_steps() - before:
            break
        time.sleep(5)
    left = sqtop_steps() - before
    assert not left, f"{len(left)} {STEP_NAME} steps left running: {sorted(left)}"
    print("no monitoring steps left behind: ok")


if __name__ == "__main__":
    main()
