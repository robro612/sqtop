"""sqtop: nvitop-style live view of the GPUs under your SLURM jobs.

Usage: sqtop [--gpu-ms 1000] [--proc-s 2] [--squeue-s 30] [--user U] [--once]

How it works:
  - squeue (every --squeue-s) finds your jobs.
  - For each RUNNING GPU job, ONE long-lived `srun --overlap --jobid=<id>` step
    streams nvidia-smi (loop mode, every --gpu-ms), nvidia-smi pmon (per-process
    %SM / %GMBW), and the job's cgroup CPU/RAM + GPU processes (every --proc-s).
    The step shares the job's allocation (no new resources), only sees that
    job's GPUs, and ends when the job ends or sqtop exits (SIGTERM cancels only
    our step, never the job).
  - History from before sqtop started is backfilled from the cluster's
    Prometheus (1-minute resolution) so charts cover the whole job.

Read-only: sqtop never signals processes or cancels jobs. Press h for keys.

Agent / script mode: `sqtop --json [--window 30m] [--jobs ID,..] [--log-lines 5]`
prints ONE JSON document and exits (a few seconds; nothing keeps running). Raw data
only, no judgements. Keys with no value are omitted. Shape:
  {time, user, window_s, note, errors?, jobs: [
    {id, raw_id?, name, state, reason?, partition, gres?, nodes?, cpus, mem_req,
     limit_s, submit, start?, workdir, batch,              # SLURM (squeue + scontrol)
     elapsed_s, left_s,                                      # running jobs only, from here on
     log: {path, age_s, tail: [last lines], stderr_path?},   # age_s = seconds since last write
     gpus: [{node?, idx, model, util, mem_gb, mem_total_gb, power_w, temp_c,
             window: {mean, p10, p90, max, mem_max_gb, covered_s}}],   # util in %, over --window
     cpu: {used, alloc},  ram_gb: {used, cache, limit},      # job cgroup; used excludes page cache
     procs: [{pid, ppid, n?, cpu_pct, rss_gb, gpu_mem_gb?, age_s, ports?, cmd, node?}],
     procs_omitted?, errors?}]}
  procs: the job's processes (SLURM internals and the batch shell excluded); entries with an
  identical command line are merged (n = count; cpu_pct and gpu_mem_gb summed, rss_gb = largest
  member, since workers share memory). ports = listening TCP ports.
  cpu_pct is sampled over 1 s. Pending jobs carry only the SLURM fields.
"""
import argparse
import json
import os
import threading
import time

from rich.console import Console, Group

from .agent import collect_json, parse_window
from .core import Monitor
from .render import proc_rows, render_charts, render_devices, render_procs, sort_rows
from .tui import SqTop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-ms", type=int, default=1000, help="nvidia-smi sample interval in ms (default 1000)")
    ap.add_argument("--proc-s", type=float, default=2.0, help="process + cgroup CPU/RAM interval (default 2s)")
    ap.add_argument("--squeue-s", type=float, default=30.0, help="squeue interval (default 30s)")
    ap.add_argument("--user", default=os.environ.get("USER"))
    ap.add_argument("--no-history", action="store_true", help="skip Prometheus backfill")
    ap.add_argument("--once", action="store_true", help="print one snapshot (after warm-up) and exit")
    js = ap.add_argument_group("agent / script mode")
    js.add_argument("--json", action="store_true", help="print one JSON document describing your jobs and exit")
    js.add_argument("--window", type=parse_window, default=parse_window("30m"),
                    help="GPU history window for the window stats (default 30m; e.g. 900, 10m, 2h)")
    js.add_argument("--jobs", help="comma-separated job ids to include (array base id matches all tasks)")
    js.add_argument("--log-lines", type=int, default=5, help="log lines per running job (default 5, 0 for none)")
    js.add_argument("--max-procs", type=int, default=15, help="process entries per job (default 15)")
    js.add_argument("--no-procs", action="store_true",
                    help="skip the in-job probe (no processes/ports/CPU/RAM; GPU values from Prometheus, <=1 min old)")
    js.add_argument("--pretty", action="store_true", help="indent the JSON")
    a = ap.parse_args()

    if a.json:
        doc = collect_json(a)
        print(json.dumps(doc, indent=1 if a.pretty else None, separators=None if a.pretty else (",", ":"),
                         ensure_ascii=False))
        return

    m = Monitor(a)
    if a.once:
        console = Console()
        try:
            m.poll_squeue()
            time.sleep(max(5.0, 2 * a.proc_s + 2))
            w = console.width
            console.print(Group(*render_devices(m, w, False, None)))
            console.print(Group(*render_charts(m, w, None, 1, 3)))
            rows = sort_rows(proc_rows(m), "job", False)
            console.print(Group(*render_procs(rows, w, len(rows) + 6, None, 0, 0, "job", False, a.user)))
        finally:
            m.shutdown()
        return

    threading.Thread(target=m.squeue_loop, daemon=True).start()
    try:
        SqTop(m).run()
    finally:
        print("sqtop: stopping streams…", flush=True)
        m.shutdown()


if __name__ == "__main__":
    main()
