"""Agent / script mode: one JSON document describing your jobs (sqtop --json)."""
import argparse
import json
import os
import pwd
import re
import subprocess
import time
import urllib.parse
import urllib.request

from .core import (
    ANSI_RE,
    JOB_PIDS_SH,
    MONITOR_MARK,
    PDEATH,
    STEP_END_RE,
    STEP_NAME,
    Job,
    fnum,
    gres_short,
    list_jobs,
    parse_dur,
    parse_mem,
    prom_url,
    read_log_lines,
    scontrol_job,
    short_args,
)

# One-shot probe run inside each running job (srun --overlap, one task per node). Samples CPU over 1 s.
#   T|h|<pid> <ticks>   U|h|<pid> <ticks>     per-process utime+stime before / after the 1 s sleep
#   J|h|<t0>|<t1>|<cpu usec 0>|<cpu usec 1>|<mem.current>|<mem.max>|<anon>|<CLK_TCK>   (8 fields)
#   S|h|<pid> <ppid> <rss KiB> <elapsed s> <args>
#   G|h|<query-gpu csv>   P|h|<compute-apps csv>   L|h|<ss -ltnp line>
REMOTE_ONCE = r"""
: MARK
gpu=$1
h=$(hostname -s)
cg=$(sed -n 's/^0:://p' /proc/self/cgroup); jcg=/sys/fs/cgroup${cg%%/step_*}
mystep=$(sed -n 's|.*/\(step_[^/]*\)/.*|\1|p' /proc/self/cgroup)
@JOB_PIDS@
st() {
  [ -n "$pids" ] && awk -v k="$1" -v h="$h" '{l=$0; sub(/^.*\) /,"",l); split(l,f," "); p=FILENAME;
    gsub(/[^0-9]/,"",p); print k "|" h "|" p " " f[12]+f[13]}' $(tr ',' '\n' <<<"$pids" | sed 's|.*|/proc/&/stat|') 2>/dev/null
}
c0=$(awk '/^usage_usec/{print $2}' "$jcg/cpu.stat" 2>/dev/null); t0=$EPOCHREALTIME; st T
sleep 1
c1=$(awk '/^usage_usec/{print $2}' "$jcg/cpu.stat" 2>/dev/null); t1=$EPOCHREALTIME; st U
anon=$(awk '/^anon /{print $2}' "$jcg/memory.stat" 2>/dev/null)
echo "J|$h|$t0|$t1|$c0|$c1|$(cat "$jcg/memory.current" 2>/dev/null)|$(cat "$jcg/memory.max" 2>/dev/null)|$anon|$(getconf CLK_TCK)"
[ -n "$pids" ] && ps -ww -o pid=,ppid=,rss=,etimes=,args= -p "$pids" | awk '$5 != "slurmstepd:"' | sed "s/^ */S|$h|/"
if [ "$gpu" = 1 ]; then
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | sed "s/^/G|$h|/"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/P|$h|/"
fi
ss -Hltnp 2>/dev/null | sed "s/^/L|$h|/"
""".replace("MARK", MONITOR_MARK).replace("@JOB_PIDS@", JOB_PIDS_SH.strip())


BATCH_SHELL_RE = re.compile(r"^/bin/(ba)?sh /var/spool/slurm\S*/slurm_script")


def parse_window(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad window {s!r} (e.g. 900, 30m, 2h)")
    return int(float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)])


def _r(x, nd=1):
    return None if x is None else round(x, nd or None)


def _gb(b, nd=1):
    return None if b is None else round(b / 2**30, nd)


def _clean(o):
    """Drop None / empty values recursively: fewer tokens, nothing ambiguous."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if v not in (None, "", [], {})}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def probe_job(job: Job):
    """One-shot in-job probe -> per-host raw records, or an error string."""
    cmd = [*PDEATH, "srun", "--overlap", f"--jobid={job.jobid}", f"--job-name={STEP_NAME}-json", "--chdir=/",
           "-N", str(job.nnodes), "--ntasks-per-node=1", "--quiet",
           "bash", "-c", REMOTE_ONCE, STEP_NAME, str(int(job.has_gpu))]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=45, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "in-job probe timed out"
    hosts: dict[str, dict] = {}
    for line in p.stdout.splitlines():
        try:
            kind, host, rest = line.split("|", 2)
        except ValueError:
            continue
        h = hosts.setdefault(host, {"T": {}, "U": {}, "S": [], "G": [], "P": [], "L": [], "J": None})
        if kind in ("T", "U"):
            pid, ticks = rest.split()
            h[kind][int(pid)] = float(ticks)
        elif kind == "J":
            h["J"] = rest.split("|")
        elif kind in h:
            h[kind].append(rest)
    err = "; ".join(l for l in p.stderr.strip().splitlines() if not STEP_END_RE.search(l))[:300]
    return hosts, (err or (None if hosts else f"in-job probe returned nothing (exit {p.returncode})"))


def prom_window(prom, uid, window):
    """{(jobid, uuid): {stat: value, "name": .., "node": ..}} from Prometheus over the window."""
    w = f"{window}s"
    sel = f'{{userid="{uid}"}}'
    queries = {
        "mean": f"avg_over_time(nvidia_gpu_duty_cycle{sel}[{w}])",
        "p10": f"quantile_over_time(0.1, nvidia_gpu_duty_cycle{sel}[{w}])",
        "p90": f"quantile_over_time(0.9, nvidia_gpu_duty_cycle{sel}[{w}])",
        "max": f"max_over_time(nvidia_gpu_duty_cycle{sel}[{w}])",
        "n": f"count_over_time(nvidia_gpu_duty_cycle{sel}[{w}])",
        "mem_max": f"max_over_time(nvidia_gpu_memory_used_bytes{sel}[{w}])",
        "util_last": f"last_over_time(nvidia_gpu_duty_cycle{sel}[5m])",
        "mem_last": f"last_over_time(nvidia_gpu_memory_used_bytes{sel}[5m])",
        "mem_total": f"last_over_time(nvidia_gpu_memory_total_bytes{sel}[5m])",
    }
    out: dict[tuple, dict] = {}

    def one(k, q):
        url = f"{prom.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({"query": q})
        with urllib.request.urlopen(url, timeout=15) as r:
            return k, json.load(r)["data"]["result"]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(queries)) as ex:
        for k, res in ex.map(lambda kv: one(*kv), queries.items()):
            for r in res:
                lab = r["metric"]
                d = out.setdefault((lab.get("jobid"), lab.get("uuid")), {})
                d[k] = float(r["value"][1])
                d["name"] = lab.get("name", "").removeprefix("NVIDIA ")
                d["node"] = lab.get("instance", "").split(".")[0].split(":")[0]
    return out


def job_record(job: Job, info: dict, probe, prom: dict, a):
    """Everything known about one job, as plain data."""
    now = time.time()
    rec = {
        "id": job.display,
        "raw_id": job.jobid if job.jobid != job.display else None,
        "name": job.name,
        "state": job.state,
        "reason": job.reason if job.state != "RUNNING" and job.reason not in ("None", "") else None,
        "partition": job.partition,
        "gres": gres_short(job.gres) if job.has_gpu else None,
        "nodes": job.nodes if job.state == "RUNNING" else None,
        "cpus": job.cpus,
        "mem_req": job.mem_req,
        "limit_s": parse_dur(job.limit),
        "submit": info.get("SubmitTime"),
        "start": info.get("StartTime") if info.get("StartTime") not in ("Unknown", "None") else None,
        "workdir": info.get("WorkDir"),
        "batch": info.get("Command"),
    }
    if job.state != "RUNNING":
        return rec
    el = job.elapsed_now()
    rec["elapsed_s"] = _r(el, 0)
    rec["left_s"] = _r(job.left_now(), 0)
    out = info.get("StdOut")
    if out:
        lg = {"path": out}
        try:
            lg["age_s"] = int(now - os.path.getmtime(out))
            if a.log_lines:
                lines = [ANSI_RE.sub("", ln).strip()[:300] for ln in read_log_lines(out, 4 * a.log_lines)]
                lg["tail"] = [ln for ln in lines if ln][-a.log_lines:]
        except OSError as e:
            lg["error"] = e.strerror
        if info.get("StdErr") and info["StdErr"] != out:
            lg["stderr_path"] = info["StdErr"]
        rec["log"] = lg

    hosts, perr = probe if probe else ({}, None)
    errors = [perr] if perr else []

    # GPUs: live values from the probe, window stats (and fallback "now") from Prometheus
    gpus = []
    live = {}
    for host, h in hosts.items():
        for gl in h["G"]:
            f = [x.strip() for x in gl.split(",")]
            if len(f) >= 8:
                live[f[1]] = (host, f)
    uuids = list(dict.fromkeys(list(live) + [u for (jid, u) in prom if jid == job.jobid]))
    for u in uuids:
        pw = prom.get((job.jobid, u), {})
        host, f = live.get(u, (pw.get("node"), None))
        g = {"node": host if job.nnodes > 1 else None,
             "idx": int(f[0]) if f else None,
             "model": f[2].removeprefix("NVIDIA ") if f else pw.get("name"),
             "util": _r(fnum(f[3]), 0) if f else _r(pw.get("util_last"), 0),
             "mem_gb": _gb(fnum(f[4]) * 2**20) if f and fnum(f[4]) is not None else _gb(pw.get("mem_last")),
             "mem_total_gb": _gb(fnum(f[5]) * 2**20) if f and fnum(f[5]) is not None else _gb(pw.get("mem_total")),
             "power_w": _r(fnum(f[6]), 0) if f else None,
             "temp_c": _r(fnum(f[7]), 0) if f else None}
        if pw.get("n"):
            g["window"] = {"mean": _r(pw.get("mean"), 0), "p10": _r(pw.get("p10"), 0), "p90": _r(pw.get("p90"), 0),
                           "max": _r(pw.get("max"), 0), "mem_max_gb": _gb(pw.get("mem_max")),
                           "covered_s": int(pw["n"] * 60)}
        gpus.append(g)
    rec["gpus"] = gpus

    # job CPU / RAM (summed over nodes)
    cores = used = cache = lim = None
    for host, h in hosts.items():
        j = h["J"]
        if not j or len(j) < 8:
            continue
        t0, t1, c0, c1, cur, mx, anon = (fnum(x) for x in j[:7])
        if None not in (t0, t1, c0, c1) and t1 > t0:
            cores = (cores or 0) + (c1 - c0) / 1e6 / (t1 - t0)
        if cur is not None:
            u = anon if anon is not None else cur
            used, cache = (used or 0) + u, (cache or 0) + max(0, cur - u)
        lim = (lim or 0) + (mx or parse_mem(job.mem_req) or 0)
    if cores is not None:
        rec["cpu"] = {"used": _r(cores, 2), "alloc": job.cpus}
    if used is not None:
        rec["ram_gb"] = {"used": _gb(used), "cache": _gb(cache), "limit": _gb(lim) if lim else None}

    # processes: merged when the command line is identical (dataloader workers, vLLM workers, ...)
    groups: dict[tuple, dict] = {}
    for host, h in hosts.items():
        tck = (fnum(h["J"][7]) if h["J"] and len(h["J"]) > 7 else None) or 100.0
        dt = (fnum(h["J"][1]) - fnum(h["J"][0])) if h["J"] else 1.0
        gmem: dict[int, float] = {}
        for pl in h["P"]:
            parts = [x.strip() for x in pl.split(",")]
            if len(parts) == 3 and parts[1].isdigit():
                gmem[int(parts[1])] = gmem.get(int(parts[1]), 0) + (fnum(parts[2]) or 0)
        ports: dict[int, set] = {}
        for ll in h["L"]:
            fields = ll.split()
            if len(fields) >= 4:
                port = fields[3].rsplit(":", 1)[-1]
                for pid in re.findall(r"pid=(\d+)", ll):
                    if port.isdigit():
                        ports.setdefault(int(pid), set()).add(int(port))
        for sl in h["S"]:
            parts = sl.split(None, 4)
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            pid, ppid, rss, etime, args = int(parts[0]), int(parts[1]), fnum(parts[2]) or 0, fnum(parts[3]), parts[4]
            if BATCH_SHELL_RE.match(args):
                continue   # the batch script itself; see "batch"
            cpu = None
            if pid in h["T"] and pid in h["U"] and dt > 0:
                cpu = 100 * (h["U"][pid] - h["T"][pid]) / tck / dt
            cmd = short_args(args)
            g = groups.get((host, cmd))
            if g is None:
                g = groups[(host, cmd)] = {"pid": pid, "ppid": ppid, "n": 0, "cpu_pct": 0.0, "rss_gb": 0.0,
                                           "gpu_mem_gb": 0.0, "age_s": etime, "ports": set(), "cmd": cmd,
                                           "node": host if job.nnodes > 1 else None}
            g["n"] += 1
            g["cpu_pct"] += cpu or 0
            g["rss_gb"] = max(g["rss_gb"], rss / 2**20)   # merged workers share memory: summing overcounts
            g["gpu_mem_gb"] += gmem.get(pid, 0) / 1024
            g["ports"] |= ports.get(pid, set())
            if etime is not None and (g["age_s"] is None or etime > g["age_s"]):
                g["pid"], g["ppid"], g["age_s"] = pid, ppid, etime
    procs = sorted(groups.values(), key=lambda g: (-g["gpu_mem_gb"], -g["cpu_pct"]))
    for g in procs:
        g["cpu_pct"] = _r(g["cpu_pct"], 0)
        g["rss_gb"] = _r(g["rss_gb"], 2)
        g["gpu_mem_gb"] = _r(g["gpu_mem_gb"], 1) or None
        g["age_s"] = _r(g["age_s"], 0)
        g["ports"] = sorted(g["ports"]) or None
        g["n"] = g["n"] if g["n"] > 1 else None
    rec["procs"] = procs[: a.max_procs]
    if len(procs) > a.max_procs:
        rec["procs_omitted"] = len(procs) - a.max_procs
    rec["errors"] = errors
    return rec


def collect_json(a):
    from concurrent.futures import ThreadPoolExecutor
    jobs = list_jobs(a.user)
    if a.jobs:
        want = {x.strip() for x in a.jobs.split(",") if x.strip()}
        jobs = [j for j in jobs if j.display in want or j.jobid in want or j.display.split("_")[0] in want]
    running = [j for j in jobs if j.state == "RUNNING"]
    uid = pwd.getpwnam(a.user).pw_uid
    with ThreadPoolExecutor(max(4, len(jobs) + len(running) + 1)) as ex:
        infos = {j.jobid: ex.submit(scontrol_job, j.jobid) for j in jobs}
        probes = {j.jobid: ex.submit(probe_job, j) for j in running} if not a.no_procs else {}
        prom_f = ex.submit(prom_window, prom_url(), uid, a.window)
        try:
            prom, prom_err = prom_f.result(), None
        except Exception as e:
            prom, prom_err = {}, f"prometheus: {e}"
        recs = [job_record(j, infos[j.jobid].result(), probes[j.jobid].result() if j.jobid in probes else None, prom, a)
                for j in jobs]
    doc = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": a.user, "window_s": a.window,
           "note": "util/%: GPU busy %; window stats from 1-min Prometheus samples; ram used excludes page cache",
           "errors": [prom_err] if prom_err else None, "jobs": recs}
    return _clean(doc)
