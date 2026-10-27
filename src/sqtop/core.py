"""SLURM / node data layer: job listing, the in-job monitoring stream, Prometheus history."""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field

STEP_NAME = "sqtop"


# srun must not outlive sqtop (even on SIGKILL): the kernel sends it SIGTERM when we die, and srun
# then cancels just its own step
PDEATH = ["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []


MISSING_POLLS = 2   # a live stream is stopped only after its job is absent from this many squeue polls


# what srun / slurmstepd print when our step is ended along with its job: not worth showing
STEP_END_RE = re.compile(r"CANCELLED|Job step aborted|forcing job termination|task \d+: (Killed|Terminated)|"
                         r"Job step \d+\.\d+ terminated|Unable to create step|has been revoked|"
                         r"Socket timed out|already completing or completed|Requested nodes are busy")


STALE_S = 5.0


RESTART_BACKOFF_S = 15.0


LIVE_HIST = 6 * 3600          # 1 s samples kept per GPU


PROC_HIST = 3600              # snapshots kept per process


BACKFILL_MAX_S = 24 * 3600


RES_STEPS = [1, 2, 5, 10, 30, 60, 120, 300, 600]   # chart seconds per braille dot column


# Runs inside the job (as an overlapping step), one task per node.
# Line protocol, all prefixed with KIND|host|:
#   G|h|<nvidia-smi --query-gpu csv>                      every gpu-ms
#   U|h|<nvidia-smi pmon line>                            every 1 s (header lines start with #)
#   J|h|<t>|<mem.current>|<mem.max>|<cpu usec>|<anon>|<loadavg 1 5 15>|<node cores>|<swap>|<swap.max>
#                                                         start of a process snapshot
#   P|h|<gpu_uuid>, <pid>, <used MiB>                     GPU processes (GPU jobs)
#   S|h|<pid> <rss KiB> <cpu s> <elapsed s> <args>        ps info for those pids
#   C|h|<pid> <rss KiB> <cpu s> <elapsed s> <args>        job processes (CPU-only jobs: no GPU to ask)
#   E|h|<t>                                               end of snapshot
REMOTE = r'''
ms=$1; pint=$2; gpu=$3
trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM INT HUP
h=$(hostname -s)
cg=$(sed -n 's/^0:://p' /proc/self/cgroup); jcg=/sys/fs/cgroup${cg%%/step_*}
mystep=$(sed -n 's|.*/\(step_[^/]*\)/.*|\1|p' /proc/self/cgroup)
ncores=$(nproc --all)
if [ "$gpu" = 1 ]; then
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu,pstate \
    --format=csv,noheader,nounits -lms "$ms" | sed -u "s/^/G|$h|/" &
  nvidia-smi pmon -s u -d 1 2>/dev/null | sed -u "s/^/U|$h|/" &
fi
while :; do
  cpu=$(awk '/^usage_usec/{print $2}' "$jcg/cpu.stat" 2>/dev/null)
  anon=$(awk '/^anon /{print $2}' "$jcg/memory.stat" 2>/dev/null)
  read -r l1 l5 l15 _ < /proc/loadavg
  echo "J|$h|$EPOCHREALTIME|$(cat "$jcg/memory.current" 2>/dev/null)|$(cat "$jcg/memory.max" 2>/dev/null)|$cpu|$anon|$l1 $l5 $l15|$ncores|$(cat "$jcg/memory.swap.current" 2>/dev/null)|$(cat "$jcg/memory.swap.max" 2>/dev/null)"
  if [ "$gpu" = 1 ]; then
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$apps" ]; then
      sed "s/^/P|$h|/" <<<"$apps"
      pids=$(awk -F', ' '{print $2}' <<<"$apps" | sort -u | paste -sd,)
      ps -ww -o pid=,rss=,times=,etimes=,args= -p "$pids" | sed "s/^ */S|$h|/"
    fi
  else
    # the job's processes: every step except our monitor and SLURM's extern step, minus slurmstepd itself
    pids=$(find "$jcg" -name cgroup.procs ! -path "*/$mystep/*" ! -path "*/step_extern/*" -exec cat {} + 2>/dev/null \
           | sort -un | paste -sd,)
    [ -n "$pids" ] && ps -ww -o pid=,rss=,times=,etimes=,args= -p "$pids" | awk '$5 != "slurmstepd:"' | sed "s/^ */C|$h|/"
  fi
  echo "E|$h|$EPOCHREALTIME"
  sleep "$pint"
done
'''


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_dur(s):
    """SLURM duration ([D-]HH:MM:SS, MM:SS) -> seconds, or None."""
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", s.strip())
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def fmt_dur(sec):
    sec = int(max(0, sec))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_ago(sec):
    for unit, n in (("h", 3600), ("m", 60)):
        if sec >= n:
            v = sec / n
            return f"{v:.0f}{unit}" if v == int(v) else f"{v:.1f}{unit}"
    return f"{sec:.0f}s"


def parse_mem(s):
    """squeue %m (e.g. 20G, 80000M) -> bytes."""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", s.strip().upper())
    if not m:
        return None
    return float(m.group(1)) * 1024 ** "BKMGT".index(m.group(2) or "M")


def human_bytes(b):
    if b is None:
        return "-"
    for unit in ("B", "K", "M", "G", "T"):
        if b < 1024 or unit == "T":
            return f"{b:.0f}{unit}" if unit in ("B", "K") else f"{b:.1f}{unit}"
        b /= 1024


def prom_url():
    try:
        with open("/usr/local/jobstats/config.py") as f:
            src = f.read()
        return re.search(r'PROM_SERVER\s*=\s*"([^"]+)"', src).group(1)
    except Exception:
        return None


@dataclass
class Job:
    jobid: str      # raw id (%A); what srun --jobid and Prometheus use
    display: str    # %i, e.g. 1139833_11
    state: str
    nnodes: int
    nodes: str
    cpus: int
    elapsed: str
    limit: str
    gres: str
    reason: str
    partition: str
    mem_req: str
    name: str
    seen: float = field(default_factory=time.time)   # when elapsed was read

    @property
    def has_gpu(self):
        return "gpu" in self.gres

    def elapsed_now(self):
        e = parse_dur(self.elapsed)
        return None if e is None else e + (time.time() - self.seen)

    def left_now(self):
        e, lim = self.elapsed_now(), parse_dur(self.limit)
        return None if e is None or lim is None else lim - e


@dataclass
class GPU:
    host: str
    index: int
    uuid: str
    name: str
    util: float | None = None
    mem_used: float | None = None   # MiB
    mem_total: float | None = None
    power: float | None = None
    power_limit: float | None = None
    temp: float | None = None
    pstate: str = ""
    t: float = 0.0
    hist: deque = field(default_factory=lambda: deque(maxlen=LIVE_HIST))   # (t, util, mem MiB)


@dataclass
class Proc:
    gpu_uuid: str
    pid: int
    gpu_mem: float          # MiB
    rss_kb: float = 0
    cpu_pct: float | None = None
    etime: float | None = None
    args: str = ""


class Stream:
    """One `srun --overlap` step streaming metrics for one job, plus Prometheus backfill."""

    def __init__(self, job: Job, gpu_ms: int, proc_s: float, prom: str | None):
        self.job = job
        self.lock = threading.Lock()
        self.gpus: dict[tuple[str, int], GPU] = {}
        self.procs: dict[str, list[Proc]] = {}                       # host -> procs
        self.node: dict[str, dict] = {}                              # host -> latest cpu/mem
        self.nhist: dict[str, deque] = {}                            # host -> (t, cores, anon, limit)
        self.phist: dict[tuple[str, int], deque] = {}                # (host,pid) -> (t, cpu%, rss KiB, gmem, sm)
        self.pmon: dict[tuple[str, int, int], tuple] = {}            # (host,pid,gpu idx) -> (t, sm, gmbw)
        self.backfill: dict[str, list] = {}                          # gpu uuid -> [(t, util, mem MiB)]
        self.errors: deque = deque(maxlen=3)
        self._pmon_cols: dict[str, list[str]] = {}
        self._pending: dict[str, dict] = {}
        self._prev_cpu: dict[str, tuple[float, float]] = {}
        self._prev_pid: dict[tuple[str, int], tuple[float, float]] = {}
        cmd = [*PDEATH, "srun", "--overlap", f"--jobid={job.jobid}", f"--job-name={STEP_NAME}", "--chdir=/",
               "-N", str(job.nnodes), "--ntasks-per-node=1", "--unbuffered", "--quiet",
               "bash", "-c", REMOTE, STEP_NAME, str(gpu_ms), str(proc_s), str(int(job.has_gpu))]
        # New session: terminal Ctrl-C must not reach srun (double SIGINT aborts the step
        # noisily); we stop it ourselves with SIGTERM.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1,
                                     start_new_session=True)
        self.started = time.time()
        self.on_exit = None
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        if prom:
            threading.Thread(target=self._backfill, args=(prom,), daemon=True).start()

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.alive():
            self.proc.terminate()

    def _backfill(self, prom):
        el = self.job.elapsed_now() or 0
        if el < 120:
            return
        end = time.time()
        start = end - min(el, BACKFILL_MAX_S)
        out: dict[str, dict[float, list]] = {}
        try:
            for i, metric in enumerate(("nvidia_gpu_duty_cycle", "nvidia_gpu_memory_used_bytes")):
                q = urllib.parse.urlencode({"query": f'{metric}{{jobid="{self.job.jobid}"}}',
                                            "start": start, "end": end, "step": 60})
                with urllib.request.urlopen(f"{prom.rstrip('/')}/api/v1/query_range?{q}", timeout=15) as r:
                    data = json.load(r)
                for res in data["data"]["result"]:
                    uuid = res["metric"].get("uuid")
                    for t, v in res["values"]:
                        row = out.setdefault(uuid, {}).setdefault(float(t), [None, None])
                        row[i] = float(v) if i == 0 else float(v) / 2**20
        except Exception as e:
            with self.lock:
                self.errors.append(f"history backfill failed: {e}")
            return
        with self.lock:
            for uuid, rows in out.items():
                self.backfill[uuid] = [(t, u, m) for t, (u, m) in sorted(rows.items())]

    def _read_err(self):
        for line in self.proc.stderr:
            line = line.strip()
            if line and not STEP_END_RE.search(line):
                with self.lock:
                    self.errors.append(line)

    def _read_out(self):
        for line in self.proc.stdout:
            try:
                self._handle(line.rstrip("\n"))
            except Exception as e:  # never let one bad line kill the reader
                with self.lock:
                    self.errors.append(f"parse: {e}: {line[:80]}")
        # EOF: the step ended (usually because the job did). Wake the monitor to re-poll squeue now.
        if self.on_exit:
            self.on_exit()

    def _handle(self, line):
        kind, host, rest = line.split("|", 2)
        now = time.time()
        with self.lock:
            if kind == "G":
                f = [x.strip() for x in rest.split(",")]
                idx = int(f[0])
                g = self.gpus.get((host, idx))
                if g is None:
                    g = self.gpus[(host, idx)] = GPU(host, idx, f[1], f[2].removeprefix("NVIDIA "))
                g.util, g.mem_used, g.mem_total = fnum(f[3]), fnum(f[4]), fnum(f[5])
                g.power, g.power_limit, g.temp = fnum(f[6]), fnum(f[7]), fnum(f[8])
                g.pstate, g.t = f[9], now
                g.hist.append((now, g.util, g.mem_used))
            elif kind == "U":
                if rest.startswith("#"):
                    cols = rest.lstrip("#").split()
                    if "pid" in cols:
                        self._pmon_cols[host] = cols
                    return
                cols = self._pmon_cols.get(host)
                vals = rest.split()
                if not cols or len(vals) < len(cols) - 1:
                    return
                rec = dict(zip(cols, vals))
                if rec.get("pid", "-") != "-":
                    self.pmon[(host, int(rec["pid"]), int(rec["gpu"]))] = (now, fnum(rec.get("sm")), fnum(rec.get("mem")))
            elif kind == "J":
                t, mem_cur, mem_max, usec, anon, load, ncores, swap, swap_max = rest.split("|")
                t, usec = fnum(t), fnum(usec)
                cores = None
                prev = self._prev_cpu.get(host)
                if prev and t and usec is not None and t > prev[0]:
                    cores = (usec - prev[1]) / 1e6 / (t - prev[0])
                if t and usec is not None:
                    self._prev_cpu[host] = (t, usec)
                n = {"cores": cores, "mem": fnum(mem_cur), "mem_max": fnum(mem_max), "anon": fnum(anon),
                     "load": [fnum(x) for x in load.split()], "ncores": fnum(ncores),
                     "swap": fnum(swap), "swap_max": fnum(swap_max)}
                self.node[host] = n
                self.nhist.setdefault(host, deque(maxlen=LIVE_HIST)).append(
                    (now, cores, n["anon"] if n["anon"] is not None else n["mem"], n["mem_max"]))
                self._pending[host] = {"t": t, "procs": {}}
            elif kind == "P":
                snap = self._pending.get(host)
                if snap is not None:
                    uuid, pid, mem = [x.strip() for x in rest.split(",")]
                    snap["procs"].setdefault(int(pid), []).append(Proc(uuid, int(pid), fnum(mem) or 0))
            elif kind in ("S", "C"):
                snap = self._pending.get(host)
                if snap is not None:
                    pid, rss, cpu_s, etimes, *args = rest.split(None, 4)
                    if kind == "C":   # CPU-only job: no GPU listing, so the ps line defines the process
                        snap["procs"].setdefault(int(pid), []).append(Proc("", int(pid), 0))
                    pid, cpu_s = int(pid), fnum(cpu_s)
                    pct = None
                    prev = self._prev_pid.get((host, pid))
                    if prev and snap["t"] and cpu_s is not None and snap["t"] > prev[0]:
                        pct = 100 * (cpu_s - prev[1]) / (snap["t"] - prev[0])
                    if snap["t"] and cpu_s is not None:
                        self._prev_pid[(host, pid)] = (snap["t"], cpu_s)
                    for p in snap["procs"].get(pid, []):
                        p.rss_kb, p.cpu_pct, p.etime = fnum(rss) or 0, pct, fnum(etimes)
                        p.args = args[0] if args else ""
            elif kind == "E":
                snap = self._pending.pop(host, None)
                if snap is not None:
                    procs = [p for ps in snap["procs"].values() for p in ps]
                    self.procs[host] = procs
                    idx_of = {g.uuid: g.index for (h, _), g in self.gpus.items() if h == host}
                    for pid, ps in snap["procs"].items():
                        p0 = ps[0]
                        sm = self.sm_of(host, pid, idx_of.get(p0.gpu_uuid))[0]
                        self.phist.setdefault((host, pid), deque(maxlen=PROC_HIST)).append(
                            (now, p0.cpu_pct, p0.rss_kb, sum(p.gpu_mem for p in ps), sm))

    def sm_of(self, host, pid, gidx):
        """(%SM, %GMBW) from pmon, None if missing/stale. Caller holds the lock."""
        r = self.pmon.get((host, pid, gidx))
        if r is None or time.time() - r[0] > STALE_S:
            return None, None
        return r[1], r[2]


def list_jobs(user):
    fmt = "%A|%i|%T|%D|%N|%C|%M|%l|%b|%r|%P|%m|%j"
    out = subprocess.run(["squeue", "-u", user, "-h", "-t", "RUNNING,PENDING", "-o", fmt],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    jobs = []
    for line in out.splitlines():
        f = line.split("|", 12)
        if len(f) < 13:
            continue
        jobs.append(Job(f[0], f[1], f[2], int(f[3] or 1), f[4], int(f[5] or 0), f[6], f[7],
                        f[8].replace("gres/", ""), f[9], f[10], f[11], f[12]))
    jobs.sort(key=lambda j: (j.state != "RUNNING", j.jobid))
    return jobs


class Monitor:
    def __init__(self, a):
        self.a = a
        self.prom = None if a.no_history else prom_url()
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.streams: dict[str, Stream] = {}
        self.dead: dict[str, float] = {}
        self.dead_errors: dict[str, list[str]] = {}
        self.missing: dict[str, int] = {}
        self.last_squeue = 0.0
        self.squeue_err = ""
        self.stop_evt = threading.Event()
        self.kick = threading.Event()

    def poll_squeue(self):
        try:
            jobs = list_jobs(self.a.user)
        except Exception as e:
            self.squeue_err = str(e)[:120]
            return
        with self.lock:
            self.jobs, self.last_squeue, self.squeue_err = jobs, time.time(), ""
        self.sync()

    def sync(self):
        with self.lock:
            running = {j.jobid: j for j in self.jobs if j.state == "RUNNING"}
            for jid in list(self.streams):
                s = self.streams[jid]
                if jid not in running and s.alive():
                    # one squeue poll can come back incomplete while the controller is busy; a job that has
                    # really ended takes its step (and so this stream) down with it anyway
                    self.missing[jid] = self.missing.get(jid, 0) + 1
                    if self.missing[jid] >= MISSING_POLLS:
                        s.stop()
                        del self.streams[jid]
                    continue
                self.missing.pop(jid, None)
                if jid not in running:
                    del self.streams[jid]
                elif not s.alive():
                    self.dead[jid] = time.time()
                    self.dead_errors[jid] = list(s.errors)
                    del self.streams[jid]
            for jid, job in running.items():
                if jid in self.streams:
                    self.streams[jid].job = job
                elif time.time() - self.dead.get(jid, 0) > RESTART_BACKOFF_S:
                    st = self.streams[jid] = Stream(job, self.a.gpu_ms, self.a.proc_s, self.prom)
                    st.on_exit = self.kick.set

    def squeue_loop(self):
        while not self.stop_evt.is_set():
            self.poll_squeue()
            deadline = time.time() + self.a.squeue_s
            while time.time() < deadline and not self.stop_evt.is_set():
                if self.kick.wait(1):
                    self.kick.clear()
                    break
                if any(not s.alive() for s in list(self.streams.values())):
                    break

    def shutdown(self):
        self.stop_evt.set()
        with self.lock:
            streams = list(self.streams.values())
        for s in streams:
            s.stop()
        deadline = time.time() + 10
        for s in streams:
            try:
                s.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                s.proc.kill()

    def snapshot(self):
        with self.lock:
            return list(self.jobs), dict(self.streams), dict(self.dead_errors)


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def short_args(args):
    parts = args.split()
    if parts:
        parts[0] = os.path.basename(parts[0])
    return " ".join(parts)


def job_mem_limit(job: Job, node: dict):
    return node.get("mem_max") or parse_mem(job.mem_req)


def gres_short(gres):
    g = re.sub(r"^gpu:", "", gres or "")
    return g or "-"


def read_log_lines(path, max_lines, tail_bytes=256 * 1024):
    """Last max_lines lines of a log, with \r-rewritten progress bars reduced to their final state."""
    size = os.path.getsize(path)
    tail = tail_bytes
    with open(path, "rb") as f:
        # progress bars can fill MBs without a newline: widen the tail until there are real lines
        while True:
            f.seek(max(0, size - tail))
            raw = f.read().decode("utf-8", "replace")
            if raw.count("\n") >= min(200, max_lines + 1) or tail >= size or tail >= 16 * 2**20:
                break
            tail *= 4
    lines = raw.split("\n")
    if size > tail and len(lines) > 1:
        lines = lines[1:]   # first line is probably partial (unless the tail is one long \r-rewritten line)
    kept = []
    for ln in lines:
        last = ln.rstrip("\r").split("\r")[-1]
        if "\r" in ln and not ANSI_RE.sub("", last).strip():
            continue
        kept.append(last)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept[-max_lines:] if max_lines else []


def scontrol_job(jobid):
    try:
        out = subprocess.run(["scontrol", "show", "job", "-o", jobid], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return {}
    return dict(re.findall(r"(\w+(?:/\w+)?)=(\S*)", out))
