"""Cluster-wide GPU availability: free GPUs by type and by node, and the queue for each type.

Each poll makes two controller calls (one `sinfo -N`, one all-users `squeue`). They run only while a
cluster screen is open (or once for `--gpus`), at most every --cluster-s seconds.
"""
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from rich.text import Text

from .core import fmt_dur, parse_dur
from .render import B

# Display order of GPU types (best first) and their VRAM. Types whose nodes carry different `NNgb`
# features (e.g. a100 nodes tagged 40gb / 80gb) are split into one group per size: a100-80g, a100-40g.
ORDER = ["h200", "h100", "a100", "l40s", "v100", "gh200"]
VRAM = {"h200": "141G", "h100": "80G", "l40s": "48G", "v100": "32G", "gh200": "96G"}
BAD_STATES = ("down", "drain", "fail", "maint", "unk", "boot", "resv", "inval", "plnd", "npc")
GPU_RE = re.compile(r"gpu:(?:([A-Za-z0-9_.-]+):)?(\d+)")
MEM_FEAT_RE = re.compile(r"(\d+)gb")
STRANDED_MEM_MB = 4096   # free GPUs on a node with no free CPUs or less memory than this can't be used

SINFO_FIELDS = "NodeList:|,Gres:|,GresUsed:|,StateCompact:|,Features:|,CPUsState:|,Memory:|,AllocMem:|,Partition:|"
SQUEUE_FIELDS = ("JobArrayID:|,UserName:|,StateCompact:|,tres-alloc:|,ReasonList:|,Feature:|,NodeList:|,NumNodes:|,"
                 "TimeLeft:|,StartTime:|,PriorityLong:|,Partition:|,SubmitTime:|,Name:|")

# pending reasons: contending for GPUs now / held back by a limit / not eligible yet
QUEUED_REASONS = ("Resources", "Priority")
HELD_PREFIXES = ("Dependency", "JobHeld", "BeginTime")
CLASSES = (("queued", "waiting on Priority / Resources"),
           ("blocked", "blocked by QOS / association / array limits"),
           ("held", "dependencies, holds, begin time"))


def gpu_counts(gres):
    """'gpu:h200:7(IDX:0-4,6-7)' -> {'h200': 7}"""
    out = defaultdict(int)
    for typ, n in GPU_RE.findall(gres):
        out[typ or "gpu"] += int(n)
    return out


def job_gpus(tres):
    """tres-alloc ('cpu=8,...,gres/gpu=2,gres/gpu:a100=2') -> ('a100', 2); ('', n) when untyped."""
    typed = re.findall(r"gres/gpu:([^=,]+)=(\d+)", tres)
    if typed:
        return typed[0][0], sum(int(n) for _, n in typed)
    m = re.search(r"gres/gpu=(\d+)", tres)
    return "", int(m.group(1)) if m else 0


def expand_hosts(s):
    """'rack5n[07-08,10],rack3n05' -> ['rack5n07', 'rack5n08', 'rack5n10', 'rack3n05']"""
    out = []
    for m in re.finditer(r"([^,\[\]]+)(?:\[([^\]]*)\])?", s):
        pre, rng = m.groups()
        if not rng:
            out.append(pre)
            continue
        for part in rng.split(","):
            a, _, b = part.partition("-")
            out += [f"{pre}{i:0{len(a)}d}" for i in range(int(a), int(b or a) + 1)]
    return out


def array_tasks(jid):
    """Tasks in a pending array record: '123_[4-9%2]' -> 6, '123_[1,3,5-7]' -> 5, plain ids -> 1."""
    m = re.search(r"_\[([^\]]+)\]", jid)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split("%")[0].split(","):
        a, _, rest = part.partition("-")
        b, _, step = rest.partition(":")
        n += (int(b) - int(a)) // int(step or 1) + 1 if b else 1
    return n


def parse_ts(s):
    try:
        return time.mktime(time.strptime(s.strip(), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def fmt_mem(mb):
    return f"{mb / 1024:.0f}G" if mb < 1024 * 1000 else f"{mb / 1024 / 1024:.1f}T"


def fmt_rel(sec):
    sec = max(0, int(sec))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    return f"{d}d{h:02d}h" if d else f"{h}h{r // 60:02d}m" if h else f"{r // 60}m"


@dataclass
class CJob:
    id: str
    user: str
    pending: bool
    base: str              # GPU type from tres-alloc ('' = any type)
    gpus: int              # per job (array record: per task)
    tasks: int             # array tasks in a pending record
    reason: str
    features: str
    hosts: list
    nnodes: int
    left_s: float | None
    start_t: float | None  # running: start; pending: Slurm's estimate (often absent)
    prio: int
    partition: str
    submit_t: float | None
    name: str
    fits: str = ""         # set when the job fits several groups, e.g. 'any a100' (no size constraint)

    @property
    def gpn(self):
        return self.gpus // max(1, self.nnodes)

    @property
    def klass(self):
        if self.reason in QUEUED_REASONS:
            return "queued"
        return "held" if self.reason.startswith(HELD_PREFIXES) else "blocked"


@dataclass
class Node:
    name: str
    base: str
    mem: int | None        # NNgb feature
    total: int
    used: int
    state: str
    ok: bool
    cpu_free: int
    cpu_total: int
    mem_free: int          # MB
    mem_total: int
    partitions: list
    group: str = ""
    jobs: list = field(default_factory=list)

    @property
    def free(self):
        return max(0, self.total - self.used)

    @property
    def stranded(self):
        return self.ok and self.free > 0 and (self.cpu_free == 0 or self.mem_free < STRANDED_MEM_MB)


@dataclass
class Group:
    base: str
    mem: int | None = None
    nodes: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    total: int = 0
    free: int = 0
    off: int = 0
    stranded: int = 0
    cpu_free: int = 0      # summed over the nodes counted in `free`
    mem_free: int = 0
    off_states: dict = field(default_factory=lambda: defaultdict(int))
    pend_gpus: dict = field(default_factory=lambda: defaultdict(int))   # class -> GPUs
    pend_jobs: dict = field(default_factory=lambda: defaultdict(int))   # class -> job (task) count
    ending_1h: int = 0
    ending_4h: int = 0
    next_start: float | None = None
    mine: int = 0          # GPUs the user has pending here

    @property
    def name(self):
        return f"{self.base}-{self.mem}g" if self.mem else self.base

    @property
    def vram(self):
        return f"{self.mem}G" if self.mem else VRAM.get(self.base, "")

    def sort_key(self):
        i = ORDER.index(self.base) if self.base in ORDER else len(ORDER)
        return (i, self.base, -(self.mem or 0))


@dataclass
class Snapshot:
    t: float
    nodes: dict
    groups: list
    user: str

    def group(self, name):
        return next((g for g in self.groups if g.name == name), None)

    def pending(self):
        """Each pending job once (a job that fits several groups is listed in each)."""
        return list({id(j): j for g in self.groups for j in g.pending}.values())


def parse_nodes(out):
    nodes = {}
    for line in out.splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) < 9:
            continue
        name, gres, used, state, feats, cpus, mem, amem, part = f[:9]
        part = part.rstrip("*")
        if name in nodes:   # sinfo -N lists a node once per partition
            nodes[name].partitions.append(part)
            continue
        total = {t: n for t, n in gpu_counts(gres).items() if n}
        if not total:
            continue
        base = max(total, key=total.get)   # nodes mixing GPU types are rare: count the main one
        mems = [int(m.group(1)) for x in feats.split(",") if (m := MEM_FEAT_RE.fullmatch(x.strip().lower()))]
        try:
            _, idle, _, ctot = (int(x) for x in cpus.split("/"))
        except ValueError:
            idle = ctot = 0
        mem_tot = int(mem) if mem.isdigit() else 0
        amem = int(amem) if amem.isdigit() else 0
        nodes[name] = Node(name, base, mems[0] if mems else None, total[base], gpu_counts(used).get(base, 0), state,
                           not any(b in state for b in BAD_STATES) and not state.endswith("*"),
                           idle, ctot, mem_tot - amem, mem_tot, [part])
    return nodes


def parse_jobs(out):
    jobs = []
    for line in out.splitlines():
        f = line.split("|", 13)
        if len(f) < 14:
            continue
        jid, user, st, tres, reason, feat, nodelist, nn, left, start, prio, part, submit, name = (x.strip() for x in f)
        base, n = job_gpus(tres)
        if not n:
            continue
        pending = st == "PD"
        feat = "" if feat == "(null)" else feat
        jobs.append(CJob(jid, user, pending, base, n, array_tasks(jid) if pending else 1,
                         reason.strip("()") if pending else "", feat,
                         [] if pending else expand_hosts(nodelist), int(nn) if nn.isdigit() else 1,
                         None if pending else parse_dur(left), parse_ts(start),
                         int(prio) if prio.isdigit() else 0, part, parse_ts(submit), name.removesuffix("|")))
    return jobs


def build(nodes, jobs, user, now=None):
    now = now or time.time()
    mems = defaultdict(set)
    for n in nodes.values():
        mems[n.base].add(n.mem)
    split = {t for t, s in mems.items() if len(s) > 1}
    groups = {}

    def grp(base, mem=None):
        return groups.setdefault((base, mem), Group(base, mem))

    for n in nodes.values():
        g = grp(n.base, n.mem if n.base in split else None)
        n.group = g.name
        g.nodes.append(n)
    for j in jobs:
        if not j.pending:
            for h in j.hosts:
                if h in nodes:
                    nodes[h].jobs.append(j)
            continue
        if not j.base:   # untyped request (gpu:N): can run on any type
            targets = [k for k in groups if k[0]]
            j.fits = "any type"
        elif j.base in split:
            want = {int(x) for x in MEM_FEAT_RE.findall(j.features.lower())} & mems[j.base]
            targets = [(j.base, m) for m in sorted(want or mems[j.base], key=lambda m: -(m or 0))]
            if len(targets) > 1:
                j.fits = f"any {j.base}"
        else:
            targets = [(j.base, None)]
        for k in targets:
            grp(*k).pending.append(j)

    for g in groups.values():
        for n in g.nodes:
            g.total += n.total
            if not n.ok:
                g.off += n.total
                g.off_states[n.state] += n.total
                continue
            if n.stranded:
                g.stranded += n.free
            elif n.free:
                g.free += n.free
                g.cpu_free += n.cpu_free
                g.mem_free += n.mem_free
            for j in n.jobs:
                if j.left_s is not None and j.left_s <= 4 * 3600:
                    g.ending_4h += j.gpn
                    g.ending_1h += j.gpn if j.left_s <= 3600 else 0
        for j in g.pending:
            g.pend_gpus[j.klass] += j.gpus * j.tasks
            g.pend_jobs[j.klass] += j.tasks
            if j.user == user:
                g.mine += j.gpus * j.tasks
            if j.klass == "queued" and j.start_t and j.start_t > now:
                g.next_start = min(g.next_start or j.start_t, j.start_t)
    return Snapshot(now, nodes, sorted(groups.values(), key=Group.sort_key), user)


def collect(user):
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout

    nodes = parse_nodes(run(["sinfo", "-N", "-h", "-O", SINFO_FIELDS]))
    jobs = parse_jobs(run(["squeue", "-h", "-t", "R,PD", "-O", SQUEUE_FIELDS]))
    return build(nodes, jobs, user)


class ClusterMonitor:
    """Polls sinfo + squeue every `interval` s, but only while some screen has called activate()."""

    def __init__(self, interval, user):
        self.interval, self.user = interval, user
        self.snap: Snapshot | None = None
        self.err = ""
        self.last = 0.0
        self.active = 0
        self.kick = threading.Event()
        self.thread = None

    def activate(self):
        self.active += 1
        if self.thread is None:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def deactivate(self):
        self.active = max(0, self.active - 1)

    def poll(self):
        try:
            self.snap, self.err = collect(self.user), ""
        except Exception as e:
            self.err = str(e)[:120]
        self.last = time.time()

    def loop(self):
        while True:
            kicked = self.kick.wait(1)
            self.kick.clear()
            if self.active and (kicked or time.time() - self.last >= self.interval):
                self.poll()


# ---- rendering ----------------------------------------------------------------------------------
BAR = 16
LEGEND = ("QUEUED / BLOCKED / HELD: pending GPUs waiting on priority or resources / held by QOS or other limits / "
          "dependencies or holds.  END≤1h ≤4h: allocated GPUs whose jobs reach their time limit within 1h / 4h "
          "(they may end sooner).  CPU/G MEM/G: free CPUs / memory per free GPU.  A pending job that fits several "
          "rows (e.g. a100 with no 40gb/80gb constraint) counts in each.")


def group_bar(g):
    if not g.total:
        return Text(" " * BAR)
    nf, no = round(BAR * g.free / g.total), round(BAR * g.off / g.total)
    t = Text("█" * nf, style="green")
    t.append("░" * max(BAR - nf - no, 0), style=B)
    t.append("×" * no, style="red")
    return t


def summary_head():
    t = Text(f"{'GPU':<11}{'VRAM':>5}  {'FREE':^9}  {'':<{BAR}}  {'CPU/G':>5}  {'MEM/G':>5}  ", style="bold")
    t.append("│", style=B)
    t.append(f"{'QUEUED':>7}{'BLOCKED':>8}{'HELD':>6}{'END≤1h':>8}{'≤4h':>5}  {'NEXT START':<10}  NOTES", style="bold")
    return t


def summary_row(g, now):
    t = Text(f"{g.name:<11}", style="bold" if g.free else "")
    t.append(f"{g.vram:>5}  ")
    t.append(f"{g.free:>3} / {g.total:<3}", style="bold green" if g.free else B)
    t.append("  ")
    t.append_text(group_bar(g))
    t.append("  ")
    if g.free:
        t.append(f"{g.cpu_free / g.free:>5.1f}  {fmt_mem(g.mem_free / g.free):>5}  ")
    else:
        t.append(f"{'-':>5}  {'-':>5}  ", style=B)
    t.append("│", style=B)
    for k, w, st in (("queued", 7, "yellow"), ("blocked", 8, "magenta"), ("held", 6, "")):
        v = g.pend_gpus.get(k, 0)
        t.append(f"{v:>{w}}", style=st if v else B)
    for v, w in ((g.ending_1h, 8), (g.ending_4h, 5)):
        t.append(f"{v:>{w}}", style="cyan" if v else B)
    ns = f"in {fmt_rel(g.next_start - now)}" if g.next_start else "-"
    t.append(f"  {ns:<10}  ", style="" if g.next_start else B)
    notes = []
    if g.nodes:
        best = max((n for n in g.nodes if n.ok and not n.stranded), key=lambda n: n.free, default=None)
        if best and best.free:
            notes.append((f"best: {best.name} ({best.free})", ""))
    if g.stranded:
        notes.append((f"{g.stranded} stranded (no cpu/mem)", "yellow"))
    if g.off:
        states = sorted(g.off_states.items(), key=lambda x: (-x[1], x[0]))
        notes.append(("off: " + ", ".join(f"{n} {s}" for s, n in states), "red"))
    if g.mine:
        notes.append((f"yours: {g.mine} pending", "bold cyan"))
    for i, (s, st) in enumerate(notes):
        t.append(", " if i else "", style=B)
        t.append(s, style=st)
    return t


NODE_HEAD = f"{'NODE':<12}{'GPU':<11}{'STATE':<9}{'GPUS':<12}{'FREE':>5}{'CPU FREE':>10}{'MEM FREE':>13}  RUNNING (user×GPUs time left, soonest first)"


def node_row(n, user):
    t = Text(f"{n.name:<12}{n.group:<11}")
    t.append(f"{n.state:<9}", style="" if n.ok else "red")
    if not n.ok:
        cells = Text("×" * n.total, style="red")
    else:
        cells = Text("■" * min(n.used, n.total), style="yellow")
        cells.append("□" * n.free, style="yellow" if n.stranded else "bold green")
    cells.truncate(11, pad=True)
    t.append_text(cells)
    t.append(" ")
    t.append(f"{n.free if n.ok else '-':>2}/{n.total:<2}", style="bold green" if n.free and n.ok and not n.stranded else B)
    t.append(f"{n.cpu_free:>5}/{n.cpu_total:<4}{fmt_mem(n.mem_free):>6}/{fmt_mem(n.mem_total):<6}  ",
             style="yellow" if n.stranded else "")
    for i, j in enumerate(sorted(n.jobs, key=lambda j: (j.left_s is None, j.left_s or 0))):
        t.append(", " if i else "", style=B)
        t.append(f"{j.user}×{j.gpn} {fmt_dur(j.left_s) if j.left_s is not None else '?'}",
                 style="bold cyan" if j.user == user else "")
    return t


QUEUE_HEAD = (f"{'JOBID':<20}{'USER':<11}{'GPUS':>7}  {'REASON':<19}{'PRIO':>9}{'WAITED':>8}{'EST START':>11}  "
              f"{'CONSTRAINT':<12}{'PART':<7}NAME")


def queue_row(j, user, now):
    gp = f"{j.gpus}" + (f"×{j.tasks}" if j.tasks > 1 else "")
    start = f"in {fmt_rel(j.start_t - now)}" if j.start_t and j.start_t > now else "-"
    waited = fmt_rel(now - j.submit_t) if j.submit_t else "-"
    return Text(f"{j.id:<20}{j.user:<11}{gp:>7}  {j.reason[:18]:<19}{j.prio:>9}{waited:>8}{start:>11}  "
                f"{(j.features or j.fits)[:11]:<12}{j.partition[:6]:<7}{j.name}", style="bold cyan" if j.user == user else "")


def pend_order(j):
    return ([c for c, _ in CLASSES].index(j.klass), -j.prio, j.submit_t or 0)


def render_summary(snap, node_mode):
    """-> (header lines, selectable (key, line) rows, footer lines)."""
    if node_mode:
        nodes = sorted(snap.nodes.values(), key=lambda n: (next(
            (i for i, g in enumerate(snap.groups) if g.name == n.group), 0), n.name))
        return [Text(NODE_HEAD, style="bold")], [(n.name, node_row(n, snap.user)) for n in nodes], []
    rows = [(g.name, summary_row(g, snap.t)) for g in snap.groups]
    return [summary_head()], rows, [Text(""), Text(LEGEND, style=B)]


def render_detail(snap, name):
    g = snap.group(name)
    if g is None:
        return [Text(f"no GPU type {name!r} in the last poll (types: {', '.join(x.name for x in snap.groups)})",
                     style="yellow")]
    out = [summary_head(), summary_row(g, snap.t), Text("")]
    nodes = sorted(g.nodes, key=lambda n: (not n.ok, n.stranded, -n.free, n.name))
    out.append(Text(f"Nodes ({len(nodes)})", style="bold underline"))
    out.append(Text(NODE_HEAD, style="bold"))
    out += [node_row(n, snap.user) for n in nodes]
    pend = sorted(g.pending, key=pend_order)
    out.append(Text(""))
    out.append(Text(f"Pending jobs ({sum(j.tasks for j in pend)})", style="bold underline"))
    shared = [j for j in pend if j.fits]
    if shared:
        out.append(Text(f"{sum(j.tasks for j in shared)} of these ({sum(j.gpus * j.tasks for j in shared)} GPUs) can also run "
                        f"on other rows: {', '.join(sorted({j.fits for j in shared}))}", style="italic"))
    if not pend:
        out.append(Text("none", style=B))
    for k, label in CLASSES:
        js = [j for j in pend if j.klass == k]
        if not js:
            continue
        out.append(Text(""))
        h = Text(f"{k.upper()}", style="bold " + {"queued": "yellow", "blocked": "magenta", "held": ""}[k])
        h.append(f"  {g.pend_gpus[k]} GPUs, {g.pend_jobs[k]} jobs: {label}", style=B)
        out += [h, Text(QUEUE_HEAD, style="bold")]
        out += [queue_row(j, snap.user, snap.t) for j in js]
    return out


def snapshot_json(snap, name=None):
    def node(n):
        return {"name": n.name, "gpu": n.group, "state": n.state, "ok": n.ok, "gpus": n.total, "free": n.free,
                "stranded": n.stranded or None, "cpus_free": n.cpu_free, "mem_free_gb": round(n.mem_free / 1024),
                "running": [{"id": j.id, "user": j.user, "gpus": j.gpn, "left_s": j.left_s} for j in n.jobs]}

    def job(j):
        return {"id": j.id, "user": j.user, "gpus": j.gpus, "tasks": j.tasks if j.tasks > 1 else None,
                "reason": j.reason, "class": j.klass, "prio": j.prio,
                "waited_s": round(snap.t - j.submit_t) if j.submit_t else None,
                "est_start_s": round(j.start_t - snap.t) if j.start_t and j.start_t > snap.t else None,
                "constraint": j.features, "fits": j.fits, "partition": j.partition, "name": j.name}

    groups = []
    for g in snap.groups:
        if name and g.name != name:
            continue
        d = {"name": g.name, "vram": g.vram, "total": g.total, "free": g.free, "off": g.off,
             "stranded": g.stranded, "cpus_per_free_gpu": round(g.cpu_free / g.free, 1) if g.free else None,
             "mem_gb_per_free_gpu": round(g.mem_free / 1024 / g.free) if g.free else None,
             "pending_gpus": dict(g.pend_gpus), "pending_jobs": dict(g.pend_jobs),
             "ending_1h": g.ending_1h, "ending_4h": g.ending_4h,
             "next_start_s": round(g.next_start - snap.t) if g.next_start else None, "mine_pending": g.mine}
        if name:
            d["nodes"] = [node(n) for n in g.nodes]
            d["pending"] = [job(j) for j in sorted(g.pending, key=pend_order)]
        groups.append(d)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(snap.t)), "user": snap.user, "groups": groups}
