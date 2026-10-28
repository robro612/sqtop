"""Rendering: nvitop-style device table, host panel, braille charts, jobs & processes table."""
import os
import time
from dataclasses import dataclass

from rich.text import Text

from .core import (
    STALE_S,
    Job,
    Monitor,
    Stream,
    fmt_ago,
    fmt_dur,
    gres_short,
    human_bytes,
    job_mem_limit,
    parse_dur,
    parse_mem,
    short_args,
)

# dot bits filled from the baseline outward: (left column, right column)
_UP = ((0x40, 0x04, 0x02, 0x01), (0x80, 0x20, 0x10, 0x08))


_DOWN = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


def bucketize(series, t_end, res, n, carry):
    """Average (t, v) samples into n buckets of res seconds ending at t_end.
    series: list of ascending sample sequences (scanned newest-first). Empty buckets
    repeat the previous value for up to `carry` seconds, else None."""
    t0 = t_end - n * res
    sums, cnt = [0.0] * n, [0] * n
    prev_t, prev_v = None, None
    for s in series:
        for t, v in reversed(s):
            if t < t0:
                if v is not None and (prev_t is None or t > prev_t):
                    prev_t, prev_v = t, v
                break
            if v is None or t >= t_end:
                continue
            i = int((t - t0) / res)
            sums[i] += v
            cnt[i] += 1
    out = []
    for i in range(n):
        if cnt[i]:
            v = sums[i] / cnt[i]
            prev_t, prev_v = t0 + (i + 1) * res, v
            out.append(v)
        elif prev_t is not None and t0 + i * res - prev_t <= carry:
            out.append(prev_v)
        else:
            out.append(None)
    return out


def mean_buckets(arrs):
    if not arrs:
        return []
    out = []
    for vals in zip(*arrs):
        vs = [v for v in vals if v is not None]
        out.append(sum(vs) / len(vs) if vs else None)
    return out


def braille(vals, height, upward, style_fn):
    """vals: 2*ncols fractions (0..1 or None) -> `height` Text rows of braille area chart."""
    ncols = len(vals) // 2
    bits_tab = _UP if upward else _DOWN
    rows = []
    for r in range(height):
        level = (height - 1 - r) * 4 if upward else r * 4  # dots between baseline and this cell
        line = Text()
        for c in range(ncols):
            bits, vmax = 0, None
            for side in (0, 1):
                v = vals[2 * c + side]
                if v is None:
                    continue
                v = max(0.0, min(1.0, v))
                vmax = v if vmax is None else max(vmax, v)
                dots = round(v * height * 4)
                if v > 0 and dots == 0:
                    dots = 1
                for k in range(max(0, min(4, dots - level))):
                    bits |= bits_tab[side][k]
            line.append(chr(0x2800 + bits) if bits else " ", style=style_fn(vmax) if bits else None)
        rows.append(line)
    return rows


def time_axis(ncols, res, fill="─"):
    span = 2 * ncols * res
    tick = next((iv for iv in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)
                 if iv / (2 * res) >= 14), 86400)
    chars = [fill] * ncols
    k = 1
    while k * tick <= span:
        col = ncols - int(k * tick / (2 * res))
        label = f"-{fmt_ago(k * tick)}"
        start = col - len(label)
        if start >= 0:
            chars[start:col] = label
        k += 1
    return Text("".join(chars), style="bright_black")


UTIL_THRESH = (30, 70)


RAM_STYLE = "blue"   # job RAM everywhere (CPU is cyan, GPU memory magenta)   # util below 30% is a problem for training jobs, so low = red (unlike nvitop)


def util_style(u):
    if u is None:
        return "bright_black"
    return "red" if u < UTIL_THRESH[0] else "yellow" if u < UTIL_THRESH[1] else "green"


def mem_style(frac):
    return "magenta"


EIGHTHS = " ▏▎▍▌▋▊▉"


def nv_bar(label, frac, width, style, value, right=""):
    """nvitop-style 'LABEL: █████▉ 4.3%      right' in exactly `width` columns."""
    t = Text(f"{label}: ", style="bold")
    room = max(1, width - len(label) - 2 - len(value) - 1 - (len(right) + 1 if right else 0))
    f = 0.0 if frac is None else max(0.0, min(1.0, frac))
    cells = f * room
    full, rem = int(cells), int((cells - int(cells)) * 8)
    t.append("█" * full + (EIGHTHS[rem] if rem and full < room else ""), style=style)
    t.append(f" {value}", style=style if frac is not None else B)
    if right:
        t.append(" " * max(1, width - len(t.plain) - len(right)))
        t.append(right, style="bold")
    return fit(t, width)


def fit(t, w, style=None):
    t = Text(t, style=style or "") if isinstance(t, str) else t.copy()
    t.truncate(max(0, w), overflow="ellipsis", pad=True)
    return t


TW = 79   # inner width of the device table, as in nvidia-smi / nvitop
SUB = (30, 22, 25)   # its sub-columns: identity | state | usage (TW = sum + 2 dividers)


B = "bright_black"


class Frame:
    """Device table (three nvidia-smi-style sub-columns) plus separate GPU and job-CPU bar columns."""

    def __init__(self, width):
        rest = width - TW - 3
        if rest >= 64:                  # room for both bar columns
            self.gw = (rest - 1) * 11 // 20
            self.cw = rest - 1 - self.gw
        else:                           # GPU column only (CPU falls back to text in the table)
            self.gw, self.cw = (rest if rest >= 30 else 0), 0

    def border(self, l, fill, mid, r):
        s = l + mid.join(fill * w for w in SUB)
        for w in (self.gw, self.cw):
            if w:
                s += mid + fill * w
        return Text(s + r, style=B)

    def row(self, left, gpu=None, cpu=None):
        """left: a (identity, state, usage) tuple of cells, or one Text spanning the whole table."""
        t = Text("│", style=B)
        if isinstance(left, tuple):
            for i, (cell, w) in enumerate(zip(left, SUB)):
                if i:
                    t.append("│", style=B)
                t.append_text(fit(cell if cell is not None else Text(""), w))
        else:
            t.append_text(fit(left, TW))
        for w, cell in ((self.gw, gpu), (self.cw, cpu)):
            if w:
                t.append("│", style=B)
                t.append_text(fit(cell if cell is not None else Text(""), w))
        t.append("│", style=B)
        return t


def two_bars(rw, l1, f1, v1, s1, l2, f2, v2, s2):
    """Two nvitop-style bars side by side in rw columns."""
    half = (rw - 3) // 2
    out = Text(" ")
    out.append_text(nv_bar(l1, f1, half, s1, v1))
    out.append(" ")
    out.append_text(nv_bar(l2, f2, rw - 3 - half, s2, v2))
    return out


def job_stats(job, s):
    """(cores used, cpu frac, ram used, ram limit, cache) summed over the job's nodes."""
    if s is None:
        return None
    with s.lock:
        nodes = list(s.node.values())
    if not nodes:
        return None
    cores = sum(n["cores"] or 0 for n in nodes) if any(n["cores"] is not None for n in nodes) else None
    used = sum((n["anon"] if n["anon"] is not None else n["mem"]) or 0 for n in nodes)
    lim = sum(job_mem_limit(job, n) or 0 for n in nodes) or None
    cache = sum((n["mem"] or 0) - (n["anon"] or 0) for n in nodes if n["anon"] is not None)
    return cores, (cores / job.cpus if cores is not None and job.cpus else None), used, lim, cache


def cpu_ram_bars(rw, job, st):
    cores, cfrac, used, mlim, _ = st
    return two_bars(rw, "CPU", cfrac, f"{cores:.1f}/{job.cpus}" if cores is not None else f"-/{job.cpus}", "cyan",
                    "RAM", used / mlim if mlim else None, f"{human_bytes(used)}/{human_bytes(mlim)}", RAM_STYLE)


def cpu_ram_stack(cw, job, st):
    """Job CPU over RAM, one full-width bar each (full mode)."""
    cores, cfrac, used, mlim, _ = st
    cpu = Text(" ") + nv_bar("CPU", cfrac, cw - 2, "cyan",
                             f"{cores:.1f}/{job.cpus} cores" if cores is not None else f"-/{job.cpus} cores")
    ram = Text(" ") + nv_bar("RAM", used / mlim if mlim else None, cw - 2, RAM_STYLE,
                             f"{human_bytes(used)}/{human_bytes(mlim)}")
    return cpu, ram


def left_style(left, limit):
    if left is None:
        return B
    if left < 300 or (limit and left < 0.05 * limit):
        return "bold red"
    if left < 1800 or (limit and left < 0.15 * limit):
        return "yellow"
    return ""


def _num_or(v, fmt, na="N/A"):
    return na if v is None else format(v, fmt)


def gpu_fields(g):
    """Fixed-width strings so the '/' in power and memory line up across rows."""
    pw = f"{_num_or(g.power, '>4.0f')}W / {_num_or(g.power_limit, '>4.0f')}W" if g.power is not None else "N/A"
    if g.mem_used is not None and g.mem_total:
        mem = f"{g.mem_used / 1024:>5.1f}GiB / {g.mem_total / 1024:>5.1f}GiB"
    else:
        mem = "N/A"
    temp = f"{g.temp:.0f}°C" if g.temp is not None else "N/A"
    util = f"{g.util:.0f}%" if g.util is not None else "N/A"
    return pw, mem, temp, util


def render_devices(m: Monitor, width: int, compact: bool, sel_job: str | None):
    """Rows are jobs and their GPUs; columns are the device table, GPU bars, and the job's CPU/RAM bars.
    A job's CPU/RAM sits on its first data row (first GPU, or a placeholder row for CPU-only jobs).
    Device-table cells use one grid (SUB) for every row: names left-aligned, numbers right-aligned."""
    jobs, streams, dead = m.snapshot()
    fr = Frame(width)
    bold = "bold"
    lines = [fr.border("╒", "═", "╤", "╕")]
    if compact:
        lines.append(fr.row((Text(f" {'GPU':>3}  Job", bold), Text(f"{'Memory-Usage':>21} ", bold),
                             Text(f"{'Temp':>5} {'Pwr:Usage/Cap':>13} {'Util':>4} ", bold)),
                            Text(" memory / utilization", bold), Text(" cores / RAM (vs. allocation)", bold)))
    else:
        lines.append(fr.row((Text(f" {'JOB':<12} NAME", bold), Text(f" {'NODE':<9} {'LEFT':>10} ", bold),
                             Text(f" {'ELAPSED':>10} / {'LIMIT':<10} ", bold)),
                            Text(" GPU", bold), Text(" Job CPU", bold)))
        lines.append(fr.row((Text(f" {'GPU':>3}  Name", bold), Text(f" {'Perf':<4}{'Temp':>16} ", bold),
                             Text(f"{'Pwr:Usage/Cap':>24} ", bold)),
                            Text(" memory", bold), Text(" cores (vs. allocation)", bold)))
        lines.append(fr.row((None, Text(f"{'Memory-Usage':>21} ", bold), Text(f"{'GPU-Util':>24} ", bold)),
                            Text(" utilization", bold), Text(" RAM (vs. limit, excl. cache)", bold)))
    lines.append(fr.border("╞", "═", "╪", "╡"))
    running = [j for j in jobs if j.state == "RUNNING"]
    if not running:
        lines.append(fr.row(Text(" no running jobs", style=B)))
    for ji, job in enumerate(running):
        s = streams.get(job.jobid)
        if ji and not compact:
            lines.append(fr.border("├", "─", "┼", "┤"))
        st = job_stats(job, s)
        cpu_cell = cpu_ram_bars(fr.cw, job, st) if st and fr.cw else None      # compact: side by side
        cpu_rows = cpu_ram_stack(fr.cw, job, st) if st and fr.cw else (None, None)  # full: stacked
        if not compact:
            el, left, lim = job.elapsed_now(), job.left_now(), parse_dur(job.limit)
            ident = Text(" ")
            ident.append(f"{job.display:<12}", style="bold reverse cyan" if job.jobid == sel_job else "bold cyan")
            ident.append(f" {job.name}")
            state = Text(f" {job.nodes[:9]:<9} ")
            state.append(f"{(fmt_dur(left) if left is not None else '-'):>10}", style=left_style(left, lim))
            usage = Text(f" {fmt_dur(el) if el is not None else job.elapsed:>10} / {job.limit[:10]:<10} ", style=B)
            lines.append(fr.row((ident, state, usage)))
            if st and not fr.cw:
                cores, _, used, mlim, cache = st
                lines.append(fr.row(Text(f"   CPU {cores or 0:.1f}/{job.cpus} cores   RAM {human_bytes(used)}/{human_bytes(mlim)}"
                                         f" (+{human_bytes(cache)} cache)", style=B)))
        if s is None:
            msg = Text("   stream stopped, re-checking job…", style=B)
            for e in dead.get(job.jobid, ()):
                msg.append(f"  {e}", style="red")
            lines.append(fr.row(msg))
            continue
        with s.lock:
            gpus = sorted(s.gpus.values(), key=lambda g: (g.host, g.index))
            errors = list(s.errors)
            started = s.started
        if not job.has_gpu:
            ident = Text(f" {'-':>3}  ")
            if compact:
                ident.append(job.display, style="bold reverse cyan" if job.jobid == sel_job else "cyan")
                ident.append(f" {job.name}", style="cyan")
            else:
                ident.append("(no GPUs)", style=B)
            if compact:
                lines.append(fr.row((ident, None, None), None, cpu_cell))
            else:
                lines += [fr.row((ident, None, None), None, cpu_rows[0]), fr.row((None, None, None), None, cpu_rows[1])]
        elif not gpus:
            msg = Text("   waiting for first sample…" if time.time() - started < 20 else "   no GPU data", style="yellow")
            if compact:
                lines.append(fr.row((msg, None, None), None, cpu_cell))
            else:
                lines += [fr.row((msg, None, None), None, cpu_rows[0]), fr.row((None, None, None), None, cpu_rows[1])]
        multi = len({g.host for g in gpus}) > 1
        now = time.time()
        for gi, g in enumerate(gpus):
            stale = now - g.t > STALE_S
            label = f"{g.host[-4:]}:{g.index}" if multi else f"{g.index}"
            pw, mem, temp, uval = gpu_fields(g)
            mfrac = g.mem_used / g.mem_total if g.mem_used is not None and g.mem_total else None
            mval = f"{100 * mfrac:.1f}%" if mfrac is not None else "N/A"
            ustyle = util_style(g.util)
            if compact:
                ident = Text(f" {label:>3}  ")
                ident.append(job.display, style="bold reverse cyan" if job.jobid == sel_job else "cyan")
                ident.append(f" {job.name}", style="cyan")
                usage = Text(f"{temp:>5} {pw:>13} ")
                usage.append(f"{uval:>4}", style=ustyle)
                usage.append(" ")
                cells = (ident, Text(f"{mem:>21} "), usage)
                gpu_cell = two_bars(fr.gw, "MEM", mfrac, mval, "magenta", "UTL", (g.util or 0) / 100, uval,
                                    ustyle) if fr.gw else None
                if stale:
                    cells = tuple(c.copy() for c in cells)
                    for c in cells:
                        c.stylize("bright_black")
                    gpu_cell = Text(f" stale: no sample for {now - g.t:.0f}s", style="yellow")
                lines.append(fr.row(cells, gpu_cell, cpu_cell if gi == 0 else None))
                continue
            # full: two rows per GPU, like nvidia-smi / nvitop; bars stacked MEM over UTL, CPU over RAM
            row1 = (Text(f" {label:>3}  {g.name}"), Text(f" {g.pstate:<4}{temp:>16} "), Text(f"{pw:>24} "))
            util_cell = Text(f"{uval:>24}", style=ustyle) + Text(" ")
            row2 = (None, Text(f"{mem:>21} "), util_cell)
            g1 = Text(" ") + nv_bar("MEM", mfrac, fr.gw - 2, "magenta", mval) if fr.gw else None
            g2 = Text(" ") + nv_bar("UTL", (g.util or 0) / 100, fr.gw - 2, ustyle, uval) if fr.gw else None
            if stale:
                for c in (*row1, *row2[1:]):
                    c.stylize("bright_black")
                g1, g2 = Text(f" stale: no sample for {now - g.t:.0f}s", style="yellow"), None
            c1, c2 = cpu_rows if gi == 0 else (None, None)
            lines += [fr.row(row1, g1, c1), fr.row(row2, g2, c2)]
        for e in errors:
            lines.append(fr.row(Text(f"   ! {e}", style="red")))
    lines.append(fr.border("╘", "═", "╧", "╛"))
    return lines


def render_host(m: Monitor, width: int, sel_job: str | None):
    """nvitop's host panel, for the selected job: its cgroup CPU/RAM against its allocation."""
    if sel_job is None:
        return []
    jobs, streams, _ = m.snapshot()
    job = next((j for j in jobs if j.jobid == sel_job), None)
    s = streams.get(sel_job)
    st = job_stats(job, s) if job else None
    if not st:
        return []
    with s.lock:
        nodes = dict(s.node)
    cores, cfrac, used, mlim, cache = st
    host, n0 = min(nodes.items())
    rw = 44 if width >= 110 else 0               # right-hand boxes: load average / cache
    lw = width - (rw + 2 if rw else 0) - 4       # inside "[ ... ]"
    el = job.elapsed_now()
    l1 = Text("[ ", style=B) + nv_bar("CPU", cfrac, lw, "cyan",
                                     f"{100 * cfrac:.1f}%" if cfrac is not None else "N/A",
                                     f"{cores:.1f}/{job.cpus} cores   ELAPSED: {fmt_dur(el) if el is not None else job.elapsed}"
                                     if cores is not None else "") + Text(" ]", style=B)
    mfrac = used / mlim if mlim else None
    l2 = Text("[ ", style=B) + nv_bar("MEM", mfrac, lw, RAM_STYLE,
                                     f"{100 * mfrac:.1f}%" if mfrac is not None else "N/A",
                                     f"USED: {human_bytes(used)}iB of {human_bytes(mlim)}iB") + Text(" ]", style=B)
    if rw:
        load = n0.get("load") or []
        nc = n0.get("ncores")
        who = f"{host} " if len(nodes) > 1 else ""
        lt = Text(f"( {who}Load Average: ", style=B)
        lt.append(" ".join(f"{x:5.2f}" for x in load) if load else "-", style="bold")
        lt.append(f" / {nc:.0f} )" if nc else " )", style=B)
        l1 += Text("  ") + fit(lt, rw)
        cfrac2 = cache / mlim if mlim else None
        l2 += Text("  [ ", style=B) + nv_bar("CACHE", cfrac2, rw - 4, "bright_black",
                                            human_bytes(cache) + "iB" if cache else "0") + Text(" ]", style=B)
    return [l1, l2]


def job_series(job: Job, s: Stream, t_end, res, n):
    """Bucketed (util frac, mem frac, cpu frac, ram frac) for one job, averaged over its GPUs/nodes."""
    carry = max(3 * res, 120)
    utils, mems, cpus, rams = [], [], [], []
    with s.lock:
        gpus = list(s.gpus.values())
        bf = {g.uuid: s.backfill.get(g.uuid, []) for g in gpus}
        nh = [list(h) for h in s.nhist.values()]
        hist = {g.uuid: list(g.hist) if res >= 30 else list(g.hist)[-(n * res + 200):] for g in gpus}
    for g in gpus:
        live = hist[g.uuid]
        first = live[0][0] if live else float("inf")
        b = [x for x in bf[g.uuid] if x[0] < first]
        tot = g.mem_total or 1
        utils.append(bucketize([[(t, u / 100 if u is not None else None) for t, u, _ in b],
                                [(t, u / 100 if u is not None else None) for t, u, _ in live]], t_end, res, n, carry))
        mems.append(bucketize([[(t, mm / tot if mm is not None else None) for t, _, mm in b],
                               [(t, mm / tot if mm is not None else None) for t, _, mm in live]], t_end, res, n, carry))
    for h in nh:
        cpus.append(bucketize([[(t, c / job.cpus * job.nnodes if c is not None and job.cpus else None) for t, c, _, _ in h]],
                              t_end, res, n, carry))
        rams.append(bucketize([[(t, a / (lim or parse_mem(job.mem_req) or 1) if a is not None else None)
                                for t, _, a, lim in h]], t_end, res, n, carry))
    return mean_buckets(utils), mean_buckets(mems), mean_buckets(cpus), mean_buckets(rams)


def series_stats(vals, last_n):
    vs = [v for v in vals[-last_n:] if v is not None]
    if not vs:
        return None, None
    return sum(vs) / len(vs), sorted(vs)[len(vs) // 10]


def chart_block(title_up, vals_up, style_up, title_dn, vals_dn, style_dn, ncols, height, res):
    rows = [title_up]
    rows += braille(vals_up, height, True, style_up)
    rows.append(time_axis(ncols, res))
    rows += braille(vals_dn, height, False, style_dn)
    rows.append(title_dn)
    return [fit(r, ncols) for r in rows]


def render_charts(m: Monitor, width: int, sel_job: str | None, res: int, height: int):
    jobs, streams, _ = m.snapshot()
    running = [j for j in jobs if j.state == "RUNNING" and j.jobid in streams]
    if not running:
        return []
    targets = [j for j in running if j.jobid == sel_job] or running
    gw = width - 2 if width < 120 else int(width * 0.62)
    hw = width - gw - 4   # " " + left + " │ " + right
    t_end = time.time()
    per = [job_series(j, streams[j.jobid], t_end, res, 2 * gw) for j in targets]
    util = mean_buckets([p[0] for p in per])
    mem = mean_buckets([p[1] for p in per])
    who = f"{targets[0].display} {targets[0].name}" if len(targets) == 1 else f"all {len(targets)} jobs (avg)"
    avg5, p10 = series_stats(util, int(300 / res) or 1)
    cur_u = next((v for v in reversed(util) if v is not None), None)
    cur_m = next((v for v in reversed(mem) if v is not None), None)
    tu = Text(" GPU UTL ", style="bold")
    tu.append(f"{100 * cur_u:.0f}%" if cur_u is not None else "-", style=util_style(100 * cur_u if cur_u is not None else None))
    if avg5 is not None:
        tu.append(f"   5m avg {100 * avg5:.0f}%  p10 {100 * p10:.0f}%", style=B)
    tu.append(f"   {who}   {res}s/dot (+/-)", style=B)
    tm = Text(" GPU MEM ", style="bold")
    tm.append(f"{100 * cur_m:.1f}%" if cur_m is not None else "-", style="magenta")
    left = chart_block(tu, util, lambda v: util_style(100 * v if v is not None else None),
                       tm, mem, mem_style, gw, height, res)
    if hw < 30:
        return [Text(" ") + r for r in left]
    hcols = hw
    per_h = [job_series(j, streams[j.jobid], t_end, res, 2 * hcols) for j in targets]
    cpu = mean_buckets([p[2] for p in per_h])
    ram = mean_buckets([p[3] for p in per_h])
    cur_c = next((v for v in reversed(cpu) if v is not None), None)
    cur_r = next((v for v in reversed(ram) if v is not None), None)
    tc = Text(" CPU ", style="bold")
    tc.append(f"{100 * cur_c:.0f}% of alloc" if cur_c is not None else "-", style="cyan")
    tr = Text(" RAM ", style="bold")
    tr.append(f"{100 * cur_r:.0f}% of limit (excl. cache)" if cur_r is not None else "-", style=RAM_STYLE)
    right = chart_block(tc, cpu, lambda v: "cyan", tr, ram, lambda v: RAM_STYLE, hcols, height, res)
    return [Text(" ") + lft + Text(" │ ", style=B) + rgt for lft, rgt in zip(left, right)]


@dataclass
class ProcRow:
    key: tuple
    jobid: str
    display: str
    name: str
    gres: str            # gres/gpu: prefix stripped, e.g. l40s:1
    node: str
    state: str
    reason: str
    jtime: float | None  # job elapsed (s)
    left: float | None   # job time left (s)
    limit: float | None
    host: str = ""
    gpu: int = -1
    gpu_label: str = "-"
    pid: int = 0
    gmem: float | None = None
    sm: float | None = None
    gmbw: float | None = None
    cpu: float | None = None
    mem: float | None = None
    args: str = ""

    @property
    def pending(self):
        return self.state != "RUNNING"


# key, header, width, default descending?, left-aligned?
COLS = [
    ("job", "JOB", 18, False, True),    # fits pending array ranges, e.g. 1234567_[10-99%4]
    ("name", "NAME", 16, False, True),
    ("gres", "GRES", 9, False, True),
    ("time", "TIME", 10, True, False),
    ("node", "NODE", 9, False, True),
    ("gpu", "GPU", 6, False, False),       # only shown when some job has several GPUs
    ("pid", "PID", 8, False, False),       # only shown when some job has several processes
    ("gmem", "GPU-MEM", 8, True, False),
    ("sm", "%SM", 4, True, False),
    ("gmbw", "%GMBW", 5, True, False),
    ("cpu", "%CPU", 6, True, False),
    ("mem", "%MEM", 5, True, False),
]


SORT_KEYS = {"j": "job", "n": "name", "r": "gres", "t": "time", "d": "node", "i": "gpu", "p": "pid",
             "g": "gmem", "s": "sm", "b": "gmbw", "c": "cpu", "m": "mem"}


def proc_rows(m: Monitor):
    jobs, streams, _ = m.snapshot()
    rows = []
    for job in jobs:
        running = job.state == "RUNNING"
        base = {"jobid": job.jobid, "display": job.display, "name": job.name, "gres": gres_short(job.gres),
                "node": job.nodes if running else "", "state": job.state, "reason": job.reason,
                "jtime": job.elapsed_now() if running else None, "left": job.left_now() if running else None,
                "limit": parse_dur(job.limit)}
        s = streams.get(job.jobid)
        if job.state != "RUNNING" or s is None:
            rows.append(ProcRow(key=("job", job.jobid), **base))
            continue
        n_before = len(rows)
        with s.lock:
            gpus = {(g.host, g.uuid): g for g in s.gpus.values()}
            multi = len({g.host for g in s.gpus.values()}) > 1
            for host, procs in s.procs.items():
                lim = job_mem_limit(job, s.node.get(host, {}))
                for p in procs:
                    g = gpus.get((host, p.gpu_uuid))
                    gi = g.index if g else -1
                    sm, gmbw = s.sm_of(host, p.pid, gi)
                    rows.append(ProcRow(key=(job.jobid, host, p.pid, p.gpu_uuid), **base, host=host, gpu=gi,
                                        gpu_label=(f"{host[-4:]}:{gi}" if multi else str(gi)) if gi >= 0 else "-",
                                        pid=p.pid, gmem=p.gpu_mem if gi >= 0 else None, sm=sm, gmbw=gmbw,
                                        cpu=p.cpu_pct, mem=100 * p.rss_kb * 1024 / lim if lim else None, args=p.args))
        if len(rows) == n_before:   # running, but no processes seen (yet)
            rows.append(ProcRow(key=("job", job.jobid), **base, args="(no processes yet)"))
    return rows


def sort_rows(rows, key, reverse):
    def k(r):
        v = {"job": (r.jobid, r.gpu, r.pid), "name": r.name, "gres": r.gres, "time": r.jtime, "node": r.node,
             "gpu": (r.host, r.gpu, r.pid), "pid": r.pid, "gmem": r.gmem, "sm": r.sm, "gmbw": r.gmbw,
             "cpu": r.cpu, "mem": r.mem}[key]
        return (v is None, v if v is not None else 0)
    desc = {c[0]: c[3] for c in COLS}[key] != reverse
    running = [r for r in rows if not r.pending]
    have = sorted([r for r in running if not k(r)[0]], key=k, reverse=desc)
    # rows missing the sort value stay below, pending jobs always last (in job order)
    return have + [r for r in running if k(r)[0]] + [r for r in rows if r.pending]


def visible_cols(rows):
    """Drop GPU / PID unless they distinguish rows within some job."""
    per_job_gpus, per_job_rows = {}, {}
    for r in rows:
        if not r.pending:
            per_job_gpus.setdefault(r.jobid, set()).add((r.host, r.gpu))
            per_job_rows[r.jobid] = per_job_rows.get(r.jobid, 0) + 1
    show = {"gpu": any(len(v) > 1 for v in per_job_gpus.values()),
            "pid": any(v > 1 for v in per_job_rows.values())}
    return [c for c in COLS if show.get(c[0], True)]


def _cell(r, key):
    """(text, style) for one process-table cell."""
    if key == "job":
        return r.display, "cyan" if not r.pending else B
    if key == "name":
        return r.name, None
    if key == "gres":
        return r.gres, None
    if key == "time":
        if r.jtime is None:
            return "-", B
        return fmt_dur(r.jtime), left_style(r.left, r.limit) or None
    if key == "node":
        return r.node or "-", B
    if key == "gpu":
        return r.gpu_label, None
    if key == "pid":
        return (str(r.pid) if r.pid else "-"), B
    if key == "gmem":
        return (f"{r.gmem / 1024:.1f}GiB" if r.gmem is not None else "-"), "magenta"
    if key == "sm":
        return (f"{r.sm:.0f}" if r.sm is not None else "-"), util_style(r.sm)
    if key == "gmbw":
        return (f"{r.gmbw:.0f}" if r.gmbw is not None else "-"), None
    if key == "cpu":
        return (f"{r.cpu:.1f}" if r.cpu is not None else "-"), None
    if key == "mem":
        return (f"{r.mem:.1f}" if r.mem is not None else "-"), None
    return "-", None


def cell_text_at(rows, r, x):
    """Full, untruncated text of the process-table cell at column x of row r (for copying)."""
    pos = 2   # "│" + leading space
    for key, _, w, _, _ in visible_cols(rows):
        if x < pos + w:
            return (r.node or "-") if key == "node" else _cell(r, key)[0] if x >= pos else None
        pos += w + 1
    return r.reason if r.pending else r.args


def render_procs(rows, width, height, cursor, offset, hscroll, sort_key, sort_rev, user):
    inner = width - 2
    cols = visible_cols(rows)
    out = [Text("╒" + "═" * inner + "╕", style=B)]
    title = Text(" Jobs & processes:", style="bold")
    who = f"{user}@{os.uname().nodename.split('.')[0]} "
    title.append(" " * max(1, inner - len(title.plain) - len(who)))
    title.append(who, style="bold")
    out.append(Text("│", style=B) + fit(title, inner) + Text("│", style=B))
    hdr = Text(" ")
    for key, name, w, desc, left in cols:
        mark = ("▼" if desc != sort_rev else "▲") if key == sort_key else ""
        label = f"{name}{mark}"
        hdr.append(f"{label:<{w}} " if left else f"{label:>{w}} ", style="bold underline" if key == sort_key else "bold")
    hdr.append("COMMAND / REASON", style="bold")
    out.append(Text("│", style=B) + fit(hdr, inner) + Text("│", style=B))
    out.append(Text("╞" + "═" * inner + "╡", style=B))
    body_h = max(0, height - len(out) - 1)
    if not rows:
        out.append(Text("│", style=B) + fit(Text("  no running or pending jobs", style=B), inner) + Text("│", style=B))
        body_h -= 1
    for i, r in enumerate(rows[offset: offset + body_h], start=offset):
        t = Text(" ")
        for key, _, w, _, left in cols:
            val, style = _cell(r, key)
            val = val[:w]
            t.append(f"{val:<{w}} " if left else f"{val:>{w}} ", style=style or "")
        if r.pending:
            t.append(f"({r.reason})", style="yellow")
        else:
            t.append(short_args(r.args)[hscroll:])
        line = fit(t, inner)
        if r.pending:
            line.stylize("bright_black", 0, len(line.plain) - len(f"({r.reason})") - 1)
        if i == cursor:
            line = Text(line.plain, style="bold black on cyan")   # one uniform bar, like nvitop
        out.append(Text("│", style=B) + line + Text("│", style=B))
    while len(out) < height - 1:
        out.append(Text("│", style=B) + Text(" " * inner) + Text("│", style=B))
    out.append(Text("╘" + "═" * inner + "╛", style=B))
    return out[:height]
