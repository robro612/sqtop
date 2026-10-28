"""Textual app and screens (main, help, process metrics, environment, log)."""
import os
import signal
import subprocess
import threading
import time

from rich.console import Group
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static

from .core import PDEATH, RES_STEPS, STEP_NAME, Job, Monitor, human_bytes, job_mem_limit, read_log_lines, scontrol_job, short_args
from .render import (
    COLS,
    RAM_STYLE,
    SORT_KEYS,
    B,
    ProcRow,
    braille,
    bucketize,
    cell_text_at,
    fit,
    mem_style,
    proc_rows,
    render_charts,
    render_devices,
    render_host,
    render_procs,
    sort_rows,
    time_axis,
    util_style,
)

CSS = """
Screen { background: ansi_default; color: ansi_default; }
#header, #status { height: 1; }
#devices, #host, #charts { height: auto; }
#procs { height: 1fr; }
.pad { padding: 0 1; }
"""


HELP = """\
sqtop: nvitop-style view of your SLURM jobs' GPUs (read-only)

  a f c    display mode: auto / full / compact      h ?     this help
  r F5     re-poll squeue now                       q ^C    quit

  ↑ ↓      select process                           Esc     clear selection
  Home End first / last process                     PgUp PgDn  page through processes
  ← → ^ $  scroll COMMAND column (also ^A ^E)       wheel   scroll, click selects
  double-click a cell: copy its full text (via your terminal's OSC 52 clipboard support)

  Enter    process metrics (charts for that process + job details)
  e        process environment (fetched from the node on demand)
  l        tail the selected job's log (StdOut/StdErr; Tab switches)
  + -      zoom charts (seconds per braille dot)

  The bottom table is one row per job process (like squeue, plus live metrics);
  pending jobs are listed last with their reason. TIME is the job's elapsed time.
  Charts show the selected row's job, or all jobs averaged when nothing is selected.
  History before sqtop started comes from Prometheus (1-minute resolution).

  Sort:  oj JOB   on NAME   or GRES   ot TIME   od NODE   og GPU-MEM   os %SM
         ob %GMBW   oc %CPU   om %MEM   oi GPU   op PID
         (GPU / PID columns appear only when a job has several GPUs / processes)
         , .  previous / next sort column        /  invert sort order
         (lowercase = default direction, uppercase = reversed)

  Colors: GPU util < 30% red, < 70% yellow, else green (low util is the problem
  for training jobs, so this is inverted from nvitop's load coloring).
  %MEM is a process's RSS as a share of its job's memory limit.
  RAM excludes page cache, which fills to the cgroup limit but is reclaimable.

Press any key to return."""


class Lines(Widget):
    """Renders whatever list of Text lines its owner gives it for the current size."""

    def __init__(self, fn, **kw):
        super().__init__(**kw)
        self.fn = fn

    def render(self):
        return Group(*self.fn(self.size.width, self.size.height))


class ProcPanel(Lines):
    def on_mouse_scroll_down(self, event):
        self.screen.move_cursor(+3)

    def on_mouse_scroll_up(self, event):
        self.screen.move_cursor(-3)

    def on_click(self, event):
        # the first click's selection can show/hide the host panel and shift this one, so the
        # double-click's y is stale: copy from the row the first click selected
        if event.chain == 2:
            self.screen.copy_cell(event.x)
        else:
            self.screen.click_row(event.y - 4)


class MainScreen(Screen):
    def __init__(self, m: Monitor):
        super().__init__()
        self.m = m
        self.mode = "a"
        self.res_i = 0
        self.sort_key, self.sort_rev = "job", False
        self.rows: list[ProcRow] = []
        self.sel = None          # selected row key
        self.row_off = 0
        self.hscroll = 0
        self.await_sort = False
        self.msg, self.msg_t = "", 0.0

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield Static(id="devices")
        yield Static(id="host")
        yield Static(id="charts")
        yield ProcPanel(self.proc_lines, id="procs")
        yield Static(id="status")

    def on_mount(self):
        self.set_interval(0.5, self.tick)
        self.tick()

    # -- state helpers
    @property
    def res(self):
        return RES_STEPS[self.res_i]

    def cur_index(self):
        if self.sel is None:
            return None
        return next((i for i, r in enumerate(self.rows) if r.key == self.sel), None)

    def sel_row(self):
        c = self.cur_index()
        return self.rows[c] if c is not None else None

    def sel_job(self):
        r = self.sel_row()
        return r.jobid if r else None

    def note(self, msg):
        self.msg, self.msg_t = msg, time.time()
        self.tick()

    def layout_for(self, w, chart_h):
        """-> (compact, show_charts, device lines). Auto mode degrades full+charts -> full -> compact
        so the process panel always keeps a few rows."""
        if self.mode == "c":
            return True, False, render_devices(self.m, w, True, self.sel_job())
        full = render_devices(self.m, w, False, self.sel_job())
        if self.mode == "f":
            return False, True, full
        avail = self.size.height - 2 - (2 if self.sel_job() else 0)   # header + status (+ host panel)
        need = 5 + min(max(1, len(self.rows)), 6)         # process box + a few rows
        if len(full) + chart_h + need <= avail:
            return False, True, full
        if len(full) + need <= avail:
            return False, False, full
        return True, False, render_devices(self.m, w, True, self.sel_job())

    # -- rendering
    def tick(self):
        w = self.size.width or 120
        self.rows = sort_rows(proc_rows(self.m), self.sort_key, self.sort_rev)
        if self.sel is not None and self.cur_index() is None:
            self.sel = None   # process went away
        ch = 4 if self.size.height >= 60 else 3
        compact, show_charts, dev = self.layout_for(w, 2 * ch + 3)
        jobs, streams, _ = self.m.snapshot()
        nrun = sum(1 for j in jobs if j.state == "RUNNING")
        npend = sum(1 for j in jobs if j.state == "PENDING")
        utils = [g.util for s in streams.values() for g in s.gpus.values() if g.util is not None]
        h = Text(" sqtop ", style="bold black on green")
        h.append(f"  {time.strftime('%a %b %d %H:%M:%S')}   ", style="bold")
        h.append(f"{nrun} running · {npend} pending · {len(utils)} GPUs", style="")
        if utils:
            avg = sum(utils) / len(utils)
            h.append("   avg util ", style=B)
            h.append(f"{avg:.0f}%", style=util_style(avg))
        mode = {"a": "auto", "f": "full", "c": "compact"}[self.mode]
        shown = "compact" if compact else "full" if show_charts else "full, no charts"
        h.append(f"   mode {mode}{f' ({shown})' if self.mode == 'a' else ''}", style=B)
        self.query_one("#header").update(h)
        self.query_one("#devices").update(Group(*dev))
        host = render_host(self.m, w, self.sel_job())
        self.query_one("#host").display = bool(host)
        self.query_one("#host").update(Group(*host))
        charts = self.query_one("#charts")
        charts.display = show_charts
        if show_charts:
            charts.update(Group(*render_charts(self.m, w, self.sel_job(), self.res, ch)))
        self.query_one("#procs").refresh()
        st = Text()
        if self.msg and time.time() - self.msg_t < 5:
            st.append(f" {self.msg}", style="yellow")
        elif self.await_sort:
            st.append(" sort by: j job  n name  r gres  t time  d node  g gpu-mem  s %sm  b %gmbw  c %cpu  m %mem"
                      "  i gpu  p pid", style="yellow")
        else:
            st.append(" h help · q quit · r refresh · ↑↓ select · Enter metrics · e env · l log · o sort · a/f/c mode · +/- zoom",
                      style=B)
        age = time.time() - self.m.last_squeue if self.m.last_squeue else 0
        right = f"gpu {self.m.a.gpu_ms}ms · procs {self.m.a.proc_s:g}s · squeue {self.m.a.squeue_s:g}s ({age:.0f}s ago) "
        st.append(" " * max(1, w - len(st.plain) - len(right)))
        st.append(right, style=B)
        if self.m.squeue_err:
            st = Text(f" squeue error: {self.m.squeue_err}", style="red")
        self.query_one("#status").update(st)

    def proc_lines(self, w, h):
        body_h = max(1, h - 5)
        c = self.cur_index()
        if c is not None:
            if c < self.row_off:
                self.row_off = c
            elif c >= self.row_off + body_h:
                self.row_off = c - body_h + 1
        self.row_off = max(0, min(self.row_off, max(0, len(self.rows) - body_h)))
        return render_procs(self.rows, w, h, c, self.row_off, self.hscroll, self.sort_key, self.sort_rev, self.m.a.user)

    # -- navigation
    def move_cursor(self, d):
        if not self.rows:
            return
        c = self.cur_index()
        c = (0 if d > 0 else len(self.rows) - 1) if c is None else max(0, min(len(self.rows) - 1, c + d))
        self.sel = self.rows[c].key
        self.tick()

    def copy_cell(self, x):
        r = self.sel_row()
        if r:
            text = cell_text_at(self.rows, r, x)
            if text:
                self.app.copy_to_clipboard(text)
                self.note(f"copied: {text[:80]}" + ("…" if len(text) > 80 else ""))

    def click_row(self, i):
        i += self.row_off
        if 0 <= i < len(self.rows):
            self.sel = self.rows[i].key
            self.tick()

    def on_key(self, event):
        k, ch = event.key, event.character
        page = max(1, self.query_one("#procs").size.height - 6)
        if self.await_sort:
            self.await_sort = False
            if ch and ch.lower() in SORT_KEYS:
                self.sort_key = SORT_KEYS[ch.lower()]
                self.sort_rev = ch.isupper()
            self.tick()
            event.stop()
            return
        if k in ("q",):
            self.app.exit()
        elif ch in ("h", "?"):
            self.app.push_screen(HelpScreen())
        elif ch in ("a", "f", "c"):
            self.mode = ch
            self.tick()
        elif k in ("r", "f5"):
            self.m.kick.set()
            self.note("re-polling squeue…")
        elif k == "down":
            self.move_cursor(+1)
        elif k == "up":
            self.move_cursor(-1)
        elif k == "pagedown":
            self.move_cursor(+page)
        elif k == "pageup":
            self.move_cursor(-page)
        elif k == "home" and self.rows:
            self.sel = self.rows[0].key
            self.tick()
        elif k == "end" and self.rows:
            self.sel = self.rows[-1].key
            self.tick()
        elif k == "escape":
            self.sel = None
            self.tick()
        elif k == "right":
            self.hscroll += 8
        elif k == "left":
            self.hscroll = max(0, self.hscroll - 8)
        elif ch == "^" or k == "ctrl+a":
            self.hscroll = 0
        elif ch == "$" or k == "ctrl+e":
            r = self.sel_row() or (self.rows[0] if self.rows else None)
            self.hscroll = max(0, len(short_args(r.args)) - 30) if r else 0
        elif ch == "o":
            self.await_sort = True
            self.tick()
        elif ch in (",", "."):
            keys = [c[0] for c in COLS]
            self.sort_key = keys[(keys.index(self.sort_key) + (1 if ch == "." else -1)) % len(keys)]
            self.sort_rev = False
            self.tick()
        elif ch == "/":
            self.sort_rev = not self.sort_rev
            self.tick()
        elif ch in ("+", "="):
            self.res_i = max(0, self.res_i - 1)
            self.tick()
        elif ch in ("-", "_"):
            self.res_i = min(len(RES_STEPS) - 1, self.res_i + 1)
            self.tick()
        elif k in ("enter",) or ch == "e":
            r = self.sel_row()
            if r and not r.pid:
                self.note("job is pending" if r.pending else "no process selected in that job yet")
            elif r:
                self.app.push_screen(MetricsScreen(self.m, r) if k == "enter" else EnvScreen(r))
            else:
                self.note("select a process first (↑/↓)")
        elif ch == "l":
            r = self.sel_row()
            jobs, _, _ = self.m.snapshot()
            running = [j for j in jobs if j.state == "RUNNING"]
            jid = r.jobid if r else (running[0].jobid if len(running) == 1 else None)
            job = next((j for j in jobs if j.jobid == jid), None)
            if job and job.state != "RUNNING":
                self.note("job is pending: no log yet")
            elif job:
                self.app.push_screen(LogScreen(job))
            else:
                self.note("select a process first (↑/↓) to pick its job's log")
        else:
            return
        self.query_one("#procs").refresh()
        event.stop()


class TextScreen(Screen):
    """Scrollable text page; Esc/q returns."""

    BACK_KEYS = ("escape", "q")

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with VerticalScroll(id="body"):
            yield Static(id="content", classes="pad")
        yield Static(id="status")

    def on_mount(self):
        self.query_one("#body").focus()

    # Textual calls on_key for every class in the MRO, so subclasses must not define their own:
    # they override extra_key() instead (return True if handled).
    def on_key(self, event):
        if self.extra_key(event):
            event.stop()
        elif event.key in self.BACK_KEYS:
            self.app.pop_screen()
            event.stop()

    def extra_key(self, event):
        return False


class HelpScreen(TextScreen):
    def on_mount(self):
        super().on_mount()
        self.query_one("#header").update(Text(" sqtop help ", style="bold reverse green"))
        self.query_one("#content").update(Text(HELP))

    def extra_key(self, event):
        if event.key not in ("up", "down", "pageup", "pagedown"):
            self.app.pop_screen()
            return True
        return False


class EnvScreen(TextScreen):
    def __init__(self, r: ProcRow):
        super().__init__()
        self.r = r

    def on_mount(self):
        super().on_mount()
        self.query_one("#header").update(Text(f" environment  pid {self.r.pid}  job {self.r.display}  {self.r.host} ",
                                              style="bold reverse green"))
        self.query_one("#content").update(Text("fetching via srun --overlap…", style="yellow"))
        self.query_one("#status").update(Text(" Esc/q back · ↑↓ PgUp PgDn scroll", style=B))
        threading.Thread(target=self.fetch, daemon=True).start()

    def fetch(self):
        cmd = [*PDEATH, "srun", "--overlap", f"--jobid={self.r.jobid}", f"--job-name={STEP_NAME}-env", "-w", self.r.host, "--chdir=/",
               "-N1", "-n1", "--quiet", "bash", "-c", 'cat /proc/$1/environ', "_", str(self.r.pid)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
            out, err = p.stdout, p.stderr.strip()
        except Exception as e:
            out, err = "", str(e)
        t = Text()
        for line in sorted(v for v in out.split("\0") if v):
            k, _, v = line.partition("=")
            t.append(k, style="bold cyan")
            t.append("=")
            t.append(v + "\n")
        if err:
            t.append(err, style="red")
        self.app.call_from_thread(self.query_one("#content").update, t if t.plain else Text("(empty)", style=B))


class LogScreen(TextScreen):
    TAIL_BYTES = 256 * 1024

    def __init__(self, job: Job):
        super().__init__()
        self.job = job
        self.paths: list[str] = []
        self.which = 0
        self.size_seen = -1

    def on_mount(self):
        super().on_mount()
        info = scontrol_job(self.job.jobid)
        self.paths = list(dict.fromkeys(p for p in (info.get("StdOut"), info.get("StdErr")) if p))
        self.set_interval(1.0, self.poll)
        self.poll()

    def poll(self):
        path = self.paths[self.which] if self.paths else None
        which = ("stdout" if self.which == 0 else "stderr") if len(self.paths) > 1 else "stdout+stderr"
        self.query_one("#header").update(Text(f" log  {self.job.display} {self.job.name}  {which}: {path or '?'} ",
                                              style="bold reverse green"))
        self.query_one("#status").update(Text(" Esc/q back · Tab stdout/stderr · g/G top/bottom · follows when at bottom",
                                              style=B))
        if not path:
            self.query_one("#content").update(Text("no StdOut path from scontrol", style="red"))
            return
        try:
            size = os.path.getsize(path)
            if size == self.size_seen:
                return
            self.size_seen = size
            lines = read_log_lines(path, 2000)
        except OSError as e:
            self.query_one("#content").update(Text(str(e), style="red"))
            return
        body = self.query_one("#body")
        at_bottom = body.scroll_y >= body.max_scroll_y - 1
        self.query_one("#content").update(Text.from_ansi("\n".join(lines)))
        if at_bottom:
            self.call_after_refresh(body.scroll_end, animate=False)

    def extra_key(self, event):
        body = self.query_one("#body")
        if event.key == "tab" and len(self.paths) > 1:
            self.which = 1 - self.which
            self.size_seen = -1
            self.poll()
        elif event.character == "g":
            body.scroll_home(animate=False)
        elif event.character == "G":
            body.scroll_end(animate=False)
        else:
            return False
        return True


class MetricsScreen(Screen):
    def __init__(self, m: Monitor, r: ProcRow):
        super().__init__()
        self.m, self.r = m, r
        self.res_i = 0
        self.info: dict = {}

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield Lines(self.draw_lines, id="procs")
        yield Static(id="status")

    def on_mount(self):
        threading.Thread(target=lambda: setattr(self, "info", scontrol_job(self.r.jobid)), daemon=True).start()
        self.set_interval(1.0, self.tick)
        self.tick()

    def tick(self):
        r = self.r
        self.query_one("#header").update(Text(f" process metrics  pid {r.pid}  job {r.display}  {r.host}  GPU {r.gpu_label} ",
                                              style="bold reverse green"))
        self.query_one("#status").update(Text(f" Esc/q/Enter back · +/- zoom ({RES_STEPS[self.res_i]}s/dot) · e env · l log",
                                              style=B))
        self.query_one("#procs").refresh()

    def draw_lines(self, w, h):
        r = self.r
        _, streams, _ = self.m.snapshot()
        s = streams.get(r.jobid)
        jobs, _, _ = self.m.snapshot()
        job = next((j for j in jobs if j.jobid == r.jobid), None)
        out = []
        if s is None or job is None:
            return [Text(" job is no longer running", style="yellow")]
        with s.lock:
            hist = list(s.phist.get((r.host, r.pid), []))
            g = next((g for g in s.gpus.values() if g.host == r.host and g.index == r.gpu), None)
            node = dict(s.node.get(r.host, {}))
            proc = next((p for p in s.procs.get(r.host, []) if p.pid == r.pid), None)
        res = RES_STEPS[self.res_i]
        ncols = w - 4
        t_end = time.time()
        gtot = (g.mem_total if g else None) or 1
        lim = job_mem_limit(job, node) or 1
        carry = max(3 * res, 10)
        sm = bucketize([[(t, v / 100 if v is not None else None) for t, _, _, _, v in hist]], t_end, res, 2 * ncols, carry)
        gm = bucketize([[(t, v / gtot) for t, _, _, v, _ in hist]], t_end, res, 2 * ncols, carry)
        cpu = bucketize([[(t, v / 100 / max(1, job.cpus) if v is not None else None) for t, v, _, _, _ in hist]],
                        t_end, res, 2 * ncols, carry)
        rss = bucketize([[(t, v * 1024 / lim) for t, _, v, _, _ in hist]], t_end, res, 2 * ncols, carry)
        last = hist[-1] if hist else (None,) * 5
        head = Text(" ")
        head.append(short_args(proc.args if proc else r.args)[: max(10, w * 2 - 4)], style="bold")
        out.append(head)
        info = self.info
        meta = Text(" ", style=B)
        for k in ("WorkDir", "StdOut", "SubmitTime", "StartTime", "TimeLimit", "AllocTRES"):
            if info.get(k):
                meta.append(f"{k}=", style=B)
                meta.append(f"{info[k]}  ")
        out.append(meta)
        out.append(Text(""))
        hh = max(2, (h - 6 - 5) // 5)   # 5 charts (title + rows each) + header lines + axis

        def block(title, vals, style):
            rows = [title] + braille(vals, hh, True, style)
            return [Text("  ") + fit(x, ncols) for x in rows]

        t1 = Text(" %SM ", style="bold") + Text(f"{last[4]:.0f}%" if last[4] is not None else "-", style=util_style(last[4]))
        t2 = Text(" GPU-MEM ", style="bold") + Text(f"{last[3] / 1024:.1f}GiB" if last[3] is not None else "-", style="magenta")
        t3 = Text(" %CPU ", style="bold") + Text(f"{last[1]:.0f}% (scale: {job.cpus} cores)" if last[1] is not None else "-",
                                                  style="cyan")
        t4 = Text(" RSS ", style="bold") + Text(f"{human_bytes(last[2] * 1024)} of {human_bytes(lim)} job limit"
                                                if last[2] is not None else "-", style=RAM_STYLE)
        if g:
            with s.lock:
                live = list(g.hist)
                bf = [x for x in s.backfill.get(g.uuid, []) if not live or x[0] < live[0][0]]
            dev = bucketize([[(t, u / 100 if u is not None else None) for t, u, _ in bf],
                             [(t, u / 100 if u is not None else None) for t, u, _ in live]],
                            t_end, res, 2 * ncols, max(3 * res, 120))
            t0 = Text(f" GPU {r.gpu_label} UTL (whole device, incl. history) ", style="bold")
            t0.append(f"{g.util:.0f}%" if g.util is not None else "-", style=util_style(g.util))
            out += block(t0, dev, lambda v: util_style(100 * v if v is not None else None))
        out += block(t1, sm, lambda v: util_style(100 * v if v is not None else None))
        out += block(t2, gm, mem_style)
        out += block(t3, cpu, lambda v: "cyan")
        out += block(t4, rss, lambda v: RAM_STYLE)
        out.append(Text("  ") + time_axis(ncols, res))
        return out[:h]

    def on_key(self, event):
        ch = event.character
        if event.key in ("escape", "q", "enter"):
            self.app.pop_screen()
        elif ch in ("+", "="):
            self.res_i = max(0, self.res_i - 1)
            self.tick()
        elif ch in ("-", "_"):
            self.res_i = min(len(RES_STEPS) - 1, self.res_i + 1)
            self.tick()
        elif ch == "e":
            self.app.push_screen(EnvScreen(self.r))
        elif ch == "l":
            jobs, _, _ = self.m.snapshot()
            job = next((j for j in jobs if j.jobid == self.r.jobid), None)
            if job:
                self.app.push_screen(LogScreen(job))
        else:
            return
        event.stop()


class SqTop(App):
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("ctrl+c", "quit", show=False, priority=True)]

    def __init__(self, m: Monitor):
        super().__init__(ansi_color=True)
        self.m = m

    def on_mount(self):
        import asyncio
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(sig, self.exit)
        self.push_screen(MainScreen(self.m))
