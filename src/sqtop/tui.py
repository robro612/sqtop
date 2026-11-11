"""Textual app and screens (main, help, job metrics, environment, log)."""
import os
import signal
import subprocess
import threading
import time

from rich.console import Group
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, Static

from .cluster import ClusterMonitor, render_detail, render_summary
from .core import (
    PDEATH,
    RES_STEPS,
    SQUEUE_STEPS,
    STEP_NAME,
    Job,
    Monitor,
    human_bytes,
    job_mem_limit,
    read_log_lines,
    scontrol_job,
    short_args,
)
from .render import (
    RAM_STYLE,
    VIEWS,
    B,
    JobRow,
    ProcRow,
    braille,
    bucketize,
    cell_text_at,
    fit,
    job_row,
    job_series,
    mem_style,
    render_charts,
    render_devices,
    render_host,
    render_table,
    sort_rows,
    table_items,
    time_axis,
    util_style,
)

CSS = """
Screen { background: ansi_default; color: ansi_default; }
#header, #status { height: 1; }
#devices, #host, #charts { height: auto; }
#procs, #jobs { height: 1fr; }
.pad { padding: 0 1; }

SettingsScreen { align: center middle; background: $background 50%; }
#settings { width: 80; height: auto; border: round green; padding: 0 1; background: $surface; }
#settings .title { text-style: bold; color: green; padding-bottom: 1; }
#settings .hint { color: $text-muted; padding-top: 1; }
.setting { height: 1; }
.setting.sel { background: $boost; text-style: bold; }
.setting .lbl { width: 30; }
.setting .val { width: 12; text-align: right; padding-right: 1; }
.setting .key { width: 1fr; color: $text-muted; padding-left: 1; }
.setting Button { min-width: 5; width: 5; height: 1; border: none; padding: 0; }
"""


HELP = """\
sqtop: nvitop-style view of your SLURM jobs' GPUs (read-only)

  a f c    display mode: auto / full / compact      h ?     this help
  r F5     re-poll squeue now                       q ^C    quit
  [ ]      squeue interval shorter / longer (5s … 10m)  s       settings popup (poll rates, mode, view, zoom)

  ↑ ↓      select row                               Esc     clear selection
  Home End first / last row                         PgUp PgDn  page through rows
  ← → ^ $  scroll COMMAND column (also ^A ^E)       wheel   scroll, click selects
  double-click a cell: copy its full text (via your terminal's OSC 52 clipboard support)

  Enter    metrics: job view: charts for the job, its details and its processes;
           process view: charts for that process + job details
  v        toggle the table: one row per job / one row per process
  g        cluster GPUs: free / queued by GPU type or by node (Enter on a type: its nodes + queue)
  e        environment of the selected process / job's main process (fetched on demand)
  l        tail the selected job's log (StdOut/StdErr; Tab switches)
  + -      chart zoom: more / fewer seconds per braille dot

  The bottom table has two sections, GPU jobs then CPU-only jobs, with pending jobs
  listed last in each with their reason. TIME is the job's elapsed time.
  Job view (default): one row per job (like squeue, plus live metrics). COMMAND is the
  job's main process (its oldest, not counting the batch script); PROCS counts its
  processes (GPU jobs: the processes using its GPUs).
  Process view (v): one row per job process; GPU / PID columns appear only when a job
  has several GPUs / processes. The per-GPU device panel above shows in both views.
  Charts show the selected row's job, or all jobs averaged when nothing is selected.
  History before sqtop started comes from Prometheus (1-minute resolution).

  Sort:  oj JOB   on NAME   or GRES   ot TIME   od NODE   og GPU-MEM   os %SM
         oc CPU   om %MEM   job view: ou %UTL  op PROCS
         process view: ob %GMBW  oi GPU  op PID
         , .  previous / next sort column        /  invert sort order
         (lowercase = default direction, uppercase = reversed)

  Colors: GPU util < 30% red, < 70% yellow, else green (low util is the problem
  for training jobs, so this is inverted from nvitop's load coloring).
  GPU-MEM is summed over the job's GPUs; %UTL is their mean device utilization and
  %SM the mean per GPU of its processes' pmon %SM. CPU is cores in use / allocated.
  %MEM is the job's RAM as a share of its memory limit (process view: the process's RSS).
  Process view %CPU is a process's CPU (100 = one core).
  RAM excludes page cache, which fills to the cgroup limit but is reclaimable.

Press any key to return."""


CLUSTER_STEPS = (15, 30, 60, 120, 300, 600)
GPU_MS_STEPS = (250, 500, 1000, 2000, 5000)
PROC_S_STEPS = (1, 2, 5, 10, 30)


def step_interval(cur, d, steps=SQUEUE_STEPS):
    """Next value in `steps` above (d=+1) or below (d=-1) cur; cur needn't be one of the steps."""
    if d > 0:
        return next((x for x in steps if x > cur), steps[-1])
    return next((x for x in reversed(steps) if x < cur), steps[0])


def fmt_interval(s):
    return f"{s / 60:g}m" if s >= 60 and s % 60 == 0 else f"{s:g}s"


class SettingsScreen(ModalScreen):
    """Popup with every adjustable setting: ↑↓ pick, ←→ (or the buttons) change, Esc/s/q close.
    GPU / process sample rates restart each job's monitoring step, so they apply on close."""

    def __init__(self, main: "MainScreen"):
        super().__init__()
        self.main, self.m, self.cm = main, main.m, main.app.cm
        self.i = 0
        self.gpu_ms, self.proc_s = self.m.a.gpu_ms, self.m.a.proc_s   # pending until close
        a, mn = self.m.a, main
        modes = ["a", "f", "c"]
        # (label, shortcut on the main screen, show(), change(d))  d = +1 raises the value (right / + / ])
        self.settings = [
            ("squeue poll interval", "[ ]", lambda: fmt_interval(a.squeue_s),
             lambda d: setattr(a, "squeue_s", step_interval(a.squeue_s, d))),
            ("cluster view poll (g)", "", lambda: fmt_interval(self.cm.interval),
             lambda d: setattr(self.cm, "interval", step_interval(self.cm.interval, d, CLUSTER_STEPS))),
            ("GPU sample interval *", "", lambda: f"{self.gpu_ms}ms",
             lambda d: setattr(self, "gpu_ms", step_interval(self.gpu_ms, d, GPU_MS_STEPS))),
            ("process sample interval *", "", lambda: fmt_interval(self.proc_s),
             lambda d: setattr(self, "proc_s", step_interval(self.proc_s, d, PROC_S_STEPS))),
            ("display mode", "a f c", lambda: {"a": "auto", "f": "full", "c": "compact"}[mn.mode],
             lambda d: setattr(mn, "mode", modes[(modes.index(mn.mode) + d) % 3])),
            ("table rows", "v", lambda: "per job" if mn.view.name == "jobs" else "per process",
             lambda d: mn.toggle_view()),
            ("chart zoom (s per dot)", "+ -", lambda: fmt_interval(RES_STEPS[mn.res_i]),
             lambda d: setattr(mn, "res_i", max(0, min(len(RES_STEPS) - 1, mn.res_i + d)))),
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings"):
            yield Static("sqtop settings", classes="title")
            for n, (label, key, _, _) in enumerate(self.settings):
                with Horizontal(classes="setting", id=f"set{n}"):
                    yield Static(label, classes="lbl")
                    yield Button("−", id=f"dn{n}", compact=True)
                    yield Static("", classes="val", id=f"val{n}")
                    yield Button("+", id=f"up{n}", compact=True)
                    yield Static(key, classes="key")
            yield Static("↑↓ select · ←→ / -+ change · Esc s q close\n"
                         "* applied on close: restarts each job's monitoring step (history is kept)",
                         classes="hint")

    def on_mount(self):
        for b in self.query(Button):
            b.can_focus = False   # keep arrows for the menu itself
        self.refresh_values()

    def refresh_values(self):
        for n, (_, _, show, _) in enumerate(self.settings):
            self.query_one(f"#val{n}", Static).update(show())
            self.query_one(f"#set{n}").set_class(n == self.i, "sel")
        self.main.tick()

    def change(self, n, d):
        self.i = n
        self.settings[n][3](d)
        self.refresh_values()

    def on_button_pressed(self, event):
        bid = event.button.id
        self.change(int(bid[2:]), +1 if bid.startswith("up") else -1)
        event.stop()

    def on_click(self, event):
        w = event.widget
        while w is not None and not (w.id or "").startswith("set"):
            w = w.parent
        if w is not None and w.id[3:].isdigit():
            self.i = int(w.id[3:])
            self.refresh_values()

    def on_key(self, event):
        k, ch = event.key, event.character
        if k in ("escape", "q") or ch == "s":
            self.dismiss()
        elif k == "up":
            self.i = (self.i - 1) % len(self.settings)
            self.refresh_values()
        elif k == "down":
            self.i = (self.i + 1) % len(self.settings)
            self.refresh_values()
        elif k in ("right", "enter") or ch in ("+", "=", "l"):
            self.change(self.i, +1)
        elif k == "left" or ch in ("-", "_", "h"):
            self.change(self.i, -1)
        event.stop()

    def on_unmount(self):
        if (self.gpu_ms, self.proc_s) != (self.m.a.gpu_ms, self.m.a.proc_s):
            self.m.set_rates(self.gpu_ms, self.proc_s)
            self.main.note(f"restarting monitors: gpu {self.gpu_ms}ms · procs {fmt_interval(self.proc_s)}")


class Lines(Widget):
    """Renders whatever list of Text lines its owner gives it for the current size."""

    def __init__(self, fn, **kw):
        super().__init__(**kw)
        self.fn = fn

    def render(self):
        return Group(*self.fn(self.size.width, self.size.height))


class JobPanel(Lines):
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
            self.screen.click_row(event.y - 2)


class MainScreen(Screen):
    def __init__(self, m: Monitor):
        super().__init__()
        self.m = m
        self.mode = "a"
        self.res_i = 0
        self.sort_key, self.sort_rev = "job", False
        self.view = VIEWS["jobs"]
        self.rows: list[JobRow | ProcRow] = []
        self.items: list[tuple] = []
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
        yield JobPanel(self.job_lines, id="jobs")
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
        so the jobs panel always keeps a few rows."""
        if self.mode == "c":
            return True, False, render_devices(self.m, w, True, self.sel_job())
        full = render_devices(self.m, w, False, self.sel_job())
        if self.mode == "f":
            return False, True, full
        avail = self.size.height - 2 - (2 if self.sel_job() else 0)   # header + status (+ host panel)
        need = 3 + min(max(1, len(self.items)), 8)        # jobs box + a few lines
        if len(full) + chart_h + need <= avail:
            return False, True, full
        if len(full) + need <= avail:
            return False, False, full
        return True, False, render_devices(self.m, w, True, self.sel_job())

    # -- rendering
    def tick(self):
        w = self.size.width or 120
        self.rows = sort_rows(self.view, self.view.rows(self.m), self.sort_key, self.sort_rev)
        self.items = table_items(self.rows)
        if self.sel is not None and self.cur_index() is None:
            self.sel = None   # job went away
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
        self.query_one("#jobs").refresh()
        st = Text()
        if self.msg and time.time() - self.msg_t < 5:
            st.append(f" {self.msg}", style="yellow")
        elif self.await_sort:
            hdr = {c[0]: c[1].lower() for c in self.view.COLS}
            st.append(" sort by: " + "  ".join(f"{k} {hdr[v]}" for k, v in self.view.SORT_KEYS.items()), style="yellow")
        else:
            st.append(" h help · q quit · r refresh · ↑↓ select · Enter metrics · v jobs/procs · e env · l log · g gpus · o sort"
                      " · a/f/c mode"
                      " · +/- zoom · [/] squeue rate · s settings",
                      style=B)
        age = time.time() - self.m.last_squeue if self.m.last_squeue else 0
        right = f"gpu {self.m.a.gpu_ms}ms · procs {fmt_interval(self.m.a.proc_s)} · squeue {fmt_interval(self.m.a.squeue_s)} ({age:.0f}s ago) "
        st.append(" " * max(1, w - len(st.plain) - len(right)))
        st.append(right, style=B)
        if self.m.squeue_err:
            st = Text(f" squeue error: {self.m.squeue_err}", style="red")
        self.query_one("#status").update(st)

    def job_lines(self, w, h):
        body_h = max(1, h - 3)
        c = self.cur_index()
        if c is not None:
            ci = self.items.index(("row", c))
            top = ci
            while top > 0 and self.items[top - 1][0] != "row":
                top -= 1   # keep the section's title + header in view
            if top < self.row_off:
                self.row_off = top
            elif ci >= self.row_off + body_h:
                self.row_off = ci - body_h + 1
        self.row_off = max(0, min(self.row_off, max(0, len(self.items) - body_h)))
        return render_table(self.view, self.rows, self.items, w, h, c, self.row_off, self.hscroll, self.sort_key, self.sort_rev,
                           self.m.a.user)

    # -- navigation
    def move_cursor(self, d):
        if not self.rows:
            return
        c = self.cur_index()
        c = (0 if d > 0 else len(self.rows) - 1) if c is None else max(0, min(len(self.rows) - 1, c + d))
        self.sel = self.rows[c].key
        self.tick()

    def toggle_view(self):
        """Switch between one row per job and one row per process, keeping the selected job."""
        jid = self.sel_job()
        self.view = VIEWS["processes" if self.view.name == "jobs" else "jobs"]
        if self.sort_key not in self.view.SORT_KEYS.values():
            self.sort_key, self.sort_rev = "job", False
        self.hscroll = 0
        self.rows = sort_rows(self.view, self.view.rows(self.m), self.sort_key, self.sort_rev)
        self.sel = next((r.key for r in self.rows if r.jobid == jid), None) if jid else None
        self.note(f"view: one row per {'job' if self.view.name == 'jobs' else 'process'}")

    def copy_cell(self, x):
        r = self.sel_row()
        if r:
            text = cell_text_at(self.view, self.rows, r, x)
            if text:
                self.app.copy_to_clipboard(text)
                self.note(f"copied: {text[:80]}" + ("…" if len(text) > 80 else ""))

    def click_row(self, i):
        i += self.row_off
        if 0 <= i < len(self.items) and self.items[i][0] == "row":
            self.sel = self.rows[self.items[i][1]].key
            self.tick()

    def on_key(self, event):
        k, ch = event.key, event.character
        page = max(1, self.query_one("#jobs").size.height - 6)
        if self.await_sort:
            self.await_sort = False
            if ch and ch.lower() in self.view.SORT_KEYS:
                self.sort_key = self.view.SORT_KEYS[ch.lower()]
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
        elif ch == "s":
            self.app.push_screen(SettingsScreen(self))
        elif ch in ("[", "]"):
            self.m.a.squeue_s = step_interval(self.m.a.squeue_s, +1 if ch == "]" else -1)
            self.note(f"squeue every {fmt_interval(self.m.a.squeue_s)}")
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
        elif ch == "g":
            self.app.push_screen(ClusterScreen(self.app.cm))
        elif ch == "v":
            self.toggle_view()
        elif ch == "o":
            self.await_sort = True
            self.tick()
        elif ch in (",", "."):
            keys = [c[0] for c in self.view.COLS]
            self.sort_key = keys[(keys.index(self.sort_key) + (1 if ch == "." else -1)) % len(keys)]
            self.sort_rev = False
            self.tick()
        elif ch == "/":
            self.sort_rev = not self.sort_rev
            self.tick()
        elif ch in ("+", "="):
            self.res_i = min(len(RES_STEPS) - 1, self.res_i + 1)
            self.tick()
        elif ch in ("-", "_"):
            self.res_i = max(0, self.res_i - 1)
            self.tick()
        elif k == "enter":
            r = self.sel_row()
            if r and r.pending:
                self.note("job is pending")
            elif r:
                self.app.push_screen(JobScreen(self.m, r.jobid) if self.view.name == "jobs" or not r.pid
                                     else ProcScreen(self.m, r))
            else:
                self.note("select a job first (↑/↓)")
        elif ch == "e":
            r = self.sel_row()
            if r and not r.pid:
                self.note("job is pending" if r.pending else "no processes seen in that job yet")
            elif r:
                self.app.push_screen(EnvScreen(r))
            else:
                self.note("select a job first (↑/↓)")
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
                self.note("select a job first (↑/↓) to pick its log")
        else:
            return
        self.query_one("#jobs").refresh()
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
    def __init__(self, r: JobRow | ProcRow):
        super().__init__()
        self.r = r

    def on_mount(self):
        super().on_mount()
        self.query_one("#header").update(Text(f" environment  job {self.r.display}  pid {self.r.pid}"
                                              f"{' (main process)' if isinstance(self.r, JobRow) else ''}  {self.r.host} ",
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


class JobScreen(Screen):
    """Full-screen charts for one job, its SLURM details and its processes."""

    def __init__(self, m: Monitor, jobid: str):
        super().__init__()
        self.m, self.jobid = m, jobid
        self.res_i = 0
        self.info: dict = {}
        self.r: JobRow | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield Lines(self.draw_lines, id="procs")
        yield Static(id="status")

    def on_mount(self):
        threading.Thread(target=lambda: setattr(self, "info", scontrol_job(self.jobid)), daemon=True).start()
        self.set_interval(1.0, self.tick)
        self.tick()

    def job(self):
        jobs, streams, _ = self.m.snapshot()
        job = next((j for j in jobs if j.jobid == self.jobid), None)
        return job, streams.get(self.jobid)

    def tick(self):
        job, s = self.job()
        if job:
            self.r = job_row(job, s)
        r = self.r
        hdr = f" job metrics  {r.display}  {r.name}  {r.node}  {r.gres if r.gpu_job else 'CPU-only'} " if r else " job metrics "
        self.query_one("#header").update(Text(hdr, style="bold reverse green"))
        self.query_one("#status").update(Text(f" Esc/q/Enter back · +/- zoom ({RES_STEPS[self.res_i]}s/dot) · e env · l log",
                                              style=B))
        self.query_one("#procs").refresh()

    def proc_table(self, s, job, n, w):
        """Header + up to n process lines (most GPU memory, then most CPU, first)."""
        with s.lock:
            gidx = {(g.host, g.uuid): g.index for g in s.gpus.values()}
            procs = [(h, p, gidx.get((h, p.gpu_uuid), -1), s.sm_of(h, p.pid, gidx.get((h, p.gpu_uuid), -1))[0])
                     for h, ps in s.procs.items() for p in ps]
        procs.sort(key=lambda x: (-x[1].gpu_mem, -(x[1].cpu_pct or 0)))
        multi = job.nnodes > 1
        hdr = Text(" ")
        if multi:
            hdr.append(f"{'NODE':<9} ", style="bold")
        hdr.append(f"{'PID':>8} ", style="bold")
        if job.has_gpu:
            hdr.append(f"{'GPU':>3} {'GPU-MEM':>8} {'%SM':>4} ", style="bold")
        hdr.append(f"{'%CPU':>6} {'RSS':>8}  COMMAND", style="bold")
        out = [Text(f" Processes ({len(procs)})", style="bold"), fit(hdr, w)]
        for h, p, gi, sm in procs[:n]:
            t = Text(" ")
            if multi:
                t.append(f"{h[:9]:<9} ", style=B)
            t.append(f"{p.pid:>8} ", style="bold" if p.pid == (self.r.pid if self.r else 0) else "")
            if job.has_gpu:
                t.append(f"{gi if gi >= 0 else '-':>3} ")
                t.append(f"{p.gpu_mem / 1024:.1f}GiB".rjust(8) + " " if gi >= 0 else f"{'-':>8} ", style="magenta")
                t.append(f"{sm:.0f}".rjust(4) + " " if sm is not None else f"{'-':>4} ", style=util_style(sm))
            t.append(f"{p.cpu_pct:.1f}".rjust(6) + " " if p.cpu_pct is not None else f"{'-':>6} ")
            t.append(f"{human_bytes(p.rss_kb * 1024):>8}  ")
            t.append(short_args(p.args))
            out.append(fit(t, w))
        if len(procs) > n:
            out.append(Text(f"  … {len(procs) - n} more", style=B))
        if not procs:
            out.append(Text("  (no processes yet)", style=B))
        return out

    def draw_lines(self, w, h):
        job, s = self.job()
        r = self.r
        if s is None or job is None or r is None:
            return [Text(" job is no longer running", style="yellow")]
        res = RES_STEPS[self.res_i]
        ncols = w - 4
        util, gmem, cpu, ram = job_series(job, s, time.time(), res, 2 * ncols)
        out = [Text(" ") + Text(short_args(r.args)[: max(10, w * 2 - 4)], style="bold")]
        meta = Text(" ", style=B)
        for k in ("WorkDir", "StdOut", "SubmitTime", "StartTime", "TimeLimit", "AllocTRES"):
            if self.info.get(k):
                meta.append(f"{k}=", style=B)
                meta.append(f"{self.info[k]}  ")
        out += [meta, Text("")]
        with s.lock:
            nproc = sum(len(ps) for ps in s.procs.values())
        nblocks = 4 if job.has_gpu else 2
        nlist = min(max(1, nproc), max(3, (h - 4) // 3))
        list_h = nlist + 2 + (1 if nproc > nlist else 0)
        hh = max(2, (h - len(out) - 1 - list_h - 1 - nblocks) // nblocks)

        def block(title, vals, style):
            return [Text("  ") + fit(x, ncols) for x in [title] + braille(vals, hh, True, style)]

        if job.has_gpu:
            t = Text(" GPU UTL (mean of the job's GPUs, incl. history) ", style="bold")
            t.append(f"{r.util:.0f}%" if r.util is not None else "-", style=util_style(r.util))
            if r.sm is not None:
                t.append(f"   %SM {r.sm:.0f}", style=B)
            out += block(t, util, lambda v: util_style(100 * v if v is not None else None))
            t = Text(" GPU MEM ", style="bold")
            t.append(f"{r.gmem / 1024:.1f} of {r.gtot / 1024:.0f} GiB" if r.gmem is not None and r.gtot else "-",
                     style="magenta")
            out += block(t, gmem, mem_style)
        t = Text(" CPU ", style="bold")
        t.append(f"{r.cores:.1f} of {job.cpus} cores" if r.cores is not None else "-", style="cyan")
        out += block(t, cpu, lambda v: "cyan")
        t = Text(" RAM ", style="bold")
        t.append(f"{r.mem:.0f}% of limit (excl. cache)" if r.mem is not None else "-", style=RAM_STYLE)
        out += block(t, ram, lambda v: RAM_STYLE)
        out.append(Text("  ") + time_axis(ncols, res))
        out.append(Text(""))
        out += self.proc_table(s, job, nlist, w)
        return out[:h]

    def on_key(self, event):
        ch = event.character
        if event.key in ("escape", "q", "enter"):
            self.app.pop_screen()
        elif ch in ("+", "="):
            self.res_i = min(len(RES_STEPS) - 1, self.res_i + 1)
            self.tick()
        elif ch in ("-", "_"):
            self.res_i = max(0, self.res_i - 1)
            self.tick()
        elif ch == "e":
            if self.r and self.r.pid:
                self.app.push_screen(EnvScreen(self.r))
        elif ch == "l":
            job, _ = self.job()
            if job:
                self.app.push_screen(LogScreen(job))
        else:
            return
        event.stop()


class ProcScreen(Screen):
    """Charts for one process (per-process view), plus its job's details."""

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
            self.res_i = min(len(RES_STEPS) - 1, self.res_i + 1)
            self.tick()
        elif ch in ("-", "_"):
            self.res_i = max(0, self.res_i - 1)
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


class ClusterPanel(Lines):
    def on_mouse_scroll_down(self, event):
        self.screen.move(+3)

    def on_mouse_scroll_up(self, event):
        self.screen.move(-3)

    def on_click(self, event):
        self.screen.click_row(event.y, event.chain == 2)


def poll_status(cm, keys, w):
    if cm.err:
        return Text(f" sinfo/squeue error: {cm.err}", style="red")
    st = Text(f" {keys}", style=B)
    right = f"sinfo+squeue every {cm.interval:g}s while open " + (f"({time.time() - cm.last:.0f}s ago) " if cm.last else "(polling…) ")
    st.append(" " * max(1, w - len(st.plain) - len(right)))
    st.append(right, style=B)
    return st


class ClusterScreen(Screen):
    """Free / queued GPUs across the cluster, one row per GPU type (n: per node)."""

    def __init__(self, cm: ClusterMonitor):
        super().__init__()
        self.cm = cm
        self.node_mode = False
        self.sel = None
        self.keys: list[str] = []
        self.n_head = 1
        self.row_off = 0

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield ClusterPanel(self.lines, id="procs")
        yield Static(id="status")

    def on_mount(self):
        self.cm.activate()
        self.set_interval(1.0, self.tick)
        self.tick()

    def on_unmount(self):
        self.cm.deactivate()

    def tick(self):
        h = Text(" sqtop gpus ", style="bold black on green")
        h.append(f"  {time.strftime('%a %b %d %H:%M:%S')}   ", style="bold")
        snap = self.cm.snap
        if snap:
            free, total = sum(g.free for g in snap.groups), sum(g.total for g in snap.groups)
            queued = sum(j.gpus * j.tasks for j in snap.pending() if j.klass == "queued")
            h.append(f"{free} of {total} GPUs free · {queued} GPUs queued", style="")
            h.append(f"   by {'node' if self.node_mode else 'GPU type'}", style=B)
        self.query_one("#header").update(h)
        self.query_one("#status").update(poll_status(
            self.cm, "Esc/q back · ↑↓ select · Enter nodes + queue for that type · n by node / by type · r refresh",
            self.size.width or 120))
        self.query_one("#procs").refresh()

    def lines(self, w, h):
        snap = self.cm.snap
        if snap is None:
            return [Text(" polling sinfo + squeue…", style="yellow")]
        head, rows, foot = render_summary(snap, self.node_mode)
        self.keys = [k for k, _ in rows]
        self.n_head = len(head)
        if self.sel not in self.keys:
            self.sel = None
        body_h = max(1, h - len(head) - len(foot))
        c = self.keys.index(self.sel) if self.sel is not None else None
        if c is not None:
            self.row_off = min(max(self.row_off, c - body_h + 1), c)
        self.row_off = max(0, min(self.row_off, len(rows) - body_h))
        out = list(head)
        for i, (k, line) in enumerate(rows[self.row_off:self.row_off + body_h]):
            if k == self.sel:
                line = line.copy()
                line.stylize("reverse")
            out.append(line)
        return [fit(x, w) for x in out + foot]

    def move(self, d):
        if not self.keys:
            return
        c = self.keys.index(self.sel) if self.sel in self.keys else None
        c = (0 if d > 0 else len(self.keys) - 1) if c is None else max(0, min(len(self.keys) - 1, c + d))
        self.sel = self.keys[c]
        self.query_one("#procs").refresh()

    def click_row(self, y, double):
        i = y - self.n_head + self.row_off
        if 0 <= i < len(self.keys):
            self.sel = self.keys[i]
            self.query_one("#procs").refresh()
            if double:
                self.open()

    def open(self):
        snap = self.cm.snap
        if snap is None or self.sel is None:
            return
        name = snap.nodes[self.sel].group if self.node_mode and self.sel in snap.nodes else self.sel
        self.app.push_screen(GroupScreen(self.cm, name))

    def on_key(self, event):
        k, ch = event.key, event.character
        if k in ("escape", "q") or ch == "g":
            self.app.pop_screen()
        elif k == "down":
            self.move(+1)
        elif k == "up":
            self.move(-1)
        elif k == "pagedown":
            self.move(+10)
        elif k == "pageup":
            self.move(-10)
        elif k == "home":
            self.move(-len(self.keys))
        elif k == "end":
            self.move(len(self.keys))
        elif k == "enter":
            self.open()
        elif ch == "n":
            self.node_mode = not self.node_mode
            self.sel, self.row_off = None, 0
            self.tick()
        elif k in ("r", "f5"):
            self.cm.kick.set()
        else:
            return
        event.stop()


class GroupScreen(TextScreen):
    """One GPU type: each node's state and running jobs, then its pending jobs."""

    def __init__(self, cm: ClusterMonitor, name: str):
        super().__init__()
        self.cm, self.gname = cm, name
        self.drawn = (None, 0)

    def on_mount(self):
        super().on_mount()
        self.set_interval(1.0, self.tick)
        self.tick()

    def tick(self):
        snap, w = self.cm.snap, self.size.width or 120
        self.query_one("#header").update(Text(f" {self.gname}: nodes + queue ", style="bold reverse green"))
        self.query_one("#status").update(poll_status(self.cm, "Esc/q back · ↑↓ PgUp PgDn scroll · r refresh", w))
        if snap is None or self.drawn == (snap.t, w):
            return
        self.drawn = (snap.t, w)
        self.query_one("#content").update(Group(*[fit(x, w - 2) for x in render_detail(snap, self.gname)]))

    def extra_key(self, event):
        if event.key in ("r", "f5"):
            self.cm.kick.set()
            return True
        return False


class SqTop(App):
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("ctrl+c", "quit", show=False, priority=True)]

    def __init__(self, m: Monitor):
        super().__init__(ansi_color=True)
        self.m = m
        self.cm = ClusterMonitor(m.a.cluster_s, m.a.user)

    def on_mount(self):
        import asyncio
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(sig, self.exit)
        self.push_screen(MainScreen(self.m))
