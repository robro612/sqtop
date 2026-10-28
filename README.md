# sqtop

An nvitop-style live view of **your SLURM jobs**: their GPUs, CPU/RAM against the allocation,
processes, logs, and history. It also has a one-shot JSON mode (`sqtop --json`) for agents and
scripts.

`squeue` shows which jobs you have; `sqtop` shows what they are doing.

<img width="3004" height="1796" alt="image" src="https://github.com/user-attachments/assets/6f947409-9e9b-470c-8c5b-090293e1115c" />


## Requirements

- SLURM 20.11+ (for `srun --overlap`), with jobs placed in **cgroup v2** (`proctrack/cgroup` +
  `task/cgroup`; sqtop reads the job's `.../job_N/step_*` cgroup for CPU/RAM and processes).
- NVIDIA GPUs with `nvidia-smi` on the compute nodes. CPU-only jobs still show CPU/RAM and processes.
- `bash` 5+ and `setpriv` (util-linux) on the nodes / login node; Python 3.10+.
- Optional, for history before sqtop started and `--json` window stats: Prometheus scraping the
  [jobstats](https://github.com/PrincetonUniversity/jobstats) GPU exporter (`nvidia_gpu_duty_cycle`).
  Point sqtop at it with `$SQTOP_PROM_URL`, or it reads `PROM_SERVER` from
  `/usr/local/jobstats/config.py`.

## Install

```bash
uv tool install git+https://github.com/robro612/sqtop   # puts `sqtop` in ~/.local/bin
uv tool install --editable .                         # or, from a clone
```

or, from the repo, `uv run sqtop`. The only dependency is `textual`.

## Interactive use

```bash
sqtop                      # live TUI
sqtop --once               # print one snapshot and exit
sqtop --squeue-s 10        # poll squeue every 10 s (default 30)
sqtop --gpu-ms 500         # sample GPUs every 0.5 s (default 1 s)
sqtop --no-history         # skip the Prometheus backfill
```

Keys (press `h` in the app for the full list):

| Key | Action |
|---|---|
| `↑` `↓` `Home` `End` `PgUp` `PgDn`, wheel, click | select a row (a job / process) |
| `Enter` | per-process metrics: charts for the process plus job details |
| `e` | process environment, fetched from the node on demand |
| `l` | tail the job's log (StdOut/StdErr; `Tab` switches) |
| `o` + letter, `,` `.`, `/` | sort: `j` job, `n` name, `r` gres, `t` time, `d` node, `g` GPU-mem, `s` %SM, `b` %GMBW, `c` %CPU, `m` %MEM; `/` flips direction |
| `a` `f` `c` | display mode: auto / full / compact |
| `+` `-` | zoom charts (seconds per braille dot) |
| `←` `→` `^` `$` | scroll the command column |
| `r` / `F5` | re-poll squeue now |
| `q` / `Ctrl-C` | quit |

Conventions:
- GPU utilization is red below 30%, yellow below 70%, green above. This is the reverse of
  nvitop, because for training jobs low utilization is the problem.
- RAM excludes page cache, which fills to the cgroup limit but can be reclaimed.
- `%MEM` is a share of the job's memory limit, not of the node.
- `TIME` is the job's elapsed time, coloured as the time limit approaches.
- The `GPU` and `PID` columns appear only when some job has several GPUs or several processes.

## Agent / script mode

```bash
sqtop --json [--window 30m] [--jobs 123456,123457_2] [--log-lines 5] [--max-procs 15] [--no-procs] [--pretty]
```

This prints **one JSON document** and exits. A call takes about 3 s and leaves nothing running.
The output is raw data with no verdicts: deciding what "idle" or "stuck" means is left to the
caller. Keys with no value are omitted. For five jobs the output is about 12 KB.

```
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
```

- `procs` lists the job's own processes, without SLURM internals or the batch shell (see
  `batch`). Entries with identical command lines, such as dataloader or vLLM workers, are merged:
  `n` is the count, `cpu_pct` and `gpu_mem_gb` are summed, and `rss_gb` is the largest member's,
  since workers share memory. `ports` lists listening TCP ports, so a vLLM server shows its
  `--port`.
- `window` stats come from the cluster's Prometheus at 1-minute resolution. `covered_s` says how
  much of the window had data, for example when the job is younger than the window.
- `cpu_pct` and `cpu.used` are sampled over 1 s inside the job.
- `--no-procs` skips the in-job probe. You still get the SLURM fields, the log and GPU values
  from Prometheus (at most 1 minute old), but no processes, ports or CPU/RAM.

Example: "which of my jobs are vLLM servers, and how busy were their GPUs over the last hour?"

```bash
sqtop --json --window 1h | jq '.jobs[] | select(any(.procs[]?; .cmd | test("vllm")))
  | {id, name, ports: [.procs[].ports // empty] | add, util: [.gpus[].window.mean]}'
```

## How it works

- `squeue` (every `--squeue-s`) lists your running and pending jobs.
- For each running job, sqtop starts **one long-lived step inside the job**:
  `srun --overlap --jobid=<id>`. It shares the job's existing allocation and requests no new
  resources. Inside it, `nvidia-smi` (loop mode), `nvidia-smi pmon` and a small bash loop stream
  GPU stats, per-process %SM/%GMBW, the job's cgroup CPU/RAM, and its processes. Because the
  step runs inside the job's resource group, it sees only that job's GPUs.
- History from before sqtop started is backfilled from the cluster's Prometheus (the jobstats
  exporter, at 1-minute resolution), so charts cover the job's whole run. The Prometheus URL comes
  from `$SQTOP_PROM_URL`, or else from `PROM_SERVER` in the cluster's jobstats config
  (`/usr/local/jobstats/config.py`). Without either, sqtop runs without history, and `--json` omits
  `window` stats and says so in `errors`.
- `--json` uses the same pieces as a one-shot: `squeue` + `scontrol`, a few Prometheus window
  queries, and one short `srun --overlap` per job, all in parallel.

Safety:
- **Read-only.** sqtop never signals processes or cancels jobs.
- Stopping a monitoring step (SIGTERM to its `srun`) cancels **only that step**, never the job.
  This was tested against live jobs.
- `srun` runs under `setpriv --pdeathsig TERM`, so if sqtop dies, even from `kill -9`, the
  kernel stops its sruns. Their steps were gone within 2 s in testing.
- When a job ends, SLURM ends its step. sqtop sees the stream close, re-polls `squeue`
  immediately, and drops the row in about 1.5 s. A live stream is stopped only after its job is
  missing from two consecutive polls, because a single `squeue` poll can come back incomplete
  while the controller is busy.
- Step names (`sqtop`, `sqtop-env`, `sqtop-json`) make them easy to spot in `squeue -s` and `sacct`.

Cost, measured with 6 jobs over 60 s:

| Where | What | Cost |
|---|---|---|
| Login node | TUI process | ~6% of one core, ~36 MB |
| Login node | `srun` clients | ~0% CPU, ~6 MB each |
| Compute nodes | one monitoring step per job | ~10% of one core and ~33 MB each, inside the job's own allocation |
| SLURM / Prometheus | `squeue` every 30 s; 2 history queries per job at startup | negligible |

## Code layout

| Module | Contents |
|---|---|
| `src/sqtop/core.py` | job listing, the in-job stream (`Stream`, `Monitor`), Prometheus backfill, log tailing |
| `src/sqtop/render.py` | device table, host panel, braille charts, jobs & processes table |
| `src/sqtop/tui.py` | Textual app and screens |
| `src/sqtop/agent.py` | `--json` mode |
| `src/sqtop/cli.py` | argument parsing, `--once`, entry point |

## Development

```bash
uv run python tests/smoke.py        # needs >=1 running job: drives every screen, checks --json, checks for leftover steps
uvx ruff check src tests
```

Known gaps:
- There is no per-process history from before sqtop started (Prometheus has none).
- There is no process tree view.
- Multi-node jobs are handled (one stream task per node, rows labelled by host) but untested.

## License

MIT. See [LICENSE](LICENSE).
