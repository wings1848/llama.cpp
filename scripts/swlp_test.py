#!/usr/bin/env python3
"""SWLP test framework -- YAML-driven, with CLI overrides, --bench-args passthrough,
optional GPU/memory/CPU monitoring, test matrix expansion via Cartesian product,
and post-hoc analysis.

Usage:
  python scripts/swlp_test.py
  python scripts/swlp_test.py --phase quick
  python scripts/swlp_test.py --config my_config.yaml --model-type moe --bench-args "--swlp-alpha 0.5"
  python scripts/swlp_test.py --dry-run
  python scripts/swlp_test.py ./results           # analyze saved results

GPU/mem/CPU monitoring is enabled by default when tools are available
(nvidia-smi for GPU, psutil for mem/CPU). Disable with --no-gpu-mon etc.
"""

import argparse, csv, itertools, json, os, shlex, shutil, subprocess, sys
import tempfile, threading, time, traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from string import Formatter

try:
    import yaml
except ImportError:
    print("ERROR: install PyYAML: pip install pyyaml", file=sys.stderr); sys.exit(1)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(s, **kw):
    class SD(dict):
        def __missing__(self, k): return '{' + k + '}'
    return Formatter().vformat(s, (), SD(kw))

def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        print(f"ERROR: empty config: {path}", file=sys.stderr); sys.exit(1)
    return cfg

def _resolve_model_path(mcfg, search_paths):
    files = [mcfg.get("file", "")]
    if mcfg.get("alt_file"):
        files.append(mcfg["alt_file"])
    for fn in files:
        if not fn: continue
        if os.path.isabs(fn):
            p = Path(fn)
            if p.exists() and p.stat().st_size > 1000:
                return str(p.resolve())
        for sp in search_paths:
            p = Path(sp) / fn
            if p.exists() and p.stat().st_size > 1000:
                return str(p.resolve())
        p = Path(fn)
        if p.exists() and p.stat().st_size > 1000:
            return str(p.resolve())
    return None

# ---------------------------------------------------------------------------
# test expansion
# ---------------------------------------------------------------------------

def _resolve_val(v, mcfg):
    if isinstance(v, str):
        return {"auto": -1, "off": 0, "all": mcfg.get("n_layers", 0)}.get(v,
               mcfg.get(v[1:], v) if v.startswith("$") else v)
    return v

def _expand_tests(entries, mcfg):
    """Cartesian product for list-valued params.  Returns list of test dicts."""
    out, aid = [], 1
    for e in entries:
        raw = e.get("params", {})
        fixed, axes = {}, {}
        for k, v in raw.items():
            rv = _resolve_val(v, mcfg)
            if isinstance(v, list):
                axes[k] = [_resolve_val(x, mcfg) for x in v]
            else:
                fixed[k] = rv
        base = dict(tags=list(e.get("tags",[])),
                    run_modes=list(e.get("run_modes",["pp","gen"])),
                    model_types=e.get("model_types",None))
        if not axes:
            o = dict(base, id=e.get("id") or aid, params=fixed,
                     desc=_fmt(e.get("desc",""),**fixed))
            out.append(o); aid += 1
        else:
            for combo in itertools.product(*axes.values()):
                p, ckw = dict(fixed), {}
                for i, k in enumerate(axes):
                    p[k] = combo[i]; ckw[k] = combo[i]
                o = dict(base, id=e.get("id") or aid, params=p,
                         desc=_fmt(e.get("desc",""),**ckw))
                out.append(o); aid += 1
    return out

def _filter_tests(tests, phases=None, ids=None, mtype=None):
    r = []
    for t in tests:
        if ids is not None and t["id"] not in ids: continue
        if mtype is not None:
            al = t.get("model_types")
            if al is not None and mtype not in al: continue
        if phases is not None and "all" not in phases:
            if not any(p in t.get("tags", []) for p in phases): continue
        r.append(t)
    return r

# ---------------------------------------------------------------------------
# monitoring  (optional polling threads)
# ---------------------------------------------------------------------------

class _Monitor:
    """Base: polling thread, start/stop/summary."""
    def __init__(self, interval_ms=500):
        self.interval = interval_ms / 1000.0
        self.samples, self._stop, self._th = [], False, None
    def poll(self): raise NotImplementedError
    def start(self):
        self._stop, self.samples = False, []
        self._th = threading.Thread(target=self._run, daemon=True); self._th.start()
    def stop(self):
        self._stop = True
        if self._th: self._th.join(timeout=3)
    def _run(self):
        while not self._stop:
            try: self.poll()
            except Exception: pass
            time.sleep(self.interval)
    def _vals(self, key):
        return [s.get(key, 0) for s in self.samples if s.get(key) is not None and s.get(key, 0) > 0]
    @staticmethod
    def _avg(lst): return sum(lst) / len(lst) if lst else -1

class _GpuMonitor(_Monitor):
    def poll(self):
        s = {"ts": time.time()}
        r = subprocess.run(["nvidia-smi",
            "--query-gpu=memory.used,memory.free,temperature.gpu,"
            "utilization.gpu,utilization.memory,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        p = [x.strip() for x in r.stdout.strip().split(",")]
        if len(p) >= 7:
            for k, i in [("gpu_vram_used_mb",0),("gpu_vram_free_mb",1),("gpu_temp_c",2),
                         ("gpu_util_pct",3),("gpu_mem_util_pct",4)]:
                s[k] = int(p[i])
        self.samples.append(s)
    def summary(self):
        v, t, u = self._vals("gpu_vram_used_mb"), self._vals("gpu_temp_c"), self._vals("gpu_util_pct")
        fm = self._vals("gpu_vram_free_mb")
        return {"gpu_vram_peak_mb": max(v) if v else -1, "gpu_vram_avg_mb": self._avg(v),
                "gpu_vram_free_min_mb": min(fm) if fm else -1,
                "gpu_temp_peak_c": max(t) if t else -1, "gpu_temp_avg_c": self._avg(t),
                "gpu_util_avg_pct": self._avg(u)}

class _MemMonitor(_Monitor):
    def __init__(self, interval_ms=500):
        super().__init__(interval_ms); self._proc = None
    def track_pid(self, pid):
        if HAS_PSUTIL:
            try: self._proc = psutil.Process(pid)
            except Exception: self._proc = None
    def poll(self):
        s = {"ts": time.time()}
        v = psutil.virtual_memory()
        s.update(mem_ram_used_mb=v.used/1048576, mem_ram_avail_mb=v.available/1048576)
        if self._proc:
            try:
                mi = self._proc.memory_info()
                s["mem_proc_rss_mb"] = mi.rss / 1048576
            except Exception: pass
        self.samples.append(s)
    def summary(self):
        v = self._vals("mem_ram_used_mb")
        r = {"mem_ram_peak_mb": max(v) if v else -1, "mem_ram_avg_mb": self._avg(v)}
        pv = self._vals("mem_proc_rss_mb")
        if pv: r["mem_proc_rss_peak_mb"] = max(pv)
        return r

class _CpuMonitor(_Monitor):
    def poll(self):
        s = {"ts": time.time(), "cpu_util_pct": psutil.cpu_percent(interval=0)}
        pc = psutil.cpu_percent(interval=0, percpu=True)
        if pc: s.update(cpu_max_core_pct=max(pc), cpu_avg_core_pct=sum(pc)/len(pc))
        self.samples.append(s)
    def summary(self):
        u = self._vals("cpu_util_pct")
        return {"cpu_util_avg_pct": self._avg(u), "cpu_util_max_pct": max(u) if u else -1}

class _MonMgr:
    def __init__(self, gpu=False, mem=False, cpu=False):
        self.mons = []
        if gpu and shutil.which("nvidia-smi"): self.mons.append(_GpuMonitor())
        if mem and HAS_PSUTIL: self.mons.append(_MemMonitor())
        if cpu and HAS_PSUTIL: self.mons.append(_CpuMonitor())
    def start(self, pid=None):
        for m in self.mons:
            if hasattr(m, 'track_pid') and pid is not None:
                try: m.track_pid(pid)
                except Exception: pass
            m.start()
    def stop(self):
        for m in self.mons: m.stop()
    def summary(self):
        r = {}
        for m in self.mons: r.update(m.summary())
        return r

# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_SWLP_FLAGS = [  # (cli_flag, param_key, default)
    ("--ngl", "ngl", 0), ("--window", "window", 0), ("--prefetch", "prefetch", 1),
    ("--expert-cache", "expert_cache", 0),
    ("--ctx", "ctx", 4096), ("--threads", "threads", 4), ("--batch", "batch", 2048),
    ("--warmup", "warmup", 1), ("--iters", "iters", 5),
]
_SWLP_BOOL = [("--expert-prefetch","expert_prefetch"), ("--adaptive","adaptive"),
              ("--pinned","pinned"), ("--swlp-verbose","swlp_verbose")]

def _build_cmd(bench, model_path, params, defaults, extra, mode, out):
    cmd = [str(bench), model_path]
    for flag, key, dfl in _SWLP_FLAGS:
        cmd += [flag, str(params.get(key, defaults.get(key, dfl)))]
    for flag, key in _SWLP_BOOL:
        if params.get(key): cmd.append(flag)
    a = params.get("swlp_alpha")
    if a is not None: cmd += ["--swlp-alpha", str(a)]
    ai = params.get("swlp_adapt_interval")
    if ai is not None: cmd += ["--swlp-adapt-interval", str(ai)]
    pt = params.get("prompt_tokens", defaults.get("prompt_tokens", 256))
    txt = params.get("prompt")
    if txt: cmd += ["--prompt", txt]
    else: cmd += ["--prompt-tokens", str(pt)]
    if mode == "gen":
        cmd += ["--gen", str(params.get("gen_tokens", defaults.get("gen_tokens", 128)))]
    cmd += ["--output", out]
    if extra: cmd.extend(extra)
    return cmd

def _run_one(model_path, params, bench, defaults, extra, mode, mon, retries, timeout):
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, f"r_{os.getpid()}_{threading.get_ident()}.json")
    for attempt in range(retries):
        cmd = _build_cmd(bench, model_path, params, defaults, extra, mode, out)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            mon.start(proc.pid if hasattr(mon,'start') else None)
            try: so, se = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill(); so, se = proc.communicate(); mon.stop()
                if attempt < retries-1: time.sleep(1); continue
                return {"error": True, "msg": f"Timeout {timeout}s", "stderr_tail": se[-500:]}
            mon.stop()
            if proc.returncode != 0:
                if attempt < retries-1: time.sleep(1); continue
                return {"error": True, "rc": proc.returncode, "stderr_tail": se[-500:],
                        "monitoring": mon.summary()}
            if os.path.exists(out) and os.path.getsize(out) > 10:
                with open(out) as f: data = json.load(f)
                os.remove(out)
                ms = mon.summary(); data["monitoring"] = ms; data.update(ms)
                return {"error": False, "data": data}
            if attempt < retries-1: time.sleep(1); continue
            return {"error": True, "msg": "No output", "stderr_tail": se[-300:] if se else "",
                    "monitoring": mon.summary()}
        except Exception as e:
            mon.stop()
            if attempt < retries-1: time.sleep(1); continue
            return {"error": True, "msg": str(e)}
    return {"error": True, "msg": "All retries exhausted"}

def _run_model(mid, mcfg, tests, out_dir, bench, defaults, extra, mode_f, mon_cfg, retries, timeout, verbose):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = out_dir / mid; d.mkdir(parents=True, exist_ok=True)
    mp = _resolve_model_path(mcfg, defaults.get("model_search_paths", []))
    if not mp:
        print(f"  SKIP {mid}: model not found ({mcfg.get('file','?')})")
        return []
    modes = ["pp","gen"] if mode_f == "both" else [mode_f]
    total = len(tests) * len(modes)
    all_r = []
    print(f"\n{'='*55}\n{mid} {mcfg.get('name',mid)} ({mcfg.get('n_layers','?')}L) / {mcfg.get('type','?')} / {len(tests)} configs x {len(modes)} modes = {total}\n{'='*55}")
    idx = 0
    for t in tests:
        for rm in modes:
            idx += 1
            desc = f"{t['desc']} [{rm}]"
            print(f"  [{idx}/{total}] #{t['id']} {desc:<45}", end="", flush=True)
            mon = _MonMgr(**mon_cfg)
            r = _run_one(mp, t["params"], bench, defaults, extra, rm, mon, retries, timeout)
            rec = dict(model_id=mid, model_name=mcfg.get("name",mid),
                       model_type=mcfg.get("type",""), n_layers=mcfg.get("n_layers",0),
                       quant=mcfg.get("quant",""), test_id=t["id"], desc=t["desc"],
                       mode=rm, timestamp=datetime.now().isoformat(), **t["params"])
            if r.get("error"):
                rec.update(error=True, error_msg=(r.get("msg") or r.get("stderr_tail","unknown"))[:200],
                           exit_code=r.get("rc",-1))
                if r.get("monitoring"): rec.update(r["monitoring"])
                print(f" FAIL: {rec['error_msg'][:60]}")
            else:
                d2 = r["data"]; rec["error"] = False
                for k in ("pp_mean_ms","pp_min_ms","pp_max_ms","pp_p50_ms","pp_p95_ms",
                          "pp_std_ms","pp_tps","pp_iter_ms","gen_mean_ms","gen_min_ms",
                          "gen_max_ms","gen_p50_ms","gen_p95_ms","gen_std_ms","gen_tps","gen_iter_ms"):
                    if k in d2: rec[k] = d2[k]
                if d2.get("monitoring"): rec.update(d2["monitoring"])
                print(f" {'PP' if rm=='pp' else 'Gen'}={d2.get('pp_tps' if rm=='pp' else 'gen_tps',0):.0f}t/s" +
                      (f" p95={d2.get('gen_p95_ms',0):.1f}ms" if rm=='gen' else ""))
            all_r.append(rec)
    jp = d / f"results_{ts}.json"
    with open(jp,"w") as f: json.dump(all_r, f, indent=2)
    ok = sum(1 for x in all_r if not x.get("error"))
    if all_r:
        cp = d / f"results_{ts}.csv"
        with open(cp,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_r[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(all_r)
    print(f"  => {ok}/{len(all_r)} ok | {jp}")
    return all_r

# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _analyze(all_r):
    ok = [r for r in all_r if not r.get("error")]
    if not ok: return print("No successful results.")
    grp = defaultdict(list)
    for r in ok: grp[r["model_id"]].append(r)
    print(f"\n{'='*105}\nRESULTS BY MODEL\n{'='*105}")
    for mid in sorted(grp):
        m = sorted(grp[mid], key=lambda x: (x.get("mode",""), x["test_id"]))
        pps = [r for r in m if r.get("mode")=="pp"]
        gns = [r for r in m if r.get("mode")=="gen"]
        print(f"\n{mid} ({m[0].get('model_name',mid)}) / {m[0].get('model_type','')} / {len(m)} results")
        if pps:
            print(f"  {'ID':>4} {'Description':<34} {'t/s':>7} {'ms':>7} {'VRAM':>7} {'Temp':>5} {'RSS':>8} {'CPU%':>6}")
            print(f"  {'-'*4} {'-'*34} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*8} {'-'*6}")
            for r in pps:
                tps = r.get("pp_tps",0); ms = r.get("pp_mean_ms",0)
                vram = r.get("gpu_vram_peak_mb",-1); temp = r.get("gpu_temp_peak_c",-1)
                rss = r.get("mem_rss_mb",-1); cpu = r.get("cpu_util_avg_pct",-1)
                print(f"  {r['test_id']:>4} {r['desc']:<34} {tps:>7.0f} {ms:>7.1f}"
                      + (f" {vram:>6.0f}M" if vram>0 else "       ")
                      + (f" {temp:>3.0f}C" if temp>0 else "    ")
                      + (f" {rss:>7.0f}M" if rss>0 else "        ")
                      + (f" {cpu:>5.1f}" if cpu>=0 else "     "))
        if gns:
            if pps: print()
            print(f"  {'ID':>4} {'Description':<34} {'t/s':>7} {'p95':>7} {'VRAM':>7} {'Temp':>5} {'RSS':>8} {'CPU%':>6}")
            print(f"  {'-'*4} {'-'*34} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*8} {'-'*6}")
            for r in gns:
                tps = r.get("gen_tps",0); p95 = r.get("gen_p95_ms",0)
                vram = r.get("gpu_vram_peak_mb",-1); temp = r.get("gpu_temp_peak_c",-1)
                rss = r.get("mem_rss_mb",-1); cpu = r.get("cpu_util_avg_pct",-1)
                print(f"  {r['test_id']:>4} {r['desc']:<34} {tps:>7.0f} {p95:>7.1f}"
                      + (f" {vram:>6.0f}M" if vram>0 else "       ")
                      + (f" {temp:>3.0f}C" if temp>0 else "    ")
                      + (f" {rss:>7.0f}M" if rss>0 else "        ")
                      + (f" {cpu:>5.1f}" if cpu>=0 else "     "))
    print()

def _cross_table(all_r):
    ok = [r for r in all_r if not r.get("error")]
    if not ok: return
    print(f"\n{'='*85}\n{'CROSS-MODEL COMPARISON':^85}\n{'='*85}")
    has_pp = any(r.get("mode")=="pp" for r in ok)
    for mode in (["gen","pp"] if has_pp else ["pp"]):
        by_model = defaultdict(list)
        for r in ok:
            if r.get("mode")!=mode: continue
            by_model[r["model_id"]].append(r)
        if not by_model: continue
        grp = []
        for mid, rs in by_model.items():
            gb = [r for r in rs if r.get("window")==0 and r.get("ngl",0)>0]
            cb = [r for r in rs if r.get("window")==0 and r.get("ngl",0)==0]
            if not rs: continue
            tps_key = "gen_tps" if mode=="gen" else "pp_tps"
            p95_key = "gen_p95_ms" if mode=="gen" else "pp_mean_ms"
            base = (gb[0] if gb else cb[0]) if (gb or cb) else None
            base_tps = base.get(tps_key, 1) if base else 1
            swlp = [r for r in rs if r.get("window",0)!=0]  # exclude baselines
            if not swlp: continue
            best = max(swlp, key=lambda r: r.get(tps_key,0))
            pct = (best.get(tps_key,0)/base_tps - 1)*100 if base_tps else 0
            ws = "auto" if best.get("window")==-1 else str(best.get("window","?"))
            cp = f"W={ws} PF={best.get('prefetch','?')}"
            if best.get("adaptive"): cp += " A"
            if best.get("pinned"): cp += " P"
            if best.get("expert_cache",0)>0: cp += f" EC={best['expert_cache']}"
            grp.append((mid, best.get("model_name",mid), base_tps, best.get(tps_key,0), pct,
                        best.get("gpu_vram_peak_mb",-1), cp,
                        best.get(p95_key,0) if mode=="gen" else 0,
                        best.get("mem_rss_mb",-1), best.get("cpu_util_avg_pct",-1)))
        if not grp: continue
        hdr = f"  [{'Gen' if mode=='gen' else 'PP'}]"
        if mode=="gen":
            print(f"\n{hdr:>3}  {'Model':<18} {'GPU base':>8}  {'Best SWLP':>8}  {'Delta':>7}  {'VRAM':>8}  {'RSS':>8}  {'p95':>7}  Config")
        else:
            print(f"\n{hdr:>3}  {'Model':<18} {'CPU base':>8}  {'Best SWLP':>8}  {'Delta':>7}  {'VRAM':>8}  {'RSS':>8}  Config")
        print(f"  {'-'*95}")
        for g in sorted(grp, key=lambda x: -x[4]):
            _, mn, bt, bst, pct, vram, cp, p95, rss, cpu = g
            if vram>0: vram_s = f"{vram:>6.0f}M"
            else: vram_s = "    - "
            if rss>0: rss_s = f"{rss:>7.0f}M"
            else: rss_s = "       "
            if mode=="gen":
                print(f"  {mn:<18} {bt:>8.0f}  {bst:>8.0f}  {pct:>+6.0f}%  {vram_s:>8}  {rss_s:>8}  {p95:>7.1f}  {cp}")
            else:
                print(f"  {mn:<18} {bt:>8.0f}  {bst:>8.0f}  {pct:>+6.0f}%  {vram_s:>8}  {rss_s:>8}  {cp}")
    print(f"{'='*95}")

def _find_baseline(gr, ngl, mode):
    """Find the matching no-SWLP (window=0) baseline for a given ngl level.
    Only returns a baseline with the exact same ngl. This ensures anomalies
    compare like-for-like (same GPU offload, with/without SWLP)."""
    exact = [r for r in gr if r.get("window")==0 and r.get("ngl",-1)==ngl and not r.get("error")]
    if exact:
        field = "gen_tps" if mode=="gen" else "pp_tps"
        return exact[0].get(field, 0)
    return 0

def _scan_bugs(all_r):
    bugs = []
    for r in all_r:
        if r.get("error"):
            bugs.append({k: r.get(k) for k in ("model_id","test_id","desc","mode","window","expert_cache")})
            bugs[-1]["error_msg"] = str(r.get("error_msg",""))[:200]
            bugs[-1]["exit_code"] = r.get("exit_code",-1)
    grp = defaultdict(list)
    for r in all_r:
        if not r.get("error"): grp[(r["model_id"],r.get("mode",""))].append(r)
    for (mid,mode), gr in grp.items():
        for r in gr:
            if r.get("window")==0: continue
            cur = r.get("gen_tps" if mode=="gen" else "pp_tps",0)
            ngl = r.get("ngl",0)
            ba = _find_baseline(gr, ngl, mode)
            if cur<=0: continue
            if ba <= 0:
                # No same-ngl baseline exists; skip (partial GPU SWLP vs no-baseline is expected)
                continue
            ratio = cur/ba
            desc_ctx = f"vs ngl={ngl} no-SWLP"
            if ngl == 0:
                # CPU-only: SWLP should be at least half of CPU baseline
                if ratio < 0.5:
                    bugs.append(dict(model_id=mid, test_id=r["test_id"], desc=r.get("desc","?"),
                                mode=mode, window=r.get("window",0), expert_cache=r.get("expert_cache",0),
                                error_msg=f"Anomaly ({desc_ctx}): {cur:.0f}t/s = {ratio:.1f}x base({ba:.0f})", exit_code=0))
            else:
                # Partial/full GPU: anomaly if dramatically different from same-ngl no-SWLP
                if ratio > 3 or ratio < 0.3:
                    bugs.append(dict(model_id=mid, test_id=r["test_id"], desc=r.get("desc","?"),
                                mode=mode, window=r.get("window",0), expert_cache=r.get("expert_cache",0),
                                error_msg=f"Anomaly ({desc_ctx}): {cur:.0f}t/s = {ratio:.1f}x base({ba:.0f})", exit_code=0))
    return bugs

def _load_results(dir):
    seen, r = set(), []
    ap = Path(dir)/"all_results.json"
    if ap.exists():
        for x in json.load(open(ap)):
            k = (x.get("model_id"),x.get("test_id"),x.get("mode"))
            if k not in seen: seen.add(k); r.append(x)
    for jf in sorted(Path(dir).rglob("results_*.json")):
        for x in json.load(open(jf)):
            k = (x.get("model_id"),x.get("test_id"),x.get("mode"))
            if k not in seen: seen.add(k); r.append(x)
    return r

def run_analysis(dir):
    inp = Path(dir)
    if not inp.exists(): print(f"ERROR: {inp} not found", file=sys.stderr); sys.exit(1)
    r = _load_results(inp)
    if not r: return print("No results found.")
    print(f"Loaded {len(r)} results"); _analyze(r); _cross_table(r)
    bugs = _scan_bugs(r)
    if bugs:
        print(f"\nBUG REPORT: {len(bugs)} issues\n{'='*55}")
        for b in bugs: print(f"  {b['model_id']} #{b['test_id']} {b['desc']} [{b['mode']}]\n    {b['error_msg'][:120]}")
    else: print("\nNo bugs or anomalies found.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser():
    p = argparse.ArgumentParser(description="SWLP test framework (YAML-driven)")
    p.add_argument("--config", help="YAML config (default: scripts/swlp_test_config.yaml)")
    p.add_argument("--models", nargs="+", help="Model IDs (default: all)")
    p.add_argument("--model-type", choices=["dense","moe"], help="Filter by model type")
    p.add_argument("--phase", nargs="+", default=["all"], help="Test tags to run (OR, default: all)")
    p.add_argument("--test-ids", type=int, nargs="+", help="Specific test IDs")
    p.add_argument("--bench-path", help="Bench executable path")
    p.add_argument("--bench-args", help="Extra bench args (quote the whole string)")
    p.add_argument("--ctx", type=int); p.add_argument("--threads", type=int); p.add_argument("--batch", type=int)
    p.add_argument("--prompt-tokens", type=int); p.add_argument("--gen-tokens", type=int)
    p.add_argument("--warmup", type=int); p.add_argument("--iters", type=int); p.add_argument("--retries", type=int)
    p.add_argument("--mode", choices=["pp","gen","both"], default="both")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--gpu-mon", action="store_true", default=None)
    p.add_argument("--no-gpu-mon", action="store_true", default=None)
    p.add_argument("--mem-mon", action="store_true", default=None)
    p.add_argument("--no-mem-mon", action="store_true", default=None)
    p.add_argument("--cpu-mon", action="store_true", default=None)
    p.add_argument("--no-cpu-mon", action="store_true", default=None)
    p.add_argument("results_dir", nargs="?",
                    help="Results dir to analyze (omit to run tests)")
    p.add_argument("--output", help="Output dir (default: from config)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Preview test plan")
    p.add_argument("--list-models", action="store_true", help="List models")
    return p

def _mon_cfg(cfg_mon, args):
    r = {"gpu": True, "mem": True, "cpu": True}
    if isinstance(cfg_mon, dict): r.update({k: bool(v) for k,v in cfg_mon.items() if k in r})
    if args.gpu_mon: r["gpu"] = True
    if args.no_gpu_mon: r["gpu"] = False
    if args.mem_mon: r["mem"] = True
    if args.no_mem_mon: r["mem"] = False
    if args.cpu_mon: r["cpu"] = True
    if args.no_cpu_mon: r["cpu"] = False
    if (r["mem"] or r["cpu"]) and not HAS_PSUTIL:
        print("WARNING: psutil not available. Install: pip install psutil", file=sys.stderr)
        if r["mem"]: r["mem"] = False
        if r["cpu"]: r["cpu"] = False
    if r["gpu"] and not shutil.which("nvidia-smi"):
        print("WARNING: nvidia-smi not found. GPU monitoring disabled.", file=sys.stderr)
        r["gpu"] = False
    return r

def main():
    args = _parser().parse_args()
    # positional argument: if it's a dir with results, analyze; otherwise use as output
    if args.results_dir:
        d = Path(args.results_dir)
        if d.exists() and d.is_dir() and (list(d.rglob("results_*.json")) or (d/"all_results.json").exists()):
            return run_analysis(d)
        else:
            # treat as output directory
            args.output = args.results_dir

    # locate config
    cp = args.config or str(SCRIPT_DIR / "swlp_test_config.yaml")
    if not os.path.exists(cp):
        print(f"ERROR: config not found: {cp}\n  Use --config <path>", file=sys.stderr); sys.exit(1)
    cfg = _load_config(cp)

    # merge config + CLI
    bc = cfg.get("bench", {})
    bench = Path(args.bench_path or bc.get("path",""))
    if not bench.is_absolute(): bench = PROJECT_ROOT / bench
    defaults = bc.get("defaults", {}).copy()
    extra = list(bc.get("extra_args", []))
    if args.bench_args: extra.extend(shlex.split(args.bench_args))
    for cli_k, cfg_k in [("ctx","ctx"),("threads","threads"),("batch","batch"),
                          ("prompt_tokens","prompt_tokens"),("gen_tokens","gen_tokens"),
                          ("warmup","warmup"),("iters","iters"),("retries","max_retries")]:
        v = getattr(args, cli_k, None)
        if v is not None: defaults[cfg_k] = v
    defaults["model_search_paths"] = cfg.get("model_search_paths", [])
    mon_cfg = _mon_cfg(cfg.get("monitoring", {}), args)
    out_dir = Path(args.output or cfg.get("output_dir", "./results"))

    all_models = cfg.get("models", {})
    if not all_models:
        print("ERROR: no models in config", file=sys.stderr); sys.exit(1)
    if args.list_models:
        for mid, mc in all_models.items():
            p = _resolve_model_path(mc, defaults["model_search_paths"])
            print(f"  {mid}: {mc.get('name',mid)} ({mc.get('n_layers','?')}L, {mc.get('type','?')}) [{'OK' if p else 'MISSING'}]")
        return

    mids = args.models or list(all_models.keys())
    if args.model_type:
        mids = [m for m in mids if all_models.get(m,{}).get("type") == args.model_type]
    if not bench.exists() and not args.dry_run:
        print(f"ERROR: bench not found: {bench}", file=sys.stderr); sys.exit(1)

    # expand & filter tests per model
    entries = cfg.get("tests", [])
    if not entries:
        print("ERROR: no tests in config", file=sys.stderr); sys.exit(1)
    plan = []
    for mid in mids:
        if mid not in all_models: continue
        mc = all_models[mid]
        expanded = _expand_tests(entries, mc)
        filtered = _filter_tests(expanded, phases=args.phase, ids=args.test_ids, mtype=mc.get("type"))
        if filtered: plan.append((mid, mc, filtered))

    if not plan:
        print("No tests to run.", file=sys.stderr); sys.exit(0 if args.dry_run else 1)

    # dry-run
    if args.dry_run:
        total = 0
        for mid, mc, tests in plan:
            nmodes = 2 if args.mode == "both" else 1
            n = len(tests) * nmodes; total += n
            print(f"\n{mid} ({mc.get('name',mid)}): {len(tests)} tests x {nmodes} modes = {n}")
            for t in tests:
                p = t["params"]; ws = "auto" if p.get("window")==-1 else str(p.get("window",0))
                ec = p.get("expert_cache",0)
                print(f"  #{t['id']:3d}: ngl={p.get('ngl',0):3d} W={ws:>4s} PF={p.get('prefetch','?')} "
                      f"Adaptive={p.get('adaptive',0)} Pinned={p.get('pinned',0)}" +
                      (f" EC={ec}" if ec else "") + f" | {t['desc']}")
        print(f"\nTotal: {total} runs across {len(plan)} models.")
        print(f"Monitor: GPU={mon_cfg['gpu']} Mem={mon_cfg['mem']} CPU={mon_cfg['cpu']}")
        print(f"Bench: {bench.resolve()}\nOutput: {out_dir.resolve()}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    all_r, all_bugs = [], []
    for mid, mc, tests in plan:
        r = _run_model(mid, mc, tests, out_dir, bench, defaults, extra, args.mode,
                       mon_cfg, defaults.get("max_retries",2), args.timeout, args.verbose)
        all_r.extend(r)
        if r: all_bugs.extend(_scan_bugs(r))

    _analyze(all_r); _cross_table(all_r)
    if all_bugs:
        print(f"\nBUG REPORT: {len(all_bugs)} issues\n{'='*55}")
        for b in all_bugs: print(f"  {b['model_id']} #{b['test_id']} {b['desc']} [{b['mode']}]\n    {b['error_msg'][:120]}")
        with open(out_dir/"bugs.json","w") as f: json.dump(all_bugs, f, indent=2)
        print(f"Saved: {out_dir}/bugs.json")
    else: print("\nNo bugs or anomalies found.")
    with open(out_dir/"all_results.json","w") as f: json.dump(all_r, f, indent=2)
    ok = sum(1 for r in all_r if not r.get("error"))
    print(f"Saved: {out_dir}/all_results.json\nSummary: {ok}/{len(all_r)} passed ({len(all_r)-ok} failed)")

if __name__ == "__main__":
    main()
