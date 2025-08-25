#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates an offline HTML scorecard for lightweight probes.
Reads JSON result files from: <RUN>/probes/results/*.json
Writes: <RUN>/probes/scorecard.html (auto-refresh)
Usage:
  python3 scripts/generate_probes_scorecard.py --run runs/phase2_mps_real --watch 15
"""
import argparse, glob, json, os, statistics, time, html
from datetime import datetime, timezone

def read_results(results_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            obj = json.load(open(fp, "r", encoding="utf-8"))
            data = obj.get("results") if isinstance(obj, dict) else obj
            if not isinstance(data, list): 
                continue
            for r in data:
                if not isinstance(r, dict): 
                    continue
                task = r.get("task"); acc = r.get("acc"); step = r.get("step"); ts = r.get("timestamp")
                if task is None or acc is None: 
                    continue
                rows.append({
                    "task": str(task),
                    "acc": float(acc),
                    "step": int(step or 0),
                    "ts": float(ts or time.time())
                })
        except Exception:
            # ignore malformed/partial writes
            continue
    return rows

def pct(v): return f"{100.0*float(v):.2f}"

def quantile(sorted_vals, q):
    if not sorted_vals: return float("nan")
    if q<=0: return sorted_vals[0]
    if q>=1: return sorted_vals[-1]
    i = q*(len(sorted_vals)-1)
    lo, hi = int(i), min(int(i)+1, len(sorted_vals)-1)
    frac = i - lo
    return sorted_vals[lo]*(1-frac) + sorted_vals[hi]*frac

def make_summary(rows):
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    summary = {"tasks": [], "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for task, arr in sorted(by_task.items()):
        accs = sorted([x["acc"] for x in arr])
        mean = statistics.fmean(accs) if accs else float("nan")
        p50  = quantile(accs, 0.5)
        p95  = quantile(accs, 0.95)
        series = sorted([{"x": x["step"], "y": x["acc"], "ts": x["ts"]} for x in arr], key=lambda z: (z["x"], z["ts"]))
        status = "green" if mean>=0.80 else ("yellow" if mean>=0.65 else "red")
        summary["tasks"].append({
            "task": task, "n": len(accs), "mean": mean, "p50": p50, "p95": p95,
            "status": status, "series": series
        })
    return summary

HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh_s}">
<title>VXOR Probes Scorecard</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0e13; color:#e9eef5; margin:24px; }}
  h1 {{ font-size: 20px; margin: 0 0 12px; }}
  .meta {{ opacity:.75; margin-bottom:16px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill,minmax(360px,1fr)); gap:14px; }}
  .card {{ background:#121722; border:1px solid #1e2533; padding:14px; border-radius:14px; box-shadow:0 6px 16px rgba(0,0,0,.25); }}
  .row {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .tag {{ font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid currentColor; }}
  .green {{ color:#16c784; }} .yellow {{ color:#f2c94c; }} .red {{ color:#ff6b6b; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th,td {{ padding:4px 0; font-variant-numeric: tabular-nums; }}
  svg {{ width:100%; height:80px; display:block; background:#0e131d; border-radius:10px; }}
  .footer {{ opacity:.7; margin-top:16px; font-size:12px; }}
</style>
<h1>VXOR – Mini-Validation Probes</h1>
<div class="meta">Updated: <b>{updated}</b> • Auto-refresh: {refresh_s}s • Files: {files} • Run: <code>{run}</code></div>
<div class="grid" id="cards"></div>
<div class="footer">Offline scorecard • p50/p95 over last probes • Sparklines show accuracy per step (0–100%).</div>
<script>
const DATA = {data_json};

function sparkline(el, series) {{
  if (!series || series.length===0) return;
  const svgNS = "http://www.w3.org/2000/svg";
  const w = el.clientWidth, h = el.clientHeight, pad=8;
  const xs = series.map(p=>p.x), ys = series.map(p=>p.y*100);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = 0, maxY = 100;
  const sx = v => pad + (w-2*pad) * (maxX===minX? 0.5 : (v-minX)/(maxX-minX));
  const sy = v => (h-pad) - (h-2*pad) * (v-minY)/(maxY-minY);
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
  const path = document.createElementNS(svgNS, 'path');
  const d = series.map((p,i)=> (i?'L':'M')+sx(p.x)+','+sy(p.y*100)).join(' ');
  path.setAttribute('d', d);
  path.setAttribute('fill','none');
  path.setAttribute('stroke','#6aa3ff');
  path.setAttribute('stroke-width','2');
  svg.appendChild(path);
  // wave markers every 10k steps
  const vline = (x) => {
    const ln = document.createElementNS(svgNS,'line');
    ln.setAttribute('x1', sx(x)); ln.setAttribute('x2', sx(x));
    ln.setAttribute('y1', pad); ln.setAttribute('y2', h-pad);
    ln.setAttribute('stroke', '#243042'); ln.setAttribute('stroke-dasharray','2,6');
    ln.setAttribute('stroke-width','1'); svg.appendChild(ln);
  };
  if (maxX>minX) {
    const start = Math.ceil(minX/10000)*10000;
    for (let t=start; t<=maxX; t+=10000) vline(t);
  }

  // goal lines @80% (green) & 65% (yellow)
  const addH = (v, col) => {
    const ln = document.createElementNS(svgNS,'line');
    ln.setAttribute('x1', pad); ln.setAttribute('x2', w-pad);
    ln.setAttribute('y1', sy(v)); ln.setAttribute('y2', sy(v));
    ln.setAttribute('stroke', col); ln.setAttribute('stroke-dasharray','4,4');
    ln.setAttribute('stroke-width','1'); svg.appendChild(ln);
  };
  addH(80, '#2a6e4c'); addH(65, '#6e662a');
  el.replaceChildren(svg);
}}

function card(t) {{
  const div = document.createElement('div'); div.className='card';
  const hdr = document.createElement('div'); hdr.className='row';
  const left = document.createElement('div'); left.innerHTML = '<b>'+t.task+'</b>';
  const tag = document.createElement('div'); tag.className='tag '+t.status; tag.textContent=t.status.toUpperCase();
  hdr.append(left, tag); div.appendChild(hdr);
  const tbl = document.createElement('table');
  tbl.innerHTML = `
    <tr><th>n</th><th>mean</th><th>p50</th><th>p95</th></tr>
    <tr><td>${{t.n}}</td><td>${{(t.mean*100).toFixed(2)}}%</td><td>${{(t.p50*100).toFixed(2)}}%</td><td>${{(t.p95*100).toFixed(2)}}%</td></tr>`;
  div.appendChild(tbl);
  const spark = document.createElement('div'); spark.style.width='100%'; spark.style.height='80px'; spark.style.marginTop='8px';
  div.appendChild(spark);
  setTimeout(()=>sparkline(spark, t.series), 0);
  return div;
}}

const root = document.getElementById('cards');
root.replaceChildren(...DATA.tasks.map(card));
</script>
"""

def write_html(summary, out_html, run_path, refresh_s, file_count):
    html_text = HTML_TEMPLATE.format(
        updated=summary.get("updated_utc",""),
        refresh_s=int(refresh_s),
        files=file_count,
        run=html.escape(run_path),
        data_json=json.dumps(summary, ensure_ascii=False)
    )
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory, e.g. runs/phase2_mps_real")
    ap.add_argument("--watch", type=int, default=0, help="Rebuild every N seconds (0 = one-shot)")
    ap.add_argument("--refresh", type=int, default=15, help="Meta refresh seconds in HTML")
    args = ap.parse_args()

    results_dir = os.path.join(args.run, "probes", "results")
    out_html = os.path.join(args.run, "probes", "scorecard.html")
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory not found: {results_dir}")

    def build_once():
        rows = read_results(results_dir)
        summary = make_summary(rows)
        write_html(summary, out_html, args.run, args.refresh, file_count=len(glob.glob(os.path.join(results_dir,"*.json"))))
        print(f"✅ Wrote {out_html} • tasks={len(summary['tasks'])} • files={len(glob.glob(os.path.join(results_dir,'*.json')))}")

    if args.watch > 0:
        try:
            while True:
                build_once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("Bye.")
    else:
        build_once()

if __name__ == "__main__":
    main()
