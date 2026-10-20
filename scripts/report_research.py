"""Tables/plots/deck generated exclusively from persisted episode rows."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from src.eval.research import crossed_summary, paired_difference

METHODS = {'B1':'Fixed-price (s,S)','B2':'EOQ (zero setup cost)','B3':'Myopic (s,S)',
           'B4':'Independent PPO','B5':'Single-agent PPO','B6':'MAPPO','B7':'Proposed variant'}


def generate(directory):
    directory = Path(directory)
    rows = json.loads((directory/'episodes.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if len(rows)!=summary['episode_rows']:
        raise ValueError('declared and enumerated episode totals disagree')
    coverage = json.loads((directory/'coverage.json').read_text())
    groups = {}
    for r in rows:
        groups.setdefault((r['scenario'],r['ablation'],r['method'],r['kind']),[]).append(r)
    stats = []
    for key, group in groups.items():
        for metric in [k for k in group[0] if k!='seed' and isinstance(group[0][k],(float,int))]:
            if metric=='training_seed':
                continue
            s = crossed_summary(group,metric)
            stats.append(dict(scenario=key[0],ablation=key[1],method=key[2],kind=key[3],metric=metric,**s))
    with (directory/'aggregate_metrics.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    paired=[]
    baseline={(r['scenario'],r['seed']):r for r in rows if r['method']=='B1' and r['ablation']=='none'}
    for key,group in groups.items():
        if key[1]!='none' or key[2]=='B1':
            continue
        if not all((r['scenario'],r['seed']) in baseline for r in group):
            continue
        for metric in ('profit','fill_rate','stockout_rate'):
            differences=[dict(training_seed=r['training_seed'],seed=r['seed'],
                difference=r[metric]-baseline[r['scenario'],r['seed']][metric]) for r in group]
            paired.append(dict(scenario=key[0],method=key[2],reference='B1',metric=metric,
                               **crossed_summary(differences,'difference')))
    trained_reference={(r['scenario'],r['method'],str(r['training_seed']),r['seed']):r for r in rows if r['ablation']=='none'}
    for key,group in groups.items():
        if key[1]!='none' or key[2] not in ('B6','B7'):
            continue
        for reference in ('B4','B5'):
            keys=[(r['scenario'],reference,str(r['training_seed']),r['seed']) for r in group]
            if not all(k in trained_reference for k in keys):
                continue
            for metric in ('profit','fill_rate','stockout_rate'):
                differences=[dict(training_seed=r['training_seed'],seed=r['seed'],
                    difference=r[metric]-trained_reference[k][metric]) for r,k in zip(group,keys)]
                paired.append(dict(scenario=key[0],method=key[2],reference=reference,metric=metric,
                                   **crossed_summary(differences,'difference')))
    with (directory/'aggregate_paired_comparisons.csv').open('w',newline='') as f:
        if paired:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    lookup = {(s['scenario'],s['ablation'],s['method'],s['metric']):s for s in stats}
    table = ['| Method | Profit mean [95% CI] | Fill rate | Stockout rate | Avg inventory | Profit SD |',
             '|---|---:|---:|---:|---:|---:|']
    for m,name in METHODS.items():
        key=('S2','none',m,'profit')
        if key not in lookup:
            table.append(f'| {name} | not measured | not measured | not measured | not measured | not measured |'); continue
        p=lookup[key]
        cells=[f"{p['mean']:.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}]"]
        for metric in ('fill_rate','stockout_rate','average_inventory'):
            cells.append(f"{lookup['S2','none',m,metric]['mean']:.3f}")
        cells.append(f"{p['std']:.2f}")
        table.append(f"| {name} | "+' | '.join(cells)+' |')
    degradation=[]
    for m in METHODS:
        base = lookup.get(('S2','none',m,'profit'))
        if not base: continue
        for scenario in [f'S{i}' for i in range(1,9)]:
            shifted = lookup.get((scenario,'none',m,'profit'))
            if shifted:
                degradation.append(dict(method=m,scenario=scenario,reference='S2',
                    absolute_profit_change=shifted['mean']-base['mean'],
                    relative_degradation=(base['mean']-shifted['mean'])/max(abs(base['mean']),1e-12)))
    with (directory/'degradation.csv').open('w',newline='') as f:
        if degradation:
            w=csv.DictWriter(f,fieldnames=list(degradation[0]));w.writeheader();w.writerows(degradation)
    measured=sum(c.get('status')=='evaluated' for c in coverage)
    probes=sum(c.get('status')=='inference_probe_not_retrained' for c in coverage)
    missing=sum(c.get('status')=='missing_checkpoint' for c in coverage)
    ablation_table=['| Ablation | Method | Economic reward mean | Fill rate | Evidence class | Training seeds |',
                    '|---|---|---:|---:|---|---:|']
    for a in [f'A{i}' for i in range(1,8)]:
        candidates=[s for s in stats if s['ablation']==a and s['metric']=='profit']
        if not candidates:
            ablation_table.append(f'| {a} | not measured | not measured | not measured | retraining/model missing | 0 |')
        for p in candidates:
            fill=lookup[p['scenario'],a,p['method'],'fill_rate']['mean']
            label='execution probe, NOT retrained' if p['kind']=='inference_probe' else ('IPPO architecture comparison; not isolated critic surgery' if a=='A1' else 'matched trained checkpoint (see provenance)')
            ablation_table.append(f"| {a} | {p['method']} | {p['mean']:.2f} | {fill:.3f} | {label} | {p['training_seeds']} |")
    report = f'''# Retail coordination: measured development evidence

Status: **development, not a final research comparison** unless final-mode provenance gates passed.
Held-out test used: **{summary['held_out_test_used']}**. Episode rows: **{len(rows)}**.
Coverage entries: {measured} evaluated, {probes} inference probes, {missing} missing checkpoint cells.
Read `coverage.json` for exact method/scenario/ablation/seed coverage.

## In-distribution S2 results (cumulative economic reward over configured horizon)

'''+ '\n'.join(table)+'''

## A1-A7 coverage and measured diagnostics

'''+ '\n'.join(ablation_table)+'''

## Interpretation and inference limits

- Profit here is canonical economic reward, including lost-sales and price-change penalties; revenue alone is not profit.
- Confidence intervals resample the crossed training-seed × shared evaluation-seed grid (2,000 reproducible bootstrap draws). A single checkpoint has only episode uncertainty: it does **not** estimate learning-seed variability. Per-checkpoint paired t intervals versus B1 are in `paired_comparisons.csv`; with five episodes they are exploratory, without multiplicity correction.
- A2-A7 inference probes on an unmodified checkpoint test execution sensitivity only. They are **not retrained component ablations** and cannot establish causal training-component contributions. A1 requires independently trained local critics; absent models are not synthesized. Where A1 reuses B4's matched-budget IPPO checkpoints, it is an architectural comparison, not a surgically isolated critic-only intervention. A5 fixes lead time at4 versus nominal midpoint4.5, so its effect includes a small mean change as well as variance removal.
- Checkpoints trained in the legacy start1800 calibration domain are evaluated in a separate train-only validation domain; their provenance remains development-only even when there are five initialization seeds. Short 16-update experiments verify repeated training/evaluation, not convergence.
- S1-S8 are configured robustness conditions, not demonstrations of successful learning in each regime. S2 and S6 currently share stochastic lead times and calendar variability; isolating supply uncertainty requires the paired A5 control.
- S4 uses controlled negative response, with extra sweep cells labeled `S4_e-0.5` and `S4_e-1.5` when configured; `S4` is the -3 endpoint. Consult coverage for which values actually ran. S7 is a 1.5× exogenous mean shift applied after training; anchors/clamps remain unchanged. This is simulated OOD, not a new real-store dataset.
- Capacity breaches count post-handling infeasible states. Receiving clips are a distinct operational statistic, not capacity violations. Fill target is 0.95. Inventory turnover is horizon COGS divided by mean end-of-day inventory value, not annualized.
- EOQ uses zero fixed setup cost and the smallest positive grid batch (5); this degenerate benchmark is disclosed, not repaired by inventing costs.
- Locked SKU selection and base prices were selected before this split reconciliation using late historical data. This is a fixed-series conditional evaluation, not a completely untouched SKU-selection holdout. Legacy calibration was trained through day1913; the new train-only artifact ends1885. Legacy smoke checkpoints remain development-only.

## Remaining acceptance gates

1. Validate/calibrate the train-only demand model, record numerical warnings, freeze economic assumptions and service target on validation only. Do not silently relock costs/capacity.
2. Train ≥5 independent development seeds for B4/B5/B6 and A1-A7 using chronological windows and the same masks/physics as evaluation. Compare learning curves, action entropy, service and validation checkpoints.
3. Decide and freeze B7 using validation, or explicitly remove its claim from a revised design. Resolve shared-scalar versus per-SKU pricing and critic state-only versus document joint-action discrepancy before final claims.
4. Freeze checkpoint hashes/configs. Run 10 independent training seeds × ≥50 (prefer100) held-out episodes per final scenario, plus retrained A1-A7. No test tuning.
5. Publish economic/service/tail metrics, seed-aware paired intervals, all missing cells and negative findings. No evidence presently establishes a coordination advantage.
'''
    (directory/'REPORT.md').write_text(report,encoding='utf-8')
    # Data-derived profit-service scatter, never invented coordinates.
    plotted=[(m,lookup['S2','none',m,'profit']['mean'],lookup['S2','none',m,'fill_rate']['mean']) for m in METHODS if ('S2','none',m,'profit') in lookup]
    minp=min([p for _,p,_ in plotted]+[0]);maxp=max([p for _,p,_ in plotted]+[1]);span=max(maxp-minp,1)
    marks=[]
    for m,p,fill in plotted:
        x=90+fill*700;y=410-(p-minp)/span*320
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#187968"/><text x="{x+12:.1f}" y="{y:.1f}">{m}: {p:.0f}</text>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 500" role="img" aria-label="Measured profit versus fill rate"><rect width="960" height="500" fill="#faf8f2"/><g font-family="Arial" font-size="18" fill="#222"><text x="70" y="35">S2: economic reward versus service</text><path d="M90 70V410H850" fill="none" stroke="#444"/><text x="370" y="470">Fill rate (0 → 1)</text><text x="95" y="60">Profit (horizon total)</text><text x="10" y="415">'+f'{minp:.0f}'+'</text><text x="10" y="90">'+f'{maxp:.0f}'+'</text>'+''.join(marks)+'</g></svg>'
    (directory/'profit_service.svg').write_text(svg,encoding='utf-8')
    # Measured scenario means with labeled axes, no missing-data interpolation.
    scenario_names=sorted({s['scenario'] for s in stats if s['ablation']=='none'})
    values=[s['mean'] for s in stats if s['ablation']=='none' and s['metric']=='profit']
    low=min(values+[0.]); high=max(values+[1.]); span=max(high-low,1.)
    marks=[]
    colors={'B1':'#187968','B2':'#8c6239','B3':'#354c72','B4':'#ac4747','B5':'#6f5e82','B6':'#222222','B7':'#6c761e'}
    for i,name in enumerate(scenario_names):
        x=110+i*120
        marks.append(f'<text x="{x}" y="470" transform="rotate(-30 {x} 470)">{html.escape(name)}</text>')
        for m in METHODS:
            s=lookup.get((name,'none',m,'profit'))
            if s:
                y=410-(s['mean']-low)/span*320
                marks.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="{colors[m]}"><title>{m}: {s["mean"]:.2f}</title></circle>')
    width=max(960,150+len(scenario_names)*120)
    robust=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 550" role="img" aria-label="Scenario economic reward means"><rect width="100%" height="100%" fill="#faf8f2"/><g font-family="Arial" font-size="18"><text x="70" y="35">Robustness: measured mean economic reward (not convergence)</text><path d="M80 70V410H{width-30}" fill="none" stroke="#444"/><text x="5" y="90">{high:.0f}</text><text x="5" y="410">{low:.0f}</text>'+''.join(marks)+''.join(f'<text x="{80+i*120}" y="530" fill="{colors[m]}">{m}</text>' for i,m in enumerate(METHODS))+'</g></svg>'
    (directory/'robustness.svg').write_text(robust,encoding='utf-8')
    # One idea per slide; fixed 16:9 design, navigation + print.
    slides=[('Coordination is the question, not the conclusion',
        '<p>Two heterogeneous actors. Weekly prices. Daily orders.</p><p>Controlled M5-calibrated demand, lost sales, bounded suppliers and shared capacity.</p>'),
        ('Held-out status and coverage',f'<p>Held-out test used: {summary["held_out_test_used"]}</p><p>{len(rows)} real episode rows · {measured} evaluated cells · {probes} execution probes.</p><p>Missing checkpoint cells: {missing}. Five scenario seeds do not equal five training seeds.</p>'),
        ('Profit and service must be read together',svg),
        ('Ablation evidence needs retraining','<p>A1: local critics. A2: no history. A3: no replenishment price.</p><p>A4: no pricing inventory. A5: fixed lead. A6: no correlation. A7: no shared capacity.</p><p>Unmodified-checkpoint masks are execution probes, not causal ablations.</p>'),
        ('What remains before a final claim','<ol><li>Freeze assumptions and select on validation only.</li><li>Five development training seeds; ten final seeds.</li><li>At least 50, preferably 100 paired held-out episodes.</li><li>Report B1-B7, S1-S8, retrained A1-A7 and negative findings.</li></ol>'),
        ('No coordination advantage established','<p>Smoke models verify infrastructure, not convergence.</p><p>Price-scalar and critic-input discrepancies remain explicit design gates.</p><p>Use REPORT.md, metrics CSVs and coverage.json as the source of truth.</p>')]
    sections=''.join(f'<section id="slide-{i+1}"><small>TEAM 10 · RESEARCH DEVELOPMENT</small><h1>{html.escape(title)}</h1>{body}</section>' for i,(title,body) in enumerate(slides))
    deck='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Retail coordination research evidence</title><style>
    *{box-sizing:border-box}body{margin:0;background:#202a28;color:#172923;font-family:Arial,sans-serif}main{width:1920px;height:1080px;position:absolute;transform-origin:top left;background:#faf8f2}section{display:none;padding:100px 140px;height:1080px}section.active{display:block}small{font:28px monospace;color:#187968}h1{font:76px Georgia,serif;max-width:1500px;line-height:1.1}p,li{font-size:40px;line-height:1.5;max-width:1450px}svg{max-height:650px;width:1300px}nav{position:fixed;bottom:16px;right:20px;color:#172923;background:#faf8f2;font-size:20px}button{font-size:20px;min-height:44px;cursor:pointer;margin:8px}button:focus-visible{outline:3px solid #eee}@media print{body{background:white}main{position:static;transform:none!important;width:auto;height:auto}section,section.active{display:block;break-after:page;width:1920px;height:1080px}nav{display:none}}
    </style><main>'''+sections+'''</main><nav><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button></nav><script>
    const slides=[...document.querySelectorAll('section')];let current=Number(localStorage.getItem('retail-evidence-slide')||0);function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.getElementById('count').textContent=(current+1)+' / '+slides.length;localStorage.setItem('retail-evidence-slide',current)}function scale(){document.querySelector('main').style.transform='scale('+Math.min(innerWidth/1920,innerHeight/1080)+')'}document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(current+1);if(e.key==='ArrowLeft')show(current-1)});addEventListener('resize',scale);show(current);scale();</script></html>'''
    (directory/'presentation.html').write_text(deck,encoding='utf-8')
    return dict(episode_rows=len(rows),aggregate_cells=len(stats),slides=len(slides))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    print(json.dumps(generate(args.directory),indent=2))
