"""Generate the outcome report only from a completed, independently verified study."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import INPUTS


def report(study, destination):
    root = Path(study); target = Path(destination)
    if target.exists():
        raise FileExistsError('report already exists; review rather than overwrite')
    summary = json.loads((root/'summary.json').read_text())
    audit = json.loads((root/'verification.json').read_text())
    assert summary['status'] == 'complete' and audit['status'] == 'passed'
    a = summary['analysis']; seeds = [211, 222, 233, 244, 255]
    gate = 'PASSES' if a['learning_gate'] else 'FAILS'
    means = a['means']; average = {metric: np.mean([means[f'learned_{s}'][metric] for s in seeds])
                                  for metric in ['reward_day', 'fill']}
    rows = ['# Parameter-space policy-gradient RL: five-seed confirmation', '',
        f'Completed18 October2026. The predeclared economic/service gate **{gate}**. '
        'This is ordering-only evidence for episodic parameter-space policy-gradient RL on the unchanged simulator. '
        'It is not PPO/MAPPO, neural representation learning, a coordination result or untouched temporal validation.', '',
        '## Primary comparison', '',
        '|Reference|Reward/day gain|98.75% interval|Fill gain (percentage points)|98.75% interval (pp)|',
        '|---|---:|---|---:|---|']
    for reference, label in [('full7_5', 'Full7.5 / own deterministic start'), ('initial', 'Own initial exploratory policy')]:
        reward, fill = a['endpoints'][reference+'_reward_day'], a['endpoints'][reference+'_fill']
        rows.append(f"|{label}|{reward['mean']:+.6f}|[{reward['ci9875'][0]:.6f},{reward['ci9875'][1]:.6f}]|"
                    f"{100*fill['mean']:+.6f}|[{100*fill['ci9875'][0]:.6f},{100*fill['ci9875'][1]:.6f}]|")
    rows += ['', f"Learned mean reward/day is{average['reward_day']:.6f}, versus{means['full7_5']['reward_day']:.6f} "
              f"for full7.5. Learned mean fill is{100*average['fill']:.4f}%, versus{100*means['full7_5']['fill']:.4f}%. "
              'The absolute95% service target remains separate and is not established by the noninferiority safeguard.', '',
              'Intervals use10000 crossed bootstrap draws over five training seeds and128 scenario columns, '
              'seed20261019, with98.75% marginal intervals for four primary endpoints. Coverage is approximate; '
              'five seeds do not remove all training-variability uncertainty. No result was selected by seed or checkpoint.', '',
              '|Training seed|Reward/day|Fill|Gain vs full7.5|Gain vs own exploratory start|', '|---|---:|---:|---:|---:|']
    for i, seed in enumerate(seeds):
        m = means[f'learned_{seed}']
        rows.append(f"|{seed}|{m['reward_day']:.6f}|{m['fill']:.6f}|"
                    f"{a['endpoints']['full7_5_reward_day']['seed_means'][i]:+.6f}|"
                    f"{a['endpoints']['initial_reward_day']['seed_means'][i]:+.6f}|")
    rows += ['', f"Every-seed-positive reward requirement: **{a['every_seed_reward_positive']}**.", '',
              '## Other reference policies (descriptive)', '', '|Controller|Reward/day|Fill|', '|---|---:|---:|']
    for label in ['full7_5', 'projected', 'constant', 'ss']:
        rows.append(f"|{label}|{means[label]['reward_day']:.6f}|{means[label]['fill']:.6f}|")
    rows += ['', 'These comparisons are descriptive, not extra preregistered inferential successes. '
              'Do not infer universal dominance, optimality, convergence or coordination. '
              'The main comparison is against a strong exact starting policy, not the weaker fitted center.', '',
              '## What was learned', '',
              'Ten per-SKU cover coefficients parameterize a daily state-feedback inventory rule. '
              'Training samples positive/negative Gaussian perturbations and uses paired complete discounted returns '
              'to estimate a score-function policy gradient, followed by Adam ascent. No teacher labels, learned critic '
              'or changed reward is involved. The method also belongs to black-box Gaussian-smoothed optimization; '
              'report that overlap honestly. See `PARAMETER_POLICY_GRADIENT_METHOD.md` for the derivation.', '',
              'Both development arms and their selection are preserved in `PARAMETER_RL_DEVELOPMENT_REPORT.md`. '
              'The10-parameter arm was chosen on64 independent training scenarios before these five fresh runs. '
              'None of the pilot weights transferred. Every confirmation run started exactly at full7.5, used1024 '
              'training episodes/64 updates, and contributed its final mean policy to this assessment.', '',
              '## Verification, accounting and limitations', '',
              f"The independent archive verifier passed:320 reconstructed score-function/Adam updates,5120 training episodes,"
              '1792 assessment episodes,independently calculated actions/rewards/paired intervals and ten exact restored '
              'simulator replays. It added zero simulated days. Six method/confirmation tests passed before launch; '
              'two export-loader tests passed separately. No full-suite rerun is claimed.', '',
              f"Training took{summary['training_seconds']:.3f}s and assessment/replay{summary['assessment_seconds']:.3f}s "
              f"(campaign{summary['seconds']:.3f}s). Confirmation totals6922episodes/193816days; including both development "
              'pilots totals8202episodes/229656days. These totals exclude historical failed studies, whose separate '
              'accounts remain preserved. Runtime ceilings were amended prospectively from measured pilot throughput; '
              'scientific sample counts and gates were unchanged.', '',
              'Training windows lie in15..1885; confirmation uses fresh stochastic keys on1886..1913. '
              'That validation calendar has been used in earlier research, so this is a fresh paired cohort conditional '
              'on the development setting, not untouched temporal confirmation. Test1914..1941 remains unopened. '
              'Prices stay fixed1.20x; demand calibration, receiving capacity, supplier dynamics, delivered-unit '
              'billing, costs and28-day terminal contract are unchanged. The full physical order grid is used, '
              'explicitly expanding the old diagnostic residual support without changing the simulator.', '',
              'All five policies are retained; no best-seed selection or demo replacement. The original project’s '
              'MAPPO pricing/replenishment coordination claim and final held-out experiment remain separate work. '
              'This study neither retroactively clears previous failed gates nor adds a ninth literature paper.', '',
              f'Artifacts: `{root.as_posix()}/`, including frozen checkpoints,complete traces,manifest,summary,verification.json '
              'and the exported `policies/` bundle when the gate passes. `confirmation_effects.png` and '
              '`confirmation_effects.pdf` visualize every seed and the aggregate comparison.', '',
              'No simulator/economic changes,historical artifact replacement,commits or pushes.']
    target.write_text('\n'.join(rows)+'\n', encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--study', required=True); p.add_argument('--destination', required=True)
    args = p.parse_args(); report(args.study, args.destination)
