"""Read-only, versioned evidence. Never constructs or loads an actor."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {
    'correction': ('LEARNING_CORRECTION_REPORT.md', 'text/plain'),
    'ordering': ('ORDERING_LEARNING_REPORT.md', 'text/plain'),
    'service': ('SERVICE_FEASIBILITY_REPORT.md', 'text/plain'),
    'operating': ('OPERATING_DIAGNOSIS.md', 'text/plain'),
    'research': ('research_archive/RESEARCH_REPORT_1075.md', 'text/plain'),
    'research_current': ('RESEARCH_REPORT.md', 'text/plain'),
    'parameter_confirmation': ('PARAMETER_RL_CONFIRMATION_REPORT.md', 'text/plain'),
    'parameter_usage': ('PARAMETER_RL_USAGE.md', 'text/plain'),
    'parameter_effects': ('outputs/rl_discovery/parameter_confirmation_v1/confirmation_effects.png', 'image/png'),
    'evidence_status': ('EVIDENCE_STATUS.md', 'text/plain'),
    'ordinal': ('ORDINAL_RESIDUAL_REPORT.md', 'text/plain'),
    'consolidation_checks': ('CONSOLIDATION_VERIFICATION.md', 'text/plain'),
    'service_summary': ('outputs/service_feasibility/paired20/summary.json', 'application/json'),
    'joint_summary': ('outputs/learning_correction/paired_s2_seed11/summary.json', 'application/json'),
    'review_report': ('PROJECT_REVIEW_REPORT.md', 'text/plain'),
    'literature': ('LITERATURE_REVIEW.md', 'text/plain'),
    'system': ('IMPLEMENTED_SYSTEM.md', 'text/plain'),
    'review_response': ('REVIEW_RESPONSE.md', 'text/plain'),
    'review_charts': ('outputs/review_response/validation_audit_v3/evidence.html', 'text/html'),
    'review_metrics': ('outputs/review_response/validation_audit_v3/metrics.json', 'application/json'),
    'compute': ('outputs/review_response/compute.json', 'application/json'),
    'final_report': ('outputs/structured_final/evaluation_recovered_v1/REPORT.md', 'text/plain'),
    'final_primary': ('outputs/structured_final/evaluation_recovered_v1/final_primary.png', 'image/png'),
    'final_robustness': ('outputs/structured_final/evaluation_recovered_v1/final_robustness.png', 'image/png'),
    'joint_pilot': ('JOINT_PARAMETER_PILOT_REPORT.md', 'text/plain'),
}


def final_snapshot():
    """Show final scores only after verified artifacts have passed release QA."""
    folder=ROOT/'outputs/structured_final/evaluation_recovered_v1'
    if not (folder/'release.json').is_file():
        return dict(status='pending',message='Final study is in progress. Verified final results will appear after report and figure review.')
    try:
        release=json.loads((folder/'release.json').read_text())
        required=['summary.json','verification.json','REPORT.md','final_primary.png','final_robustness.png']
        if release['status']!='released' or release['visual_qa'] is not True:raise ValueError('release not reviewed')
        for name in required:
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=release['sha256'][name]:raise ValueError('released evidence changed')
        summary=json.loads((folder/'summary.json').read_text());proof=json.loads((folder/'verification.json').read_text())
        if summary['status']!='complete' or proof['status']!='passed':raise ValueError('incomplete evidence')
        if proof['checked_episodes']!=17000 or proof['exact_replays']!=40:raise ValueError('incomplete coverage')
        if proof['primary_gate']!=summary['primary_gate']:raise ValueError('gate mismatch')
        analysis=summary['analysis']['S2'];seeds=[211,222,233,244,255,266,277,288,299,310]
        means={metric:sum(analysis['means'][f'main_{s}'][metric] for s in seeds)/10 for metric in ['reward_day','fill']}
        return dict(status='verified',gate=summary['primary_gate'],main=means,
            reference=analysis['means']['full7_5'],primary=analysis['primary'],
            every_seed_positive=analysis['every_seed_reward_positive'])
    except (OSError,ValueError,KeyError,TypeError):
        return dict(status='unavailable',message='Final evidence verification is unavailable. Development results below remain separate.')


def snapshot():
    data = json.loads((ROOT / 'demo/progress_snapshot.json').read_text(encoding='utf-8'))
    data['structured_final']=final_snapshot()
    data['evidence'] = [dict(id=key, url='/api/evidence/' + key,
                             available=(ROOT / relative).is_file())
                        for key, (relative, _) in EVIDENCE.items()]
    return data
