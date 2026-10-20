import json
from scripts import evaluate_research


def test_runner_persists_real_paired_rows_and_coverage(tmp_path):
    cfg = dict(episodes=5, days=7, split='validation', scenarios=['S1'],
               ablations=[], methods=['B1','B2','B3'], checkpoints=[],
               output=str(tmp_path), mode='development')
    result = evaluate_research.run(cfg)
    assert result['episode_rows'] == 15
    rows = json.loads((tmp_path/'episodes.json').read_text())
    assert len(rows) == 15
    assert len({r['seed'] for r in rows}) == 5
    assert (tmp_path/'metrics.csv').exists()
    assert (tmp_path/'paired_comparisons.csv').exists()
    assert result['held_out_test_used'] is False


def test_crossed_seed_summary_preserves_training_seed_unit():
    from src.eval.research import crossed_summary
    rows = [dict(training_seed=t,seed=e,profit=float(t+e)) for t in range(5) for e in range(5)]
    s = crossed_summary(rows,'profit',seed=12,replicates=200)
    assert s['mean'] == 4
    assert s['training_seeds'] == s['evaluation_seeds'] == 5
    assert s['ci_low'] < 4 < s['ci_high']
    assert s['training_seed_std'] > 0
    assert s == crossed_summary(rows,'profit',seed=12,replicates=200)


def test_learned_policy_adapters_load_normalized_checkpoints(tmp_path):
    import torch
    from scripts.train_research import build_agent
    from src.experiments.research import make_research_env
    from src.eval.research import evaluate_episode
    torch.set_num_threads(1)
    env = make_research_env(days=7)
    for method, name in [('B4','independent'),('B5','joint'),('B6','mappo')]:
        torch.manual_seed(1)
        agent = build_agent(name,env,True)
        checkpoint = tmp_path/f'{name}.pt'; agent.save(checkpoint)
        order, price = evaluate_research.policies(method,env,dict(path=str(checkpoint),stable=True))
        row = evaluate_episode(env,order,price,10)
        assert row['capacity_violations']==0
        assert torch.isfinite(torch.tensor(row['profit']))


def test_final_configuration_cannot_open_test_without_coverage(tmp_path):
    import pytest
    cfg=dict(mode='development',split='test',output=str(tmp_path),episodes=5,checkpoints=[])
    with pytest.raises(ValueError,match='forbidden'):
        evaluate_research.run(cfg)


def test_proposed_variant_requires_explicit_identity(tmp_path):
    import pytest
    from src.experiments.research import make_research_env
    env=make_research_env(days=7)
    with pytest.raises(ValueError,match='selected'):
        evaluate_research.policies('B7',env,dict(path='not-used'))
    from scripts.train_research import build_agent
    agent=build_agent('mappo',env,True)
    path=tmp_path/'candidate.pt';agent.save(path)
    order,price=evaluate_research.policies('B7',env,dict(path=str(path),stable=True,
        variant_family='B6',variant_name='explicit-test-candidate'))
    assert callable(order) and callable(price)


def test_a1_cannot_relabel_centralized_model_as_no_critic(tmp_path):
    import pytest
    cfg=dict(mode='development',split='validation',output=str(tmp_path),episodes=5,
        scenarios=[],ablations=['A1'],methods=['B6'],checkpoints=[dict(method='B6',ablation='A1',path='unused')])
    with pytest.raises(ValueError,match='A1'):
        evaluate_research.run(cfg)


def test_runner_keeps_elasticity_sweep_conditions_separate(tmp_path):
    cfg=dict(mode='development',split='validation',output=str(tmp_path),days=7,
        episodes=5,scenarios=['S4'],ablations=[],methods=['B1'],checkpoints=[],
        elasticity_sweep=[-.5,-1.5,-3.])
    result=evaluate_research.run(cfg)
    rows=json.loads((tmp_path/'episodes.json').read_text())
    assert result['episode_rows']==15
    assert len({r['scenario'] for r in rows})==3


def test_discovery_ignores_incomplete_runs_and_retains_exact_provenance(tmp_path):
    complete=tmp_path/'mappo'/'seed_11';complete.mkdir(parents=True)
    (complete/'config.json').write_text(json.dumps(dict(method='mappo',seed=11,stable=True,updates=128,days=28)))
    (complete/'checkpoint.pt').write_bytes(b'fixture-not-loaded')
    assert evaluate_research.discover_checkpoints(tmp_path)==[]
    (complete/'summary.json').write_text(json.dumps(dict(checkpoint_roundtrip_verified=True)))
    found=evaluate_research.discover_checkpoints(tmp_path)
    assert len(found)==1 and found[0]['method']=='B6'
    assert found[0]['training_seed']==11 and found[0]['stable'] is True
    assert found[0]['frozen'] is False


def test_discovery_reads_new_clean_training_layout(tmp_path):
    complete=tmp_path/'mappo'/'S2'/'seed_11';complete.mkdir(parents=True)
    (complete/'config.json').write_text(json.dumps(dict(method='mappo',scenario='S2',seed=11,updates=64,days=28,
        train_window_range=[15,1858],random_train_window_per_episode=True,held_out_test_used=False,
        calibration_sha256='recorded-source-hash',protocol='clean development')))
    (complete/'checkpoint.pt').write_bytes(b'fixture-not-loaded')
    (complete/'summary.json').write_text(json.dumps(dict(checkpoint_roundtrip_verified=True)))
    found=evaluate_research.discover_checkpoints(tmp_path)
    assert len(found)==1 and found[0]['training_end_day']==1885
    assert found[0]['training_scenario']=='S2' and found[0]['frozen'] is False


def test_training_scenario_selection_does_not_merge_duplicate_seed_models():
    import pytest
    models=[dict(method='B6',training_seed=11,training_scenario=s) for s in ['S1','S2','S3']]
    selected=evaluate_research.select_checkpoints(models,'S2')
    assert len(selected)==1 and selected[0]['training_scenario']=='S2'
    with pytest.raises(ValueError,match='duplicate'):
        evaluate_research.select_checkpoints([models[1],models[1]],'S2')


def test_final_requires_all_methods_and_complete_elasticity_sweep(tmp_path):
    import pytest
    cfg=dict(mode='final',split='test',output=str(tmp_path),episodes=100,
        methods=['B1'],scenarios=[f'S{i}' for i in range(1,9)],ablations=[f'A{i}' for i in range(1,8)])
    with pytest.raises(ValueError,match='B1-B7'):
        evaluate_research.run(cfg)
    cfg['methods']=[f'B{i}' for i in range(1,8)]
    with pytest.raises(ValueError,match='elasticity'):
        evaluate_research.run(cfg)
