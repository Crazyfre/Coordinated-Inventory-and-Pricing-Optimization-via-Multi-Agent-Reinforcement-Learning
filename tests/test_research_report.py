import json
from scripts import report_research


def test_report_contains_only_measured_cells_and_labels_missing(tmp_path):
    rows = [dict(scenario='S2',ablation='none',method='B1',training_seed='not_applicable',kind='baseline',seed=i,profit=10.+i,fill_rate=.8,stockout_rate=.2,average_inventory=20.,capacity_violations=0) for i in range(5)]
    (tmp_path/'episodes.json').write_text(json.dumps(rows))
    (tmp_path/'summary.json').write_text(json.dumps(dict(held_out_test_used=False,episode_rows=5)))
    (tmp_path/'coverage.json').write_text('[]')
    result = report_research.generate(tmp_path)
    assert result['episode_rows']==5
    text = (tmp_path/'REPORT.md').read_text()
    assert '12.00' in text and 'not measured' in text
    assert '| A7 |' in text
    assert (tmp_path/'profit_service.svg').exists()
    assert (tmp_path/'robustness.svg').exists()
    assert (tmp_path/'aggregate_paired_comparisons.csv').exists()
    deck = (tmp_path/'presentation.html').read_text()
    assert 'keydown' in deck and 'localStorage' in deck and 'Held-out' in deck


def test_report_pairs_coordination_against_independent_training(tmp_path):
    import csv
    rows=[dict(scenario='S2',ablation='none',method=m,training_seed=t,kind='checkpoint_evaluation',seed=e,
          profit=float(e+(2 if m=='B6' else 0)),fill_rate=.8,stockout_rate=.2,average_inventory=20.,capacity_violations=0)
          for m in ['B4','B6'] for t in [11,22] for e in range(5)]
    (tmp_path/'episodes.json').write_text(json.dumps(rows))
    (tmp_path/'summary.json').write_text(json.dumps(dict(held_out_test_used=False,episode_rows=len(rows))))
    (tmp_path/'coverage.json').write_text('[]')
    report_research.generate(tmp_path)
    comparisons=list(csv.DictReader((tmp_path/'aggregate_paired_comparisons.csv').open()))
    pair=[r for r in comparisons if r['method']=='B6' and r['reference']=='B4' and r['metric']=='profit']
    assert len(pair)==1 and float(pair[0]['mean'])==2
