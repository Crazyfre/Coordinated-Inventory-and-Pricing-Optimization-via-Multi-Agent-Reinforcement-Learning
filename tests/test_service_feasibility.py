import numpy as np


def test_conditional_bound_respects_initial_sku_stock():
    from scripts.audit_service_feasibility import conditional_bound
    demand = np.array([[10., 0.], [10., 0.], [700., 0.]])
    result = conditional_bound(demand, np.array([5., 50.]), 500., 2)
    assert result == 505 / 720


def test_fit_status_does_not_equate_finite_with_converged():
    from scripts.audit_service_feasibility import fit_status
    assert fit_status([1., .5], {'converged': False}, [1., 2.])['accepted'] is False
    assert fit_status([1., .5], {'converged': True}, [1., 2.])['accepted'] is True
    assert fit_status([1., .5], {}, [1., 2.])['accepted'] is False


def test_rank_probe_exposes_exact_redundancy():
    from scripts.audit_nb_probes import design_dependency
    result = design_dependency(np.array([[1.,0.,0.],[1.,1.,1.],[1.,2.,2.]]), ['const','a','duplicate'])
    assert result['rank'] == 2
    assert result['null_residual_max'] < 1e-10
    assert len(result['drop_columns']) == 1


def test_zero_demand_bound_matches_fill_convention():
    from scripts.audit_service_feasibility import conditional_bound
    assert conditional_bound(np.zeros((3,2)),np.ones(2),500,2) == 1.
