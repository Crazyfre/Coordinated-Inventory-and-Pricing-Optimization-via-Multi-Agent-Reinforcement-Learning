"""Inventory engine tests — hand-computed transitions and capacity (Phase R1).

The transition tests below are the Doc S2.6 equations checked against
numbers computed BY HAND in the plan's exit criteria.
"""

import numpy as np
import pytest

from src.env.inventory import InventoryEngine


class TestLostSalesTransition:
    def test_hand_computed_basic(self):
        """I=[10,5]; A=[5,5]; D=[12,3]
        SKU0: avail 15, S=12, U=0, I'=3
        SKU1: avail 10, S=3,  U=0, I'=7
        """
        eng = InventoryEngine(n_skus=2, capacity=100, initial_inventory=[10.0, 5.0])
        accepted, clipped = eng.receive(np.array([5.0, 5.0]))
        assert np.allclose(accepted, [5, 5]) and clipped.sum() == 0
        sales, lost = eng.sell(np.array([12.0, 3.0]))
        assert np.allclose(sales, [12, 3])
        assert np.allclose(lost, [0, 0])
        assert np.allclose(eng.on_hand, [3, 7])

    def test_hand_computed_stockout(self):
        """I=[3,0]; D=[10,4] -> S=[3,0], U=[7,4], I'=[0,0]"""
        eng = InventoryEngine(n_skus=2, capacity=100, initial_inventory=[3.0, 0.0])
        eng.receive(np.zeros(2))
        sales, lost = eng.sell(np.array([10.0, 4.0]))
        assert np.allclose(sales, [3, 0])
        assert np.allclose(lost, [7, 4])
        assert np.allclose(eng.on_hand, [0, 0])

    def test_sales_capped_at_availability(self):
        eng = InventoryEngine(n_skus=1, capacity=1000, initial_inventory=[7.0])
        _, _ = eng.sell(np.array([100.0]))
        assert eng.on_hand[0] == pytest.approx(0.0)

    def test_inventory_never_negative(self):
        eng = InventoryEngine(n_skus=4, capacity=500, initial_inventory=np.full(4, 2.0))
        for _ in range(10):
            _, _ = eng.sell(np.full(4, 9.0))
            assert np.all(eng.on_hand >= 0)


class TestCapacityEnforcement:
    def test_hand_computed_proportional_clip(self):
        """C=100, on-hand total=90, arrivals [50,30] (total 80 > headroom 10).
        scale = 10/80 = 0.125 -> accepted [6.25, 3.75], clipped [43.75, 26.25].
        """
        eng = InventoryEngine(n_skus=2, capacity=100.0, initial_inventory=[60.0, 30.0])
        accepted, clipped = eng.receive(np.array([50.0, 30.0]))
        assert np.allclose(accepted, [6.25, 3.75])
        assert np.allclose(clipped, [43.75, 26.25])
        assert eng.total_on_hand == pytest.approx(100.0)
        assert eng.capacity_violation_events == 1
        assert eng.capacity_clipped_units == pytest.approx(70.0)

    def test_on_hand_never_exceeds_capacity(self):
        eng = InventoryEngine(n_skus=10, capacity=200.0, initial_inventory=np.full(10, 5.0))
        rng = np.random.default_rng(0)
        for _ in range(50):
            eng.receive(rng.uniform(0, 60, size=10))
            eng.sell(rng.poisson(5, size=10).astype(float))
            assert eng.total_on_hand <= 200.0 + 1e-9

    def test_no_clip_below_headroom(self):
        eng = InventoryEngine(n_skus=2, capacity=100.0, initial_inventory=[10.0, 10.0])
        accepted, clipped = eng.receive(np.array([40.0, 39.0]))
        assert clipped.sum() == 0
        assert eng.capacity_violation_events == 0

    def test_symmetric_scaling_order_independent(self):
        """Proportional rule must give each SKU the same fill fraction."""
        eng = InventoryEngine(n_skus=3, capacity=50.0, initial_inventory=np.zeros(3))
        accepted, clipped = eng.receive(np.array([10.0, 20.0, 70.0]))
        frac0 = accepted[0] / 10.0
        frac1 = accepted[1] / 20.0
        frac2 = accepted[2] / 70.0
        assert frac0 == pytest.approx(frac1) == pytest.approx(frac2)

    def test_negative_arrival_rejected(self):
        eng = InventoryEngine(n_skus=2, capacity=100.0)
        with pytest.raises(ValueError):
            eng.receive(np.array([5.0, -1.0]))

    def test_initial_inventory_scaled_to_capacity(self):
        eng = InventoryEngine(n_skus=2, capacity=50.0, initial_inventory=[400.0, 100.0])
        assert eng.total_on_hand == pytest.approx(50.0)


class TestSharedCapacityCoupling:
    def test_one_sku_consumes_another_headroom(self):
        """SKU0 fills the warehouse; SKU1's delivery gets squeezed."""
        eng = InventoryEngine(n_skus=2, capacity=100.0, initial_inventory=[100.0, 0.0])
        accepted, clipped = eng.receive(np.array([0.0, 25.0]))
        assert accepted[0] == 0.0
        assert accepted[1] == pytest.approx(0.0)
        assert clipped[1] == pytest.approx(25.0)

    def test_freed_space_available_next_day(self):
        eng = InventoryEngine(n_skus=2, capacity=100.0, initial_inventory=[100.0, 0.0])
        eng.receive(np.array([0.0, 10.0]))  # fully clipped
        eng.sell(np.array([60.0, 0.0]))     # frees 60 units
        accepted, clipped = eng.receive(np.array([0.0, 30.0]))
        assert accepted[1] == pytest.approx(30.0) and clipped[1] == 0.0
