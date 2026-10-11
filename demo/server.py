"""Web demo backend — stdlib only (no new dependencies).

Serves demo/index.html and a small JSON API that composes the REAL project
modules (nothing is faked for the UI):

    src/config/locked_config.py    -> locked v1 constants (C=500, lead [2,7], ...)
    src/config/skus.json           -> locked 10-SKU selection + real M5 prices
    src/env/replenishment_env.py   -> ReplenishmentEnv (R2) as the simulator
    src/demand/simulator.py        -> CalibratedDemandSimulator (P1) as demand source
    src/env/forecast.py            -> RecentMeanForecaster (R2)

Heuristic policies (do-nothing / constant / (s,S)) are the R1 baselines;
they stand in for the RL agents (R3/P4 are not yet trained).

Usage:   .venv/Scripts/python demo/server.py [--port 8000]
Open:    http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.locked_config import get_locked_config  # noqa: E402
from src.config.sku_config import load_skus  # noqa: E402
from src.demand.simulator import CalibratedDemandSimulator  # noqa: E402
from src.env.forecast import RecentMeanForecaster  # noqa: E402
from src.env.replenishment_env import ReplenishmentEnv  # noqa: E402

CFG = get_locked_config()
N = CFG.env.n_skus

# ---------------------------------------------------------------------------
# SKU metadata (skus.json locked order == the simulator's series_order)
# ---------------------------------------------------------------------------
_SKU_ROWS = load_skus()
SKUS = [
    {
        "series_id": s.series_id,
        "item": s.item_id,
        "cat": s.cat_id,
        "base_price": s.base_price,
        "price_min": s.price_min,
        "price_max": s.price_max,
        "mean_daily": s.mean_daily_sales,
        # Procurement cost: documented demo assumption (60% of base price),
        # uniform across policies; the locked doc leaves unit costs to calibrate.
        "unit_cost": round(s.base_price * 0.6, 4),
    }
    for s in _SKU_ROWS
]
BASE_PRICES = np.array([s["base_price"] for s in SKUS])
UNIT_COSTS = np.array([s["unit_cost"] for s in SKUS])

POLICIES = ("do_nothing", "constant", "sS")

GRID = np.arange(0, CFG.replenishment.q_max + 1, CFG.replenishment.order_step)


def _to_grid_index(qty: np.ndarray) -> np.ndarray:
    """The env's action space is MultiDiscrete over the order grid, so a
    desired quantity must be snapped to the nearest grid index (step 5)."""
    idx = np.round(np.asarray(qty, dtype=float) / CFG.replenishment.order_step)
    return np.clip(idx, 0, len(GRID) - 1).astype(int)


def _make_policy(name: str, params: dict, base_demand: np.ndarray):
    """Return fn(env) -> order qty vector (n,). Uses env.observation_components().

    Knobs are expressed in DAYS OF DEMAND COVER so one slider works across
    SKUs whose M5 mean daily demand spans ~4-99 units (FOODS vs HOBBIES).
    """
    if name == "do_nothing":
        return lambda env: np.zeros(N, dtype=int)
    if name == "constant":
        days = float(params.get("order_qty", 2))  # units/day = days x base demand
        q = np.maximum(days * base_demand, 0.0)
        return lambda env: _to_grid_index(q)
    if name == "sS":
        s_days = float(params.get("s", 7))       # reorder point: days of cover
        S_days = float(params.get("S", 12))      # order-up-to: days of cover

        def fn(env: ReplenishmentEnv):
            oc = env.observation_components()
            position = oc["inventory"] + env.pipeline.outstanding_from(env.day + 1)
            s_lvl = s_days * base_demand
            S_lvl = S_days * base_demand
            need = np.where(position < s_lvl, S_lvl - position, 0.0)
            return _to_grid_index(np.maximum(need, 0.0))

        return fn
    raise ValueError(f"unknown policy: {name}")


# ---------------------------------------------------------------------------
# Episode runner — drives the REAL ReplenishmentEnv + CalibratedDemandSimulator
# ---------------------------------------------------------------------------

def run_episode(
    *,
    policy_name: str = "sS",
    params: dict | None = None,
    elasticity: float = -1.5,
    shock_std: float = 0.10,
    capacity: float = 500.0,
    lead_min: int = 2,
    lead_max: int = 7,
    price_mode: str = "base",          # "base" (flat) | "weekly" (hook-driven)
    weekly_multiplier: float = 1.00,
    days: int = 112,
    seed: int = 0,
) -> dict:
    params = params or {}
    rng = np.random.default_rng(seed)
    lt_cfg = replace(CFG.lead_time, min_days=int(lead_min), max_days=int(lead_max))

    sim = CalibratedDemandSimulator(
        elasticity=float(elasticity),
        shock_std=float(shock_std),
        rng=rng,
        start_day_index=1,
    )

    # Weekly pricing: the env hook sets a NEW price on days 1, 8, 15 ... and
    # HOLDS it between hook days (locked cadence, Doc S2.5). In "base" mode
    # the hook never fires, so price stays at the M5 base prices.
    if price_mode == "weekly":
        mult = float(weekly_multiplier)

        def pricing_hook(day: int):
            return BASE_PRICES * mult if (day - 1) % 7 == 0 else None
    else:
        pricing_hook = None

    env = ReplenishmentEnv(
        config=CFG,
        episode_length_days=int(days),
        demand_source=sim,
        forecaster=RecentMeanForecaster(N, window=7),
        pricing_hook=pricing_hook,
        base_price=BASE_PRICES,
        unit_cost=UNIT_COSTS,
        initial_inventory=5.0 * sim.level_anchor,  # 5 days of expected demand
        capacity=float(capacity),
        obs_pipe_horizon=int(lead_max),
    )
    policy = _make_policy(policy_name, params, base_demand=sim.level_anchor)
    sim.set_current_price(BASE_PRICES)  # price must be set before any sampling
    obs, info = env.reset(seed=seed)
    env.pipeline.cfg = lt_cfg  # apply the UI's lead-time knob (pipeline exists after reset)
    # Common-random-numbers discipline (Doc S5.6): give the demand simulator a
    # DEDICATED stream so lead-time/capacity/policy knobs change only what they
    # should — the demand path stays identical across knob comparisons.
    sim.reset(rng=np.random.default_rng(seed * 1000 + 7))

    days_out = []
    while True:
        orders = policy(env)
        obs, r, _term, trunc, info = env.step(orders)
        days_out.append({
            "day": info["day"],
            "demand": info["demand"].tolist(),
            "sales": info["sales"].tolist(),
            "lost": info["lost_sales"].tolist(),
            "orders": orders.tolist(),
            "arrivals": info["arrivals"].tolist(),
            "clipped": info["orders_clipped"].tolist(),
            "price": info["price"].tolist(),
            "on_hand": env.inventory.on_hand.tolist(),
            "pipeline_out": env.pipeline.outstanding().tolist(),
            "mu": None if sim._last_mu is None else sim._last_mu.tolist(),
            "reward": float(r),
            "reward_terms": info["reward_terms"],
        })
        if trunc:
            break

    demand_arr = np.array([d["demand"] for d in days_out])
    sales_arr = np.array([d["sales"] for d in days_out])
    total_demand = float(demand_arr.sum())
    total_sales = float(sales_arr.sum())
    rewards = np.array([d["reward"] for d in days_out])
    rt = days_out[0]["reward_terms"] is not None

    return {
        "meta": {
            "policy": policy_name, "params": params,
            "elasticity": float(elasticity), "shock_std": float(shock_std),
            "capacity": float(capacity), "lead": [int(lead_min), int(lead_max)],
            "price_mode": price_mode, "weekly_multiplier": float(weekly_multiplier),
            "days": int(days), "seed": int(seed),
        },
        "skus": [s["series_id"] for s in SKUS],
        "base_prices": BASE_PRICES.tolist(),
        "unit_costs": UNIT_COSTS.tolist(),
        "days": days_out,
        "summary": {
            "profit_per_day": float(rewards.mean()),
            "profit_total": float(rewards.sum()),
            "fill_rate": (total_sales / total_demand) if total_demand > 0 else 1.0,
            "total_demand": total_demand,
            "total_sales": total_sales,
            "total_lost": total_demand - total_sales,
            "avg_inventory": float(np.mean([np.sum(d["on_hand"]) for d in days_out])),
            "clipped_units": float(sum(sum(d["clipped"]) for d in days_out)),
            "capacity_violation_events": int(env.inventory.capacity_violation_events),
            "revenue_total": float(sum(d["reward_terms"]["revenue"] for d in days_out)) if rt else 0.0,
            "ordering_total": float(sum(d["reward_terms"]["ordering_cost"] for d in days_out)) if rt else 0.0,
            "holding_total": float(sum(d["reward_terms"]["holding_cost"] for d in days_out)) if rt else 0.0,
            "stockout_total": float(sum(d["reward_terms"]["stockout_cost"] for d in days_out)) if rt else 0.0,
            "price_change_total": float(sum(d["reward_terms"]["price_change_cost"] for d in days_out)) if rt else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Pure demand-only exploration (P1 simulator standalone, no inventory)
# ---------------------------------------------------------------------------

def demand_curve(
    sku_index: int = 0,
    elasticity: float = -1.5,
    shock_std: float = 0.10,
    n_days: int = 30,
    seed: int = 0,
) -> dict:
    sim = CalibratedDemandSimulator(
        elasticity=float(elasticity), shock_std=float(shock_std),
        rng=np.random.default_rng(seed), start_day_index=1,
    )
    p = BASE_PRICES[sku_index]
    curve = []
    for m in CFG.pricing.multipliers:
        prices = BASE_PRICES.copy()
        prices[sku_index] = p * m
        sim.reset()
        sim.set_current_price(prices)
        mus = []        # deterministic per-day mean (calendar-driven, no noise)
        samples = []    # actual NB draws (what the env would realize)
        for day in range(1, int(n_days) + 1):
            samples.append(sim(day)[sku_index])
            mus.append(sim.current_mu()[sku_index])
        curve.append({
            "multiplier": float(m),
            "price": float(p * m),
            "mean_demand": float(np.mean(mus)),      # smooth curve (elasticity)
            "sampled_mean": float(np.mean(samples)),  # noisy NB realization
            "std_demand": float(np.std(samples)),
        })
    return {
        "sku": SKUS[sku_index]["series_id"],
        "base_price": float(p),
        "elasticity": float(elasticity),
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def _json_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "RLDemo/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code: int, obj):
        body = _json_bytes(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._serve_file(REPO_ROOT / "demo" / "index.html", "text/html")
        if path == "/api/info":
            return self._info()
        if path == "/api/calibration":
            return self._send(200, calibration_snapshot())
        return self._error(404, f"unknown GET path {path}")

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/run":
                return self._send(200, run_episode(**_coerce_run(body)))
            if self.path == "/api/demand_curve":
                return self._send(200, demand_curve(
                    sku_index=int(body.get("sku_index", 0)),
                    elasticity=float(body.get("elasticity", -1.5)),
                    shock_std=float(body.get("shock_std", 0.10)),
                    n_days=int(body.get("n_days", 30)),
                    seed=int(body.get("seed", 0)),
                ))
            return self._error(404, f"unknown POST path {self.path}")
        except Exception as exc:  # surface the real error to the UI
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def _info(self):
        return self._send(200, {
            "skus": SKUS,
            "locked": {
                "n_skus": CFG.env.n_skus,
                "capacity_definition": CFG.env.capacity_definition,
                "lost_sales": CFG.env.lost_sales,
                "pricing_multipliers": list(CFG.pricing.multipliers),
                "order_grid": f"0..{CFG.replenishment.q_max} step {CFG.replenishment.order_step}",
                "lead_time": f"truncnormal [{CFG.lead_time.min_days},{CFG.lead_time.max_days}] days",
                "base_elasticity": CFG.demand.base_elasticity,
                "gamma": CFG.rl.gamma,
                "reward": {
                    "holding_cost_per_unit": CFG.reward.holding_cost_per_unit,
                    "stockout_cost_per_unit": CFG.reward.stockout_cost_per_unit,
                    "price_change_cost": CFG.reward.price_change_cost,
                },
            },
            "policies": list(POLICIES),
        })

    def _serve_file(self, path: Path, ctype: str):
        if not path.exists():
            return self._error(404, f"missing file: {path.name}")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def calibration_snapshot() -> dict:
    """Per-SKU calibration stats + residual correlation, straight from the
    P1 artifact (nb_params.json) — displayed on the Data & Calibration tab."""
    from src.demand.calibration import load_nb_params

    params = load_nb_params()
    short = lambda sid: sid.replace("CA_3_", "")
    skus = []
    for sid in params["series_order"]:
        s = params["skus"][sid]
        v = s["validation"]
        skus.append({
            "series": short(sid),
            "beta_p": float(s["params"]["log_price_rel"]),
            "phi": float(s["phi"]),
            "mae": float(v["mae_test"]),
            "y_mean": float(v["y_test_mean"]),
            "mu_mean": float(v["mu_test_mean"]),
            "ratio": float(v["mean_ratio_test"]),
        })
    rc = params["residual_correlation"]
    return {
        "skus": skus,
        "series": [short(sid) for sid in params["series_order"]],
        "corr_matrix": rc["matrix"],
        "corr_note": (
            f"Pearson residuals across the {N} SKUs on common training days: "
            f"mean off-diagonal {rc['mean_offdiag']:+.3f}, max |corr| "
            f"{rc['max_abs_offdiag']:.3f} over {rc['n_common_days']} days. "
            "Green = positive co-movement, red = negative. This matrix drives "
            "the simulator's correlated latent shock (scenario S5)."
        ),
    }


def _coerce_run(body: dict) -> dict:
    return {
        "policy_name": str(body.get("policy", "sS")),
        "params": dict(body.get("params", {})),
        "elasticity": float(body.get("elasticity", -1.5)),
        "shock_std": float(body.get("shock_std", 0.10)),
        "capacity": float(body.get("capacity", 500.0)),
        "lead_min": int(body.get("lead_min", 2)),
        "lead_max": int(body.get("lead_max", 7)),
        "price_mode": str(body.get("price_mode", "base")),
        "weekly_multiplier": float(body.get("weekly_multiplier", 1.0)),
        "days": int(body.get("days", 112)),
        "seed": int(body.get("seed", 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Demo UI  ->  http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
