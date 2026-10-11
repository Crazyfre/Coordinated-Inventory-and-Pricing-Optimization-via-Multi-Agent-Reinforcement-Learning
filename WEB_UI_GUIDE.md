# Web Demo UI — Team 10

The demo runs the real replenishment environment and M5-calibrated demand simulator. The policies are heuristics; PPO and coordinated MAPPO agents have not been trained yet.

## Start the demo

From D:\projects\RL_Project:

```powershell
.venv/Scripts/python demo/server.py
```

Open **http://127.0.0.1:8000**. Stop the server with Ctrl+C. For a different port, use `--port 8080`. No frontend packages, network fonts or new Python dependencies are needed. The local M5 data/cache and calibration artifact must be available.

## A simple presentation walkthrough

1. **Project overview.** Explain the three steps: set prices, restock, measure the outcome. Point to the completed foundation, R1, R2 and P1 modules. Explain that the weekly pricing and daily replenishment agents are the next stage. Expand the data evidence for the selected products, held-out NB calibration and correlation matrix.
2. **Store simulation.** Use **Default store**, then **Run simulation**. Explain economic reward per day, customer demand served, and lost sales. The demand-versus-sales chart makes shortages visible; the stock-versus-capacity chart shows shared storage.
3. **Compare rules.** Run the three heuristics under the same scenario and seed. Their demand streams are identical when price and demand settings match. This is an illustrative comparison in the implemented simulator, not evidence that RL outperforms a baseline.
4. **Less storage.** Select the preset and run again. Examine lost sales and cancelled inbound units. A preset changes settings; it does not run automatically. Old results remain visible with a settings-changed notice until rerun.
5. **20% higher prices.** Run this preset to explore the margin-versus-demand trade-off. All products use the same chosen multiplier, applied at weekly decision points and held between them. This is a manually selected price, not a learned pricing policy.
6. **Price & demand.** Pick a product and explore its response. Green is expected daily demand; blue is the sampled NB mean. Expected demand falls as price rises under the controlled negative elasticity; random samples can vary around that trend.

## Controls and results

The main controls are the restocking rule, shared storage capacity, and selling price. **Advanced settings** contains reorder thresholds (7/12 days of average demand by default), the fixed daily order size (1 day of average demand by default), controlled elasticity, correlated-shock strength, delivery bounds, simulation length and seed. The UI restricts delivery bounds to the implemented 2–7 day range; equal bounds provide a fixed lead time.

Reordering uses stock on hand plus stock in transit. Quantities are snapped to the locked order grid and passed as grid indices. The daily ordering rule is a heuristic, not an RL actor.

**Daily reward, costs and product results** contains the reward chart, all cost components, average stock, receiving events that clipped arrivals, and per-product demand, sales, service and final stock. The backend key `capacity_violation_events` counts receiving calls that required clipping; it does not mean accepted stock exceeded capacity. The UI names this accurately.

The price-response view uses the locked five price multipliers returned by the backend. Sampling settings expose shock strength, days per price point and seed. Its results also show a notice when settings change.

## Assumptions to state clearly

- M5 supplies sales, price and calendar data. Inventory, capacity, supply delays and operational costs are simulated.
- Procurement cost in this demo is **60% of each SKU's base price**. Final cost calibration is pending.
- Economic reward subtracts procurement, holding, stockout and price-change costs from revenue. It includes modeled penalties, so describe it as economic reward rather than accounting profit.
- Price elasticity is controlled. Observational fitted price coefficients are predictive associations, not causal estimates.
- Correlated shocks use the residual correlation matrix. NB sampling remains random when shared shock strength is zero.
- Excess arrivals are cancelled proportionally at receiving; procurement costs apply to delivered units.
- The current demand forecast is a recent-mean stub. RL training and final MAPPO integration remain future work.

## Files and API

- `demo/index.html`: responsive three-view UI, no framework or external assets.
- `demo/server.py`: existing Python stdlib backend, unchanged by this refactor.
- `GET /api/info`, `GET /api/calibration`, `POST /api/run`, `POST /api/demand_curve`.

## Verification for this refactor

113 project tests passed. All existing endpoints were exercised; paired replenishment runs retained identical demand paths, end-of-day stock stayed within capacity, and expected demand decreased across the price grid. Frontend syntax and handlers were checked with a DOM harness against the real API (run, comparison, price curve, controls, navigation and stale results). Browser rendering was not verified because no browser surface was available in this session.

Before presenting, open the demo in Chrome and check the desktop and narrow-window layout once.
