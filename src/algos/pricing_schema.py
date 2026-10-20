"""Explicit pricing action schema; legacy metadata is shared-scope only."""
import numpy as np


def pricing_schema(env):
    scope = getattr(env, "pricing_scope", "shared")
    if scope not in ("shared", "per_sku"):
        raise ValueError("pricing_scope must be shared or per_sku")
    multipliers = list(map(float, env.config.pricing.multipliers))
    n = int(getattr(env, "n", env.config.env.n_skus))
    expected = [len(multipliers)] * (n if scope == "per_sku" else 1)
    nvec = list(map(int, getattr(env, "pricing_nvec", expected)))
    if nvec != expected:
        raise ValueError("pricing action schema disagrees with environment scope")
    sim = getattr(env, "sim", None)
    series = getattr(sim, "series_ids", None)
    return dict(version=1, pricing_scope=scope, pricing_nvec=nvec,
                multipliers=multipliers,
                sku_order=list(series) if series is not None else list(range(n)))


def validate_checkpoint_schema(data, expected):
    actual = data.get("action_schema")
    if actual is None:
        if expected["pricing_scope"] == "shared":
            return  # historical checkpoint; original scalar pricing only
        raise ValueError("legacy checkpoint has no pricing action schema; shared scope required")
    if actual != expected:
        raise ValueError(f"incompatible pricing action schema: checkpoint {actual}, expected {expected}")


def pricing_action(action, schema):
    action = np.asarray(action)
    if action.shape != (len(schema["pricing_nvec"]),):
        raise ValueError("pricing actor action shape disagrees with saved schema")
    if schema["pricing_scope"] == "shared":
        return int(action[0])
    return action.astype(np.int64, copy=True)
