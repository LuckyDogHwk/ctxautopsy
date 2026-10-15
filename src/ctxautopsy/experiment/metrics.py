"""单次运行指标（模块 E）。"""
from dataclasses import dataclass, field

from .cache_model import TierUsage


@dataclass
class RunMetrics:
    task_id: str
    strategy: str
    repeat: int
    seed: int
    tokens_before: int
    tokens_after: int
    compression_rate_pct: float
    tool_tokens_removed: int
    damage_constraint: int
    damage_negation: int
    damage_toolchain: int
    damage_numeric: int
    cost_baseline_usd: float
    cost_compressed_usd: float
    cost_delta_usd: float
    cost_delta_pct: float
    baseline_tiers: TierUsage
    compressed_tiers: TierUsage
    replay_cost_upper_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "strategy": self.strategy,
            "repeat": self.repeat, "seed": self.seed,
            "tokens_before": self.tokens_before, "tokens_after": self.tokens_after,
            "compression_rate_pct": self.compression_rate_pct,
            "tool_tokens_removed": self.tool_tokens_removed,
            "damage_constraint": self.damage_constraint,
            "damage_negation": self.damage_negation,
            "damage_toolchain": self.damage_toolchain,
            "damage_numeric": self.damage_numeric,
            "cost_baseline_usd": self.cost_baseline_usd,
            "cost_compressed_usd": self.cost_compressed_usd,
            "cost_delta_usd": self.cost_delta_usd,
            "cost_delta_pct": self.cost_delta_pct,
            "baseline_tiers": self.baseline_tiers.to_dict(),
            "compressed_tiers": self.compressed_tiers.to_dict(),
            "replay_cost_upper_usd": self.replay_cost_upper_usd,
        }


def compute_metrics(task_id: str, strategy: str, repeat: int, seed: int,
                    tokens_before: int, tokens_after: int,
                    tool_tokens_removed: int,
                    check_results: dict,
                    baseline_tiers: TierUsage, compressed_tiers: TierUsage,
                    baseline_cost_usd: float, compressed_cost_usd: float,
                    replay_cost_upper_usd: float = 0.0) -> RunMetrics:
    for key in ("constraint", "negation", "toolchain", "numeric"):
        if check_results.get(key, 0) < 0:
            raise AssertionError(f"检查器 {key} 返回负数: {check_results[key]}")
    before = max(0, tokens_before)
    rate = (before - tokens_after) / before * 100 if before > 0 else 0.0
    delta = compressed_cost_usd - baseline_cost_usd
    delta_pct = delta / baseline_cost_usd * 100 if baseline_cost_usd > 0 else 0.0
    return RunMetrics(
        task_id=task_id, strategy=strategy, repeat=repeat, seed=seed,
        tokens_before=before, tokens_after=tokens_after,
        compression_rate_pct=round(rate, 2),
        tool_tokens_removed=max(0, tool_tokens_removed),
        damage_constraint=check_results.get("constraint", 0),
        damage_negation=check_results.get("negation", 0),
        damage_toolchain=check_results.get("toolchain", 0),
        damage_numeric=check_results.get("numeric", 0),
        cost_baseline_usd=round(baseline_cost_usd, 8),
        cost_compressed_usd=round(compressed_cost_usd, 8),
        cost_delta_usd=round(delta, 8),
        cost_delta_pct=round(delta_pct, 2),
        baseline_tiers=baseline_tiers,
        compressed_tiers=compressed_tiers,
        replay_cost_upper_usd=round(replay_cost_upper_usd, 6),
    )
