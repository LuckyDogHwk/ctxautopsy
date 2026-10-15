"""实验编排（模块 E）：任务 × 策略 × 重复，离线全确定性。

每格流程：
1) generate_task(task_id, seed) → 合成任务
2) history_before = 压缩时点前的历史块（不含常驻、不含当前轮）
3) history_after = baseline 恒等 或 模块 C 策略压缩
4) 模块 D 跑四项损坏检查（输入只含 before/after 历史块 + 任务约束注解）
5) simulate_turns 分别跑基准/压缩两场景得四档账单（模块 B 价格表计价）
6) compute_metrics → RunMetrics 入 rows

seed 派生：crc32(f"{task_id}|{strategy}|{repeat}") % 2**31（非负、互异）。
策略为确定性函数：重复次数不影响单次结果（每次 repeat 用不同 seed 实例化数据）。
"""
from dataclasses import dataclass, field
import statistics
import zlib

from ..checkers import run_all_checkers
from ..compressors import create_compressor, list_compressors
from ..schema import estimate_tokens
from .cache_model import CacheAssumptions, CompressPlan, TierUsage, simulate_turns
from .metrics import RunMetrics, compute_metrics
from .synthetic import SyntheticTask, generate_task


@dataclass
class StrategySpec:
    name: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params}


@dataclass
class ExperimentConfig:
    tasks: list = field(default_factory=lambda: ["t_short", "t_tool_heavy", "t_numeric"])
    strategies: list = field(default_factory=lambda: [
        StrategySpec("baseline", {}),
        StrategySpec("truncate_head", {"keep_ratio": 0.5}),
        # max_tokens=200 = 恰好一个工具结果的大小（800 字符/4）：
        # 窗口保留最后一个结果却丢掉了它的调用 → 孤儿结果，工具链截断
        StrategySpec("sliding_window", {"max_tokens": 200}),
        StrategySpec("freq_drop", {"mode": "stopwords", "top_ratio": 0.2}),
        StrategySpec("rule_summary", {"min_chars": 120, "apply_to": ["tool_result"]}),
    ])
    repeats: int = 3
    compress_from_turn: int = 3
    every_turn: bool = False
    cache: CacheAssumptions = field(default_factory=CacheAssumptions)
    include_replay_cost: bool = False
    output_dir: str = "experiments/experiment_results"
    price_model: str = "deepseek-v4-flash"      # 计价模型（模块 B 价格表的键）

    def to_dict(self) -> dict:
        return {
            "tasks": self.tasks, "strategies": [s.to_dict() for s in self.strategies],
            "repeats": self.repeats, "compress_from_turn": self.compress_from_turn,
            "every_turn": self.every_turn, "cache": self.cache.to_dict(),
            "include_replay_cost": self.include_replay_cost,
            "output_dir": self.output_dir, "price_model": self.price_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        cache = CacheAssumptions(**d.get("cache", {}))
        return cls(
            tasks=d.get("tasks", ["t_short", "t_tool_heavy", "t_numeric"]),
            strategies=[StrategySpec(s["name"], s.get("params", {}))
                        for s in d.get("strategies", [])],
            repeats=d.get("repeats", 3),
            compress_from_turn=d.get("compress_from_turn", 3),
            every_turn=d.get("every_turn", False),
            cache=cache,
            include_replay_cost=d.get("include_replay_cost", False),
            output_dir=d.get("output_dir", "experiment_results"),
            price_model=d.get("price_model", "deepseek-v4-flash"),
        )


@dataclass
class ExperimentResult:
    config: dict
    rows: list
    warnings: list = field(default_factory=list)

    def summary(self) -> dict:
        """按 (task, strategy) 聚合均值。"""
        cells: dict = {}
        for r in self.rows:
            key = f"{r.task_id}|{r.strategy}"
            cells.setdefault(key, []).append(r)
        per_cell = {}
        for key, rs in cells.items():
            def mean(f):
                return round(statistics.mean(f(x) for x in rs), 4)
            per_cell[key] = {
                "n": len(rs),
                "compression_rate_pct_mean": mean(lambda x: x.compression_rate_pct),
                "damage_constraint_mean": mean(lambda x: x.damage_constraint),
                "damage_negation_mean": mean(lambda x: x.damage_negation),
                "damage_toolchain_mean": mean(lambda x: x.damage_toolchain),
                "damage_numeric_mean": mean(lambda x: x.damage_numeric),
                "cost_delta_usd_mean": mean(lambda x: x.cost_delta_usd),
                "cost_delta_pct_mean": mean(lambda x: x.cost_delta_pct),
            }
        # 关键发现：省 token 反涨价 / 各策略主导损坏
        cells_token_saved_cost_up = [
            key.split("|") for key, v in per_cell.items()
            if v["compression_rate_pct_mean"] > 0 and v["cost_delta_pct_mean"] > 0
        ]
        dominant = {}
        for key, v in per_cell.items():
            strategy = key.split("|")[1]
            damages = {"constraint": v["damage_constraint_mean"],
                       "negation": v["damage_negation_mean"],
                       "toolchain": v["damage_toolchain_mean"],
                       "numeric": v["damage_numeric_mean"]}
            top = max(damages, key=damages.get)
            if damages[top] > 0:
                dominant[strategy] = top
        return {
            "per_cell": per_cell,
            "cells_token_saved_cost_up": cells_token_saved_cost_up,
            "dominant_damage_by_strategy": dominant,
        }

    def to_csv_rows(self) -> list[dict]:
        return [r.to_dict() for r in self.rows]


def _compress_history(strategy: str, params: dict, history):
    if strategy == "baseline":
        return list(history)
    comp = create_compressor(strategy, params)
    return list(comp.compress(history).messages)


def _total_cost(turns) -> float:
    return sum(t.cost_usd for t in turns)


def _sum_tiers(turns) -> TierUsage:
    total = TierUsage()
    for t in turns:
        total = total + t.tiers
    return total


def run_experiment(cfg: ExperimentConfig, price_row=None,
                   task_factory=None) -> ExperimentResult:
    """主入口：离线、全确定性。price_row 缺省用模块 B 默认价格表的 price_model 行。"""
    if price_row is None:
        from ..billing import DEFAULT_PRICES
        price_row = DEFAULT_PRICES.rows[cfg.price_model]
    task_factory = task_factory or generate_task

    names = [s.name for s in cfg.strategies]
    if "baseline" not in names:
        raise ValueError(f"strategies 必须包含 baseline，收到: {names}")
    unknown = [n for n in names if n != "baseline" and n not in list_compressors()]
    if unknown:
        raise ValueError(f"未知策略名: {unknown}，合法策略: baseline / {list_compressors()}")

    rows: list[RunMetrics] = []
    warnings: list[str] = []
    for task_id in cfg.tasks:
        for spec in cfg.strategies:
            for repeat in range(cfg.repeats):
                seed = zlib.crc32(f"{task_id}|{spec.name}|{repeat}".encode()) % 2**31
                task: SyntheticTask = task_factory(task_id, seed)

                if cfg.compress_from_turn > task.cfg.n_turns:
                    warnings.append(f"{task_id}: compress_from_turn 超过轮数，压缩未生效")

                # 压缩时点前的历史块（不含常驻、不含当前轮）
                history_before = []
                for turn_blocks in task.turns[:cfg.compress_from_turn - 1]:
                    history_before.extend(turn_blocks)
                history_after = _compress_history(spec.name, spec.params, history_before)

                # 四项损坏检查（ground truth 来自任务注解）：
                # 只检查窗口内真实存在的约束——注入在压缩窗口之后的约束
                # 从未进入过被压对象，检查器看不到就不该计丢失
                constraints_in_window = [
                    c for c in task.constraints
                    if any(c in m.text for m in history_before)
                ]
                check_results = run_all_checkers(
                    history_before, history_after,
                    {"constraints": constraints_in_window},
                )

                # 两场景账单
                plan = CompressPlan(
                    compress_from_turn=cfg.compress_from_turn,
                    every_turn=cfg.every_turn, strategy=spec.name, params=spec.params,
                )
                baseline_turns = simulate_turns(
                    task, CompressPlan.no_compression(), cfg.cache, price_row,
                    None)
                compressed_turns = simulate_turns(
                    task, plan, cfg.cache, price_row,
                    None if spec.name == "baseline"
                    else (lambda h, p: _compress_history(spec.name, p, h)))

                tokens_before = sum(estimate_tokens(m.text) for m in history_before)
                tokens_after = sum(estimate_tokens(m.text) for m in history_after)
                tool_before = sum(estimate_tokens(m.text) for m in history_before
                                  if m.kind == "tool_result")
                tool_after = sum(estimate_tokens(m.text) for m in history_after
                                 if m.kind == "tool_result")

                # 重放成本上界（可选，默认不估：无法确知 agent 是否真的重放）
                replay = 0.0
                if cfg.include_replay_cost:
                    for m in history_before:
                        if m.kind == "tool_result" and not any(
                                a.uid == m.uid for a in history_after):
                            replay += (estimate_tokens(m.text) * price_row.input) / 1e6

                rows.append(compute_metrics(
                    task_id=task_id, strategy=spec.name, repeat=repeat, seed=seed,
                    tokens_before=tokens_before, tokens_after=tokens_after,
                    tool_tokens_removed=max(0, tool_before - tool_after),
                    check_results=check_results,
                    baseline_tiers=_sum_tiers(baseline_turns),
                    compressed_tiers=_sum_tiers(compressed_turns),
                    baseline_cost_usd=_total_cost(baseline_turns),
                    compressed_cost_usd=_total_cost(compressed_turns),
                    replay_cost_upper_usd=replay,
                ))

    return ExperimentResult(config=cfg.to_dict(), rows=rows, warnings=warnings)
