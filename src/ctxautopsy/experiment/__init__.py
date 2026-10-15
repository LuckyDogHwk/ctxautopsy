"""实验编排（模块 E）：合成任务 × 压缩策略 × 缓存账单的对照实验。

入口：python -m ctxautopsy.experiment（默认演示矩阵，全确定性、毫秒级）。
"""
from .cache_model import CacheAssumptions, CompressPlan, TierUsage, TurnUsage, simulate_turns
from .metrics import RunMetrics, compute_metrics
from .report import CACHE_ASSUMPTION_NOTES, render_markdown, write_csv, write_outputs, main
from .runner import ExperimentConfig, ExperimentResult, StrategySpec, run_experiment
from .synthetic import SyntheticTask, TaskConfig, ToolPair, generate_task

__all__ = [
    "CacheAssumptions", "CompressPlan", "TierUsage", "TurnUsage", "simulate_turns",
    "RunMetrics", "compute_metrics",
    "CACHE_ASSUMPTION_NOTES", "render_markdown", "write_csv", "write_outputs", "main",
    "ExperimentConfig", "ExperimentResult", "StrategySpec", "run_experiment",
    "SyntheticTask", "TaskConfig", "ToolPair", "generate_task",
]
