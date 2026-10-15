"""实验报告输出（模块 E）：CSV + Markdown。

Markdown 固定五小节：
1) 概览表（每格均值：压缩率 / 四项损坏计数 / 账单差）
2) 关键发现（"省 token 反涨价"格子、各策略主导损坏类型）
3) 缓存假设清单（CacheAssumptions 逐条 + A1-A7 全文——防止数字被误读为真实账单）
4) 价格表与来源（真实价格行来源原文）
5) 复现方法（命令、seed 派生规则；token 口径标注）
"""
import csv
import json
from pathlib import Path

from .runner import ExperimentResult


# 缓存建模假设清单（"诚实建模"的载体，随报告原文输出）
CACHE_ASSUMPTION_NOTES = [
    ("A1", "口径假设：全项目 token 用 chars/4 估算，非真实 tokenizer；"
           "中文内容建议 chars_per_token 调至 1.5~2 重跑做敏感性对比。"),
    ("A2", "缓存对象假设：每轮请求整条 prompt 打缓存标记；只有完整断点段可写入/命中，"
           "低于 min_cache 的片段按新输入计。"),
    ("A3", "前缀语义假设：默认单断点严格前缀（悲观下界）——历史任何改写使从首个差异点起"
           "全部内容失效、以 cache_write 价（≈新 token 价）重写；多断点（乐观上界）——"
           "内容未变的断点段即使位置移动也可命中。模型不声称复刻真实缓存，"
           "只保证变化方向与量级对比'结构对'。"),
    ("A4", "压缩时点与节奏：压缩只作用于历史轮（默认不动常驻前缀与当前轮）；"
           "one-shot 与 every_turn 两种节奏分别可跑。"),
    ("A5", "重放成本：工具链损坏后 agent 是否真的重放无法离线确知，默认不计入账单；"
           "include_replay_cost=True 时输出推测上界并标注'非实测'。"),
    ("A6", "输出口径：压缩不改变本轮 output（压缩对象是历史）。"),
    ("A7", "模型不含真实计费细节（最低消费、长上下文附加费等），所有美元数字是"
           "'模型估算值'，只用于策略间相对对比，绝对数字不等于真实账单。"),
]


def write_csv(result: ExperimentResult, path: str) -> str:
    """encoding=utf-8、newline=""（Windows 防 \r\r）。"""
    fieldnames = [
        "task_id", "strategy", "repeat", "seed",
        "tokens_before", "tokens_after", "compression_rate_pct", "tool_tokens_removed",
        "damage_constraint", "damage_negation", "damage_toolchain", "damage_numeric",
        "base_new_input", "base_cache_write", "base_cache_read", "base_output", "base_cost_usd",
        "comp_new_input", "comp_cache_write", "comp_cache_read", "comp_output", "comp_cost_usd",
        "cost_delta_usd", "cost_delta_pct", "replay_cost_upper_usd",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in result.rows:
            w.writerow({
                "task_id": r.task_id, "strategy": r.strategy,
                "repeat": r.repeat, "seed": r.seed,
                "tokens_before": r.tokens_before, "tokens_after": r.tokens_after,
                "compression_rate_pct": r.compression_rate_pct,
                "tool_tokens_removed": r.tool_tokens_removed,
                "damage_constraint": r.damage_constraint,
                "damage_negation": r.damage_negation,
                "damage_toolchain": r.damage_toolchain,
                "damage_numeric": r.damage_numeric,
                "base_new_input": r.baseline_tiers.new_input,
                "base_cache_write": r.baseline_tiers.cache_write,
                "base_cache_read": r.baseline_tiers.cache_read,
                "base_output": r.baseline_tiers.output,
                "base_cost_usd": r.cost_baseline_usd,
                "comp_new_input": r.compressed_tiers.new_input,
                "comp_cache_write": r.compressed_tiers.cache_write,
                "comp_cache_read": r.compressed_tiers.cache_read,
                "comp_output": r.compressed_tiers.output,
                "comp_cost_usd": r.cost_compressed_usd,
                "cost_delta_usd": r.cost_delta_usd,
                "cost_delta_pct": r.cost_delta_pct,
                "replay_cost_upper_usd": r.replay_cost_upper_usd,
            })
    return path


def render_markdown(result: ExperimentResult, price_row, strategy_names: dict | None = None) -> str:
    cfg = result.config
    summary = result.summary()
    strategy_names = strategy_names or {}

    lines = ["# 上下文压缩对照实验报告", ""]
    lines.append("> 所有 token 数为 chars/4 估算口径，所有美元数为缓存模型估算值（假设清单见第 3 节），"
                 "只用于策略间相对对比，不等于真实账单。")
    lines.append("")

    # 1) 概览表
    lines.append("## 1. 概览（每格均值）")
    lines.append("")
    lines.append("| 任务 | 策略 | 压缩率 | 约束损坏 | 否定词损坏 | 工具链损坏 | 数值损坏 | 账单差(USD) | 账单差% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, v in summary["per_cell"].items():
        task, strategy = key.split("|")
        lines.append(
            f"| {task} | {strategy} | {v['compression_rate_pct_mean']:.1f}% | "
            f"{v['damage_constraint_mean']:.1f} | {v['damage_negation_mean']:.1f} | "
            f"{v['damage_toolchain_mean']:.1f} | {v['damage_numeric_mean']:.1f} | "
            f"{v['cost_delta_usd_mean']:+.6f} | {v['cost_delta_pct_mean']:+.1f}% |")
    lines.append("")

    # 2) 关键发现
    lines.append("## 2. 关键发现")
    lines.append("")
    up = summary["cells_token_saved_cost_up"]
    if up:
        lines.append(f"**省 token 反涨价**：{len(up)} 个任务×策略组合压缩率 > 0 且账单反而上涨：")
        for task, strategy in up:
            lines.append(f"- {task} × {strategy}")
    else:
        lines.append("**省 token 反涨价**：本次矩阵未出现（可调大压缩率或换 every_turn 节奏复现）。")
    lines.append("")
    lines.append("**各策略主导损坏类型**（该策略在全部任务上的平均损坏计数最大值）：")
    for strategy, damage in summary["dominant_damage_by_strategy"].items():
        lines.append(f"- {strategy}: {damage}")
    if result.warnings:
        lines.append("")
        lines.append("**警告**：")
        for w in result.warnings:
            lines.append(f"- {w}")
    lines.append("")

    # 3) 缓存假设清单
    lines.append("## 3. 缓存建模假设清单（诚实建模，逐条原文）")
    lines.append("")
    lines.append("| 编号 | 假设 |")
    lines.append("|---|---|")
    for code, note in CACHE_ASSUMPTION_NOTES:
        lines.append(f"| {code} | {note} |")
    ca = cfg["cache"]
    lines.append("")
    lines.append("本次运行的参数："
                 f"chars_per_token={ca['chars_per_token']}、"
                 f"breakpoint_tokens={ca['breakpoint_tokens']}、"
                 f"min_cache_tokens={ca['min_cache_tokens']}、"
                 f"multi_breakpoint={ca['multi_breakpoint']}、"
                 f"every_turn={cfg['every_turn']}。")
    lines.append("")

    # 4) 价格表与来源
    lines.append("## 4. 价格表与来源")
    lines.append("")
    lines.append(f"- 计价模型：{cfg['price_model']}")
    lines.append(f"- 价格行（USD/百万 token）：输入 {price_row.input} / "
                 f"缓存写5m {price_row.cache_write_5m} / 缓存写1h {price_row.cache_write_1h} / "
                 f"缓存读 {price_row.cache_read} / 输出 {price_row.output}")
    lines.append(f"- 来源：{price_row.source}")
    if price_row.note:
        lines.append(f"- 备注：{price_row.note}")
    lines.append("")

    # 5) 复现方法
    lines.append("## 5. 复现方法")
    lines.append("")
    lines.append("```")
    lines.append("python -m ctxautopsy.experiment            # 默认演示矩阵（3 任务 × 5 策略 × 3 重复）")
    lines.append("python -m ctxautopsy.experiment --config config.json --out-dir results")
    lines.append("```")
    lines.append(f"- 矩阵：{len(cfg['tasks'])} 任务 × {len(cfg['strategies'])} 策略 × {cfg['repeats']} 重复"
                 f" = {len(result.rows)} 格")
    lines.append(f"- seed 派生：crc32(\"task|strategy|repeat\") % 2**31，全链路确定性，任何机器可复现")
    lines.append(f"- 压缩时点：第 {cfg['compress_from_turn']} 轮起，every_turn={cfg['every_turn']}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: ExperimentResult, price_row, out_dir: str) -> tuple[str, str]:
    """写 metrics.csv、report.md、config.json，返回 (csv_path, md_path)。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(result, str(out / "metrics.csv"))
    md_path = out / "report.md"
    md_path.write_text(render_markdown(result, price_row), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(result.config, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, str(md_path)


def main(argv: list[str] | None = None) -> int:
    """python -m ctxautopsy.experiment：默认演示矩阵；--config x.json 自定义。"""
    import argparse

    from ..billing import DEFAULT_PRICES
    from .runner import ExperimentConfig, run_experiment

    ap = argparse.ArgumentParser(prog="ctxautopsy experiment", description="上下文压缩对照实验")
    ap.add_argument("--config", default=None, help="实验配置 JSON")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = ExperimentConfig()
    if args.config:
        cfg = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.out_dir:
        cfg.output_dir = args.out_dir

    result = run_experiment(cfg)
    price_row = DEFAULT_PRICES.rows[cfg.price_model]
    csv_path, md_path = write_outputs(result, price_row, cfg.output_dir)
    print(f"实验完成：{len(result.rows)} 格 → {csv_path} / {md_path}")
    return 0
