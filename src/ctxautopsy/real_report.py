"""真实会话数据的 Markdown 报告渲染（只含聚合数字，无任何消息文本）。"""
from .billing import DEFAULT_PRICES, PriceTable, cost_sessions
from .session_log import SessionReport, aggregate_sessions, parse_sessions_dir


def render_real_report(reports: list[SessionReport], project=None,
                       price_table: PriceTable | None = None) -> str:
    price_table = price_table or DEFAULT_PRICES
    project = project or cost_sessions(reports, price_table)
    agg = aggregate_sessions(reports)
    t = agg.totals

    lines = ["# 真实会话解剖报告（本机 ~/.claude/projects 实测）", ""]
    lines.append("> 数据：本机 Claude Code 会话日志（jsonl），全部数字为实测聚合，"
                 "不包含任何消息文本。token 为 API usage 里的权威数字；"
                 "「构成桶」一栏为启发式估算（CJK≈1 token/字，已在表内标注）。")
    lines.append("")

    lines.append("## 1. 全量概览")
    lines.append("")
    lines.append(f"- 会话文件：{len(reports)} 个，API 调用：{t.api_calls:,} 次")
    lines.append(f"- 新输入 {t.new_input_tokens / 1e6:.2f}M / 缓存写 {t.cache_write_tokens:,} "
                 f"（本机网关不报告缓存写）/ 缓存读 {t.cache_read_tokens / 1e6:.1f}M / "
                 f"输出 {t.output_tokens / 1e6:.2f}M")
    if t.input_tokens_total > 0:
        lines.append(f"- **缓存读占输入总量 {t.cache_read_tokens / t.input_tokens_total * 100:.1f}%**"
                     f"——缓存命中/未命中价差 30-50 倍，这个比例决定了成本结构")
    lines.append("")

    lines.append("## 2. 成本：官方价推算 vs harness 自报")
    lines.append("")
    lines.append(f"- 按 DeepSeek 官方价推算：**${project.total_cost:.2f}**（空闲时段价）")
    lines.append(f"- harness（Claude Code）自报累计：**${agg.total_self_reported_cost_usd:.2f}**"
                 f"（其自身标记 hasUnknownModelCost=True，口径不明）")
    if project.delta_vs_self_reported is not None:
        lines.append(f"- 差距：推算值比自报低 **{abs(project.delta_vs_self_reported) * 100:.1f}%**"
                     f"——即通过第三方网关实际付费约为官方价的 "
                     f"**{1 / (1 + project.delta_vs_self_reported):.0f} 倍**。"
                     f"这是实测差距，只报告不做归因。")
    if project.unpriced_models:
        lines.append(f"- 未定价模型（跳过）：{project.unpriced_models}")
    lines.append("")

    lines.append("## 3. 最大的 10 个会话")
    lines.append("")
    lines.append("| 会话 | 调用 | 新输入 | 缓存读 | 缓存读占比 | 常驻开销(估算) |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(reports, key=lambda x: x.api_calls, reverse=True)[:10]:
        u = r.usage
        ratio = u.cache_read_tokens / u.input_tokens_total * 100 if u.input_tokens_total else 0
        lines.append(f"| {r.source_file[:40]} | {r.api_calls:,} | "
                     f"{u.new_input_tokens / 1e6:.2f}M | {u.cache_read_tokens / 1e6:.1f}M | "
                     f"{ratio:.1f}% | {r.composition.resident_overhead_tokens:,} |")
    lines.append("")

    # 构成桶（选调用数最多的会话）
    main = max(reports, key=lambda x: x.api_calls) if reports else None
    if main:
        c = main.composition
        lines.append("## 4. 最大会话的上下文构成（估算桶）")
        lines.append("")
        lines.append(f"- 会话：{main.source_file}（{main.api_calls:,} 次调用，"
                     f"语义：{main.usage_semantics}）")
        lines.append(f"- 常驻开销：**{c.resident_overhead_tokens:,} token**（方法：{c.overhead_method}）"
                     f"——会话一启动、每次调用都重发的部分")
        lines.append(f"- 用户文本 {c.user_text_tokens:,} / 工具输出 {c.tool_output_tokens:,} / "
                     f"思考 {c.thinking_tokens:,} / 压缩摘要 {c.compact_summary_tokens:,}")
        lines.append(f"- 模型分布："
                     + "、".join(f"{k} {v.api_calls:,} 次" for k, v in main.per_model.items()))
        lines.append("")

    lines.append("## 5. 价格表（可溯源，可用 JSON 覆盖）")
    lines.append("")
    for model, row in price_table.rows.items():
        if "-peak" in model:
            continue
        lines.append(f"- {model}：输入 {row.input} / 缓存写5m {row.cache_write_5m} / "
                     f"缓存读 {row.cache_read} / 输出 {row.output}（{row.currency}/百万 token）"
                     f"——{row.source}")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```")
    lines.append("python -m ctxautopsy analyze ~/.claude/projects --out real_report.md")
    lines.append("```")
    return "\n".join(lines)


def generate_real_report(log_dir: str, out_path: str) -> str:
    """扫描真实日志 → 写 Markdown 报告，返回报告文本。"""
    from pathlib import Path
    reports = parse_sessions_dir(log_dir)
    md = render_real_report(reports)
    Path(out_path).write_text(md, encoding="utf-8")
    return md
