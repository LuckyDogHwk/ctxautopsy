"""四档计费模型（模块 B）。

核心命题：省 token ≠ 省钱。缓存读比未命中便宜 10-50 倍，
压缩动作破坏缓存前缀时，后续轮次全部按未命中价重算。

成本公式（每百万 token 计价，价格表可配置）：
    成本 = 新输入×Pi + 缓存写5m×P5m + 缓存写1h×P1h + 缓存读×Pr + 输出×Po

内置价格表 DEFAULT_PRICES（全部可溯源，禁止编造）：
- Claude 系：Anthropic 官方定价（claude-api 技能内置价格快照 2026-10-06）。
  claude-sonnet-5-5：输入 $2.00 / 输出 $10.00 / 缓存读 $0.20；
  claude-opus-5-5：输入 $4.00 / 输出 $20.00 / 缓存读 $0.20；
  claude-sonnet-5 同 sonnet-5-5；claude-fable-5-1：$10.00 / $50.00 / 读 $0.25。
  缓存写 5m = 1.25×输入价、1h = 2×输入价（官方 prompt caching 定价惯例）。
- DeepSeek 系：官方定价页（2026-09-10 起峰谷分时，空闲档 USD）
  https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
  v4-pro：命中 $0.022 / 未命中 $0.66 / 输出 $1.98；
  flash：命中 $0.003 / 未命中 $0.15 / 输出 $0.60。
  高峰（北京时间工作日 9:00-12:00、14:00-18:00）×2，见 is_peak_beijing。
  DeepSeek 无 5m/1h 缓存写分层，两档同价（=未命中价）。

用户覆盖价格表（优先级从高到低）：
1. load_price_table(path) 直接传 JSON 路径；
2. 环境变量 CTXAUTOPSY_PRICES 指向 JSON 文件；
3. 不传 → 内置 DEFAULT_PRICES。
JSON 按模型名整行替换默认行，未覆盖的行保留；结构非法报中文 ValueError。

未知模型处理：不编价格。无价格行的模型返回 priced=False、
total_cost=None（CtxAutopsy 是验尸官，猜价格等于伪造账单）。

交叉验证：cost_sessions 对比推算成本与 harness 自报 totalCostUSD，
输出 delta_vs_self_reported = (推算-自报)/自报。仅当全部计价为 USD
（或提供 fx_cny_to_usd 汇率换算 CNY）时才计算，避免跨币种假比较。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .session_log import SessionReport, UsageTiers


# ---------------------------------------------------------------- 价格表 ----

@dataclass(frozen=True)
class PriceRow:
    """一个模型的价格行（每百万 token）。source 必填，禁止无来源数字进默认表。"""
    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float
    currency: str = "USD"
    source: str = ""
    note: str = ""


@dataclass(frozen=True)
class PriceTable:
    default_model: str
    rows: dict


_ANTHROPIC_SRC = "Anthropic 官方定价（claude-api 技能内置价格快照 2026-10-06）；缓存写 5m=1.25×、1h=2×输入价"
_DEEPSEEK_SRC = "DeepSeek 官方定价页（空闲时段，2026-09-10 起峰谷分时）https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
_DEEPSEEK_PEAK_NOTE = "高峰时段×2（北京时间工作日 9-12、14-18）；无 5m/1h 缓存写分层，两档同价"

DEFAULT_PRICES = PriceTable(
    default_model="claude-sonnet-5-5",
    rows={
        "claude-sonnet-5-5": PriceRow(
            input=2.00, cache_write_5m=2.50, cache_write_1h=4.00,
            cache_read=0.20, output=10.00,
            source=_ANTHROPIC_SRC, note="当前 Sonnet"),
        "claude-sonnet-5": PriceRow(
            input=2.00, cache_write_5m=2.50, cache_write_1h=4.00,
            cache_read=0.20, output=10.00,
            source=_ANTHROPIC_SRC, note="上一代 Sonnet，同价"),
        "claude-opus-5-5": PriceRow(
            input=4.00, cache_write_5m=5.00, cache_write_1h=8.00,
            cache_read=0.20, output=20.00,
            source=_ANTHROPIC_SRC, note="当前 Opus，缓存读 $0.20"),
        "claude-fable-5-1": PriceRow(
            input=10.00, cache_write_5m=12.50, cache_write_1h=20.00,
            cache_read=0.25, output=50.00,
            source=_ANTHROPIC_SRC, note="Fable 5.1，缓存读 $0.25"),
        "deepseek-v4-pro": PriceRow(
            input=0.66, cache_write_5m=0.66, cache_write_1h=0.66,
            cache_read=0.022, output=1.98,
            source=_DEEPSEEK_SRC, note=_DEEPSEEK_PEAK_NOTE),
        "deepseek-v4-pro[1m]": PriceRow(
            input=0.66, cache_write_5m=0.66, cache_write_1h=0.66,
            cache_read=0.022, output=1.98,
            source=_DEEPSEEK_SRC, note=_DEEPSEEK_PEAK_NOTE),
        "deepseek-v4-flash": PriceRow(
            input=0.15, cache_write_5m=0.15, cache_write_1h=0.15,
            cache_read=0.003, output=0.60,
            source=_DEEPSEEK_SRC, note=_DEEPSEEK_PEAK_NOTE),
        "deepseek-v4.1-flash": PriceRow(
            input=0.15, cache_write_5m=0.15, cache_write_1h=0.15,
            cache_read=0.003, output=0.60,
            source=_DEEPSEEK_SRC, note=_DEEPSEEK_PEAK_NOTE),
        # 高峰参考行（仅供 compute_cost_range 做成本区间，不用于默认计价）
        "deepseek-v4-pro-peak": PriceRow(
            input=1.32, cache_write_5m=1.32, cache_write_1h=1.32,
            cache_read=0.044, output=3.96,
            source=_DEEPSEEK_SRC, note="高峰时段参考价（×2）"),
        "deepseek-v4-flash-peak": PriceRow(
            input=0.30, cache_write_5m=0.30, cache_write_1h=0.30,
            cache_read=0.006, output=1.20,
            source=_DEEPSEEK_SRC, note="高峰时段参考价（×2）"),
    },
)


def load_price_table(path: str | Path | None = None) -> PriceTable:
    """加载价格表：path > 环境变量 CTXAUTOPSY_PRICES > 内置默认表。

    用户 JSON 按模型名整行替换默认行，未覆盖的行保留。
    格式：
    {
      "default_model": "claude-sonnet-5-5",
      "rows": {"模型名": {"input": 2.0, "cache_write_5m": 2.5, "cache_write_1h": 4.0,
                          "cache_read": 0.2, "output": 10.0,
                          "currency": "USD", "source": "...", "note": ""}}
    }
    """
    if path is None:
        env = os.environ.get("CTXAUTOPSY_PRICES")
        if not env:
            return DEFAULT_PRICES
        path = env
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"价格表文件 {path} 不是合法 JSON: {e}") from e
    except OSError as e:
        raise ValueError(f"价格表文件 {path} 无法读取: {e}") from e

    rows = dict(DEFAULT_PRICES.rows)
    rows_raw = data.get("rows")
    if not isinstance(rows_raw, dict):
        raise ValueError("价格表 JSON 需要顶层 \"rows\" 对象")
    for model, spec in rows_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"模型 {model} 的价格必须是对象")
        need = ("input", "cache_write_5m", "cache_write_1h", "cache_read", "output")
        missing = [k for k in need if k not in spec]
        if missing:
            raise ValueError(f"模型 {model} 的价格缺字段: {missing}")
        rows[model] = PriceRow(
            input=float(spec["input"]),
            cache_write_5m=float(spec["cache_write_5m"]),
            cache_write_1h=float(spec["cache_write_1h"]),
            cache_read=float(spec["cache_read"]),
            output=float(spec["output"]),
            currency=str(spec.get("currency", "USD")),
            source=str(spec.get("source", "用户自定义（未标注来源）")),
            note=str(spec.get("note", "")),
        )
    return PriceTable(
        default_model=str(data.get("default_model", DEFAULT_PRICES.default_model)),
        rows=rows,
    )


def is_peak_beijing(dt: datetime) -> bool:
    """北京时间是否处于高峰时段。

    高峰：周一至周五 09:00-12:00、14:00-18:00（即 UTC 01:00-04:00、06:00-10:00）。
    限制：未内置中国法定节假日日历（调休与假期按普通工作日处理），文档已声明。
    """
    if dt.weekday() >= 5:  # 周六日全天空闲
        return False
    h = dt.hour
    return (9 <= h < 12) or (14 <= h < 18)


# ---------------------------------------------------------------- 成本计算 ----

@dataclass(frozen=True)
class TierCost:
    tier: str          # new_input / cache_write / cache_read / output
    tokens: int
    cost: float


@dataclass(frozen=True)
class CostBreakdown:
    """五档成本明细（compute_cost 返回，供峰谷区间等场景使用）。"""
    new_input_cost: float
    cache_write_cost: float
    cache_read_cost: float
    output_cost: float

    @property
    def total(self) -> float:
        return (self.new_input_cost + self.cache_write_cost
                + self.cache_read_cost + self.output_cost)


def compute_cost(usage: UsageTiers, row: PriceRow) -> CostBreakdown:
    """按价格行计价（含 5m/1h 分档）。token → 计价：数量 / 1e6 × 单价。"""
    m = 1_000_000
    w5m, w1h = usage.cache_write_5m_tokens, usage.cache_write_1h_tokens
    remainder = max(0, usage.cache_write_tokens - w5m - w1h)
    if usage.cache_write_tokens == 0 and w5m + w1h > 0:
        # 只有分档数据、无总量：以分档之和为准（数据一致性修复）
        remainder = 0
        usage_cache_write_total = w5m + w1h
    else:
        usage_cache_write_total = usage.cache_write_tokens
    # 未分档的剩余缓存写按 5m 价
    w5m_effective = w5m + (remainder if usage_cache_write_total > 0 else 0)
    return CostBreakdown(
        new_input_cost=usage.new_input_tokens / m * row.input,
        cache_write_cost=w5m_effective / m * row.cache_write_5m + w1h / m * row.cache_write_1h,
        cache_read_cost=usage.cache_read_tokens / m * row.cache_read,
        output_cost=usage.output_tokens / m * row.output,
    )


def compute_cost_range(usage: UsageTiers, offpeak: PriceRow, peak: PriceRow) -> tuple[CostBreakdown, CostBreakdown]:
    """峰谷计费下的成本区间 [空闲价, 高峰价]。真实账单落在这个区间内。"""
    return compute_cost(usage, offpeak), compute_cost(usage, peak)


@dataclass
class CostResult:
    """一次用量计价的结果。"""
    model: str
    priced: bool
    tiers: list
    total_cost: float | None
    cost_if_uncached: float | None   # 反事实：同样的 token 全部按未命中价
    cache_savings: float | None      # 缓存带来的节省 = 反事实 − 实际
    currency: str | None = None
    price_row_note: str | None = None
    warnings: list = field(default_factory=list)


def cost_usage(usage: UsageTiers, model: str, table: PriceTable | None = None) -> CostResult:
    """对一份分档用量计价。无价格行的模型返回 priced=False，不编价格。"""
    table = table or DEFAULT_PRICES
    row = table.rows.get(model)
    if row is None:
        return CostResult(
            model=model, priced=False, tiers=[],
            total_cost=None, cost_if_uncached=None, cache_savings=None,
            currency=None, price_row_note=None,
            warnings=[f"模型 {model} 无价格行：未计价（CtxAutopsy 不编造价格；"
                      f"请用 load_price_table 补充该模型价格）"],
        )
    breakdown = compute_cost(usage, row)
    # 缓存写分档告警
    warnings = []
    w5m, w1h = usage.cache_write_5m_tokens, usage.cache_write_1h_tokens
    remainder = usage.cache_write_tokens - w5m - w1h
    if remainder > 0:
        warnings.append(f"缓存写未分 5m/1h 档，剩余 {remainder} tokens 按 5m 价计价")

    m = 1_000_000
    tiers = [
        TierCost("new_input", usage.new_input_tokens, breakdown.new_input_cost),
        TierCost("cache_write", usage.cache_write_tokens,
                 breakdown.cache_write_cost),
        TierCost("cache_read", usage.cache_read_tokens, breakdown.cache_read_cost),
        TierCost("output", usage.output_tokens, breakdown.output_cost),
    ]
    total = breakdown.total
    uncached = usage.input_tokens_total / m * row.input + usage.output_tokens / m * row.output
    return CostResult(
        model=model, priced=True, tiers=tiers,
        total_cost=total, cost_if_uncached=uncached,
        cache_savings=uncached - total,
        currency=row.currency, price_row_note=row.note,
        warnings=warnings,
    )


@dataclass
class ProjectCost:
    """多会话成本汇总。"""
    total_cost: float | None = None           # 单一币种时为总成本；多币种时见 totals_by_currency
    totals_by_currency: dict = field(default_factory=dict)  # 币种 → 成本合计
    by_model: dict = field(default_factory=dict)            # 模型 → 成本合计
    unpriced_models: list = field(default_factory=list)
    per_session: list = field(default_factory=list)
    self_reported_total_usd: float | None = None
    delta_vs_self_reported: float | None = None
    sessions: int = 0
    warnings: list = field(default_factory=list)


def cost_sessions(reports: list[SessionReport], table: PriceTable | None = None,
                  fx_cny_to_usd: float | None = None) -> ProjectCost:
    """对一组会话报告计价，产出项目级汇总与 harness 自报成本对比。

    - 会话内按 per_model 分模型计价（一个会话可能混用多个模型）。
    - 跨币种不假求和：totals_by_currency 分币种列出；仅当全部计价为 USD，
      或提供 fx_cny_to_usd（CNY 折 USD）时才计算 delta_vs_self_reported。
    """
    table = table or DEFAULT_PRICES
    p = ProjectCost(sessions=len(reports))
    computed_by_currency: dict = {}
    reported_total = 0.0
    has_reported = False

    for r in reports:
        if r.self_reported_cost_usd is not None:
            reported_total += r.self_reported_cost_usd
            has_reported = True
        session_tiers: dict = {}
        session_cost = 0.0
        dominant_model, dominant_tokens = None, -1
        for model, u in r.per_model.items():
            c = cost_usage(u, model, table)
            if c.priced:
                session_cost += c.total_cost or 0.0
                for t in c.tiers:
                    session_tiers[t.tier] = session_tiers.get(t.tier, 0.0) + t.cost
                p.by_model[model] = p.by_model.get(model, 0.0) + (c.total_cost or 0.0)
                if c.currency:
                    computed_by_currency[c.currency] = \
                        computed_by_currency.get(c.currency, 0.0) + (c.total_cost or 0.0)
                if u.input_tokens_total > dominant_tokens:
                    dominant_tokens = u.input_tokens_total
                    dominant_model = model
            else:
                if model not in p.unpriced_models:
                    p.unpriced_models.append(model)
        p.per_session.append({
            "session_id": r.session_id,
            "source_file": r.source_file,
            "model": dominant_model,
            "api_calls": r.api_calls,
            "cost": session_cost,
            "tiers": session_tiers,
            "self_reported_cost_usd": r.self_reported_cost_usd,
        })

    p.totals_by_currency = computed_by_currency
    if len(computed_by_currency) == 1:
        p.total_cost = next(iter(computed_by_currency.values()))
    elif computed_by_currency:
        p.total_cost = None
        p.warnings.append(
            "存在多种计价币种，total_cost 置空，请用 totals_by_currency 分币种查看")
    p.self_reported_total_usd = reported_total if has_reported else None

    # 与 harness 自报对比：USD 口径 +（可选）CNY 汇率换算
    usd_total = computed_by_currency.get("USD", 0.0)
    if fx_cny_to_usd is not None:
        usd_total += computed_by_currency.get("CNY", 0.0) / fx_cny_to_usd
    elif "CNY" in computed_by_currency or len(computed_by_currency) > 1:
        p.warnings.append(
            "计价含 CNY 且未提供 fx_cny_to_usd 汇率，无法与 harness 自报 USD 直接比较")
        usd_total = None
    if usd_total is not None and has_reported and reported_total > 0:
        p.delta_vs_self_reported = (usd_total - reported_total) / reported_total
    elif usd_total is not None:
        p.warnings.append("harness 自报成本为 0 或缺省，无法计算差异")
    return p


def costs_to_json(project: ProjectCost) -> str:
    """成本汇总序列化为 JSON。只含聚合数字与告警，不含任何消息文本。"""
    def _r6(x):
        return round(x, 6) if isinstance(x, float) else x
    return json.dumps({
        "total_cost": _r6(project.total_cost),
        "totals_by_currency": {k: _r6(v) for k, v in project.totals_by_currency.items()},
        "self_reported_total_usd": project.self_reported_total_usd,
        "delta_vs_self_reported": _r6(project.delta_vs_self_reported),
        "by_model": {k: _r6(v) for k, v in project.by_model.items()},
        "unpriced_models": project.unpriced_models,
        "sessions": project.sessions,
        "warnings": project.warnings,
        "per_session": project.per_session,
    }, ensure_ascii=False, indent=2)
