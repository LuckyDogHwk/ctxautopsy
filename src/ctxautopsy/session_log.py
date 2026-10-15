"""解析 Claude Code 会话 jsonl，产出分档用量与会话构成统计（模块 A）。

真实日志结构（2026-10-15 本机实测确认，详见 findings.md）：
- assistant 条目携带 message.usage；缓存写分 5m/1h 两档时在
  usage.cache_creation.ephemeral_5m_input_tokens / ephemeral_1h_input_tokens，
  旧格式只有 usage.cache_creation_input_tokens（无分档）
- usage 语义分两种（按总量自动探测）：
  * "total"（Anthropic 标准）：input_tokens 是总输入，含缓存读与缓存写
  * "new_only"（本机网关实测）：input_tokens 不含缓存读
    （95% 的调用 input < cache_read，如 input=1484、cache_read=135936），
    此时 input 全部视为新输入
- cost-state 条目携带 harness 自报累计成本（口径不明，只作交叉参考）
- 大量噪音条目（queue-operation / summary / progress / attachment / cost-state 等）
  与坏行，跳过并计数

隐私约定：只落聚合统计，不落任何消息文本（tests 里有哨兵文本保障）。
"""
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class UsageTiers:
    """一次（或累计）API 调用的分档用量。input_tokens_total = 新输入+缓存写+缓存读。"""
    new_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0

    @property
    def input_tokens_total(self) -> int:
        return self.new_input_tokens + self.cache_write_tokens + self.cache_read_tokens

    def __add__(self, other: "UsageTiers") -> "UsageTiers":
        return UsageTiers(
            new_input_tokens=self.new_input_tokens + other.new_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_write_5m_tokens=self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            cache_write_1h_tokens=self.cache_write_1h_tokens + other.cache_write_1h_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            api_calls=self.api_calls + other.api_calls,
        )


@dataclass
class Composition:
    """会话构成桶：上下文里的 token 都花在哪了。"""
    resident_overhead_tokens: int = 0      # 常驻开销估计（首轮输入 − 首轮前的用户文本）
    overhead_method: str = ""
    user_text_tokens: int = 0              # 用户真实文本
    tool_output_tokens: int = 0            # 工具结果
    thinking_tokens: int = 0               # 思考块
    compact_summary_tokens: int = 0        # 压缩摘要（isCompactSummary 的用户条目）
    tool_result_blocks: int = 0
    thinking_blocks: int = 0


@dataclass
class SessionReport:
    """一个会话文件的解剖结果。"""
    source_file: str = ""
    session_id: str = ""
    usage_semantics: str = "none"          # new_only | total | none
    usage: UsageTiers = field(default_factory=UsageTiers)
    composition: Composition = field(default_factory=Composition)
    per_model: dict = field(default_factory=dict)   # 模型名 → UsageTiers
    total_lines: int = 0
    parsed_lines: int = 0
    parse_errors: int = 0
    skipped_by_type: dict = field(default_factory=dict)
    self_reported_cost_usd: float | None = None
    warnings: list = field(default_factory=list)

    @property
    def api_calls(self) -> int:
        return self.usage.api_calls


@dataclass
class Aggregate:
    """多会话汇总。"""
    totals: UsageTiers = field(default_factory=UsageTiers)
    sessions: int = 0
    total_self_reported_cost_usd: float | None = None
    warnings: list = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的估算：CJK 按 1 token/字，其余按 4 字符/token（四舍五入）。

    校准于真实日志与测试 fixture；仅用于相对比较，绝对值不承诺准确（报告会写明）。
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    non_cjk = len(text) - cjk
    return cjk + int(non_cjk / 4 + 0.5)


def _user_entry_text_tokens(entry: dict) -> int:
    """一条 user 条目的 text 块 token 数。isMeta 行不计。"""
    if entry.get("isMeta"):
        return 0
    msg = entry.get("message") or {}
    total = 0
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            total += estimate_tokens(block.get("text") or "")
    return total


def parse_session(path: str | Path) -> SessionReport:
    """解析单个 jsonl 会话文件。坏行与噪音条目跳过并计数。"""
    r = SessionReport(source_file=str(Path(path).name))
    # 原始行：(input, creation_total, eph_5m, eph_1h, read, output, model)
    rows: list[tuple[int, int, int, int, int, int, str]] = []
    user_text_before_first_call = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r.total_lines += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                r.parse_errors += 1
                continue
            r.parsed_lines += 1
            if not r.session_id:
                r.session_id = entry.get("sessionId") or ""

            etype = entry.get("type")
            if etype == "assistant":
                msg = entry.get("message") or {}
                content = msg.get("content") or []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        r.composition.thinking_blocks += 1
                        r.composition.thinking_tokens += estimate_tokens(block.get("thinking") or "")
                usage = msg.get("usage") or {}
                if "input_tokens" not in usage and "output_tokens" not in usage:
                    continue  # 无用量信息的条目不算 API 调用
                cc = usage.get("cache_creation") or {}
                rows.append((
                    usage.get("input_tokens", 0),
                    usage.get("cache_creation_input_tokens", 0),
                    cc.get("ephemeral_5m_input_tokens", 0),
                    cc.get("ephemeral_1h_input_tokens", 0),
                    usage.get("cache_read_input_tokens", 0),
                    usage.get("output_tokens", 0),
                    msg.get("model") or entry.get("requestedModel") or "unknown",
                ))
            elif etype == "user":
                if entry.get("isCompactSummary"):
                    r.composition.compact_summary_tokens += _user_entry_text_tokens(entry)
                else:
                    text_tokens = _user_entry_text_tokens(entry)
                    r.composition.user_text_tokens += text_tokens
                    if not rows:  # 首轮 API 调用之前的用户文本（常驻开销估计用）
                        user_text_before_first_call += text_tokens
                # 工具结果块（无论 isCompactSummary 与否都计入）
                msg = entry.get("message") or {}
                for block in msg.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        r.composition.tool_result_blocks += 1
                        result = block.get("content")
                        if isinstance(result, str):
                            r.composition.tool_output_tokens += estimate_tokens(result)
                        elif isinstance(result, list):
                            for b in result:
                                if isinstance(b, dict) and b.get("type") == "text":
                                    r.composition.tool_output_tokens += estimate_tokens(b.get("text") or "")
            else:
                r.skipped_by_type[etype] = r.skipped_by_type.get(etype, 0) + 1
                if etype == "cost-state":
                    r.self_reported_cost_usd = entry.get("totalCostUSD")

    # ---- 语义探测 ----
    if not rows:
        r.warnings.append("该会话没有可用的 usage 数据（0 次 API 调用）")
        return r
    total_input = sum(x[0] for x in rows)
    total_read = sum(x[4] for x in rows)
    r.usage_semantics = "new_only" if total_read > total_input else "total"

    # ---- 归一化聚合（统一为：input 总量 = 新输入 + 缓存写 + 缓存读） ----
    for input_tokens, creation_total, eph_5m, eph_1h, read, output, model in rows:
        split = eph_5m + eph_1h
        write_total = split if split > 0 else creation_total
        if r.usage_semantics == "new_only":
            new_input = input_tokens  # 网关口径：input 不含缓存读，全部视为新输入
        else:
            new_input = input_tokens - read - write_total
        u = UsageTiers(
            new_input_tokens=new_input,
            cache_write_tokens=write_total,
            cache_write_5m_tokens=eph_5m,
            cache_write_1h_tokens=eph_1h,
            cache_read_tokens=read,
            output_tokens=output,
            api_calls=1,
        )
        r.usage += u
        r.per_model.setdefault(model, UsageTiers())
        r.per_model[model] += u

    # ---- 常驻开销估计：首轮输入 − 首轮之前的用户文本（语义感知） ----
    first = rows[0]
    first_write = first[2] + first[3] if first[2] + first[3] > 0 else first[1]
    if r.usage_semantics == "new_only":
        base = first[0]
        r.composition.overhead_method = "first_call_minus_user_text"
    else:
        base = first[0] - first[4] - first_write
        r.composition.overhead_method = "first_call_minus_cache_minus_user_text"
    r.composition.resident_overhead_tokens = max(0, base - user_text_before_first_call)

    return r


def parse_sessions_dir(directory: str | Path, *, limit: int | None = None) -> list[SessionReport]:
    """递归扫描目录下全部会话 jsonl，按文件名排序返回。读不了的文件跳过。

    limit 非空时只解析前 limit 个文件（按文件名序，适合超大目录抽样）。
    """
    files = sorted(Path(directory).rglob("*.jsonl"))
    if limit is not None:
        files = files[:limit]
    reports = []
    for p in files:
        try:
            reports.append(parse_session(p))
        except OSError:
            continue
    reports.sort(key=lambda r: r.source_file)
    return reports


def sessions_to_json(reports: list[SessionReport]) -> str:
    """序列化为 JSON 字符串。只含聚合数字，不含任何消息文本（隐私硬约束）。"""
    out = []
    for r in reports:
        out.append({
            "session_id": r.session_id,
            "source_file": r.source_file,
            "usage_semantics": r.usage_semantics,
            "api_calls": r.api_calls,
            "usage": {
                "new_input_tokens": r.usage.new_input_tokens,
                "cache_write_tokens": r.usage.cache_write_tokens,
                "cache_write_5m_tokens": r.usage.cache_write_5m_tokens,
                "cache_write_1h_tokens": r.usage.cache_write_1h_tokens,
                "cache_read_tokens": r.usage.cache_read_tokens,
                "output_tokens": r.usage.output_tokens,
                "api_calls": r.usage.api_calls,
                "input_tokens_total": r.usage.input_tokens_total,
            },
            "per_model": {m: {
                "new_input_tokens": u.new_input_tokens,
                "cache_write_tokens": u.cache_write_tokens,
                "cache_read_tokens": u.cache_read_tokens,
                "output_tokens": u.output_tokens,
                "api_calls": u.api_calls,
            } for m, u in r.per_model.items()},
            "composition": {
                "resident_overhead_tokens": r.composition.resident_overhead_tokens,
                "overhead_method": r.composition.overhead_method,
                "user_text_tokens": r.composition.user_text_tokens,
                "tool_output_tokens": r.composition.tool_output_tokens,
                "thinking_tokens": r.composition.thinking_tokens,
                "compact_summary_tokens": r.composition.compact_summary_tokens,
                "tool_result_blocks": r.composition.tool_result_blocks,
                "thinking_blocks": r.composition.thinking_blocks,
            },
            "self_reported_cost_usd": r.self_reported_cost_usd,
            "warnings": r.warnings,
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


def aggregate_sessions(reports: list[SessionReport]) -> Aggregate:
    """多会话汇总：总用量 + 自报成本合计。缺自报成本的会话计入 warnings。"""
    a = Aggregate(sessions=len(reports))
    missing = 0
    for r in reports:
        a.totals += r.usage
        if r.self_reported_cost_usd is not None:
            a.total_self_reported_cost_usd = (a.total_self_reported_cost_usd or 0.0) \
                + r.self_reported_cost_usd
        else:
            missing += 1
    if missing:
        a.warnings.append(f"{missing} 个会话无 harness 自报成本，total_self_reported_cost_usd 为部分合计")
    return a
