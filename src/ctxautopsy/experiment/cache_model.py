"""四档用量模拟与缓存建模（模块 E 核心）。

逐轮模拟"每轮请求的 prompt 有多少命中缓存读、多少重写缓存、多少是新输入"，
再用模块 B 的价格表计价。这是「省 token ≠ 省钱」的结构复现装置：
压缩改写历史 → 缓存前缀失效 → 被迫按 cache_write 价（≈新 token 价）重写，
而省下的只是 cache_read 价（≈1/10）的 token。

诚实建模：CacheAssumptions 的每一条假设都会原样写进报告（A1-A7），
本模型只保证变化方向与量级"结构对"，不声称复刻真实缓存；
所有 token 均为 chars/4 估算口径，所有美元数字是模型估算值。
"""
from dataclasses import dataclass, field

from ..schema import Msg, estimate_tokens


@dataclass
class CacheAssumptions:
    """缓存建模全部显式假设（报告第 3 节原文输出）。"""
    chars_per_token: float = 4.0            # A1 token 口径：字符数/4（估算，非真实 tokenizer）
    breakpoint_tokens: int = 1024           # A2 缓存断点粒度
    min_cache_tokens: int = 1024            # A2 写入缓存的最小整段长度
    resident_cached_after_turn: int = 1     # A2 常驻前缀从第几轮起可命中缓存
    compression_breaks_prefix: bool = True  # A3 压缩改写历史 → 从首个差异点起缓存失效
    re_cache_after_compress: bool = True    # A3 压缩后的历史下一轮重新写缓存
    multi_breakpoint: bool = False          # A3 False=单断点严格前缀（悲观下界）；True=多断点（乐观上界）
    compress_resident: bool = False         # A4 默认不动常驻前缀（防止拿 system prompt 造假结论）

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("chars_per_token", "breakpoint_tokens", "min_cache_tokens",
                 "resident_cached_after_turn", "compression_breaks_prefix",
                 "re_cache_after_compress", "multi_breakpoint", "compress_resident")}


@dataclass
class TierUsage:
    new_input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

    def prompt_tokens(self) -> int:
        return self.new_input + self.cache_write + self.cache_read

    def to_dict(self) -> dict:
        return {"new_input": self.new_input, "cache_write": self.cache_write,
                "cache_read": self.cache_read, "output": self.output}

    def __add__(self, other: "TierUsage") -> "TierUsage":
        return TierUsage(self.new_input + other.new_input,
                         self.cache_write + other.cache_write,
                         self.cache_read + other.cache_read,
                         self.output + other.output)


@dataclass
class TurnUsage:
    turn: int
    prompt_tokens: int
    tiers: TierUsage
    cost_usd: float

    def to_dict(self) -> dict:
        return {"turn": self.turn, "prompt_tokens": self.prompt_tokens,
                "tiers": self.tiers.to_dict(), "cost_usd": self.cost_usd}


@dataclass
class CompressPlan:
    compress_from_turn: int = 3             # 从该轮起压缩历史（1 = 首轮，历史为空）
    every_turn: bool = False                # True = 每轮重压历史（前缀永不稳定）
    strategy: str = "baseline"
    params: dict = field(default_factory=dict)

    @staticmethod
    def no_compression() -> "CompressPlan":
        return CompressPlan(strategy="baseline")


def _shared_block_tokens(prev_blocks: list[Msg], cur_blocks: list[Msg],
                         chars_per_token: float) -> int:
    """与上一轮请求的最长公共前缀 token 数（块级逐块比较，
    首个差异块内按 floor(公共字符/chars_per_token) 折算）。"""
    shared = 0
    for a, b in zip(prev_blocks, cur_blocks):
        if a.uid == b.uid and a.text == b.text:
            shared += estimate_tokens(a.text)
        else:
            # 公共前缀字符数
            n = 0
            for ca, cb in zip(a.text, b.text):
                if ca != cb:
                    break
                n += 1
            shared += int(n / chars_per_token)
            break
    return shared


def simulate_turns(task, plan: CompressPlan, cache: CacheAssumptions,
                   price_row, compress_fn=None) -> list[TurnUsage]:
    """逐轮模拟四档用量并按价格行计价。

    price_row：模块 B 的 PriceRow（用 input / cache_write_5m / cache_read / output）。
    compress_fn：(历史块列表, 参数字典) -> 压缩后块列表；None 或 baseline 时恒等。
    """
    b = cache.breakpoint_tokens
    turns: list[TurnUsage] = []
    prev_request: list[Msg] = []
    cached = 0  # 上一轮结束后缓存内的 token 数（简化：单断点连续前缀）

    for t in range(1, task.cfg.n_turns + 1):
        # 历史 = 前 t-1 轮的块展开（每轮请求只带历史 + 本轮 user 块）
        history: list[Msg] = []
        for turn_blocks in task.turns[:t - 1]:
            history.extend(turn_blocks)
        # 压缩历史（按 plan）
        if plan.strategy != "baseline" and t >= plan.compress_from_turn and compress_fn is not None:
            history = list(compress_fn(history, plan.params))
        user_block = task.turns[t - 1][0]
        request = list(task.resident) + history + [user_block]

        P = sum(estimate_tokens(m.text) for m in request)
        shared = _shared_block_tokens(prev_request, request, cache.chars_per_token)

        def align_down(x: int) -> int:
            return (x // b) * b

        cr = align_down(min(cached, shared))
        cw = align_down(P - cr)
        if cw < cache.min_cache_tokens:
            cw = 0
        ni = P - cr - cw

        # 多断点乐观上界：剩余部分按断点切块，与上一轮请求同位块相同 → 计入 cr
        if cache.multi_breakpoint and cw > 0:
            prev_txt = "".join(m.text for m in prev_request)
            cur_txt = "".join(m.text for m in request)
            cpt = cache.chars_per_token
            extra_cr = 0
            pos = cr
            while pos + b <= P:
                start_ch = int(pos * cpt)
                chunk = cur_txt[start_ch:int((pos + b) * cpt)]
                if chunk and chunk in prev_txt:
                    extra_cr += b
                pos += b
            cr += extra_cr
            cw = align_down(P - cr)
            if cw < cache.min_cache_tokens:
                cw = 0
            ni = P - cr - cw

        # 本轮 output = thinking + tool_call + reply 块（不含 user 与 tool_result）
        out_blocks = [m for m in task.turns[t - 1][1:]
                      if m.kind == "tool_call"
                      or (m.kind == "text" and m.role == "assistant")]
        out_tokens = sum(estimate_tokens(m.text) for m in out_blocks)

        tiers = TierUsage(new_input=ni, cache_write=cw, cache_read=cr, output=out_tokens)
        cost = (ni * price_row.input + cw * price_row.cache_write_5m
                + cr * price_row.cache_read + out_tokens * price_row.output) / 1e6
        # 不在此处舍入：单轮成本在微美元量级，round(x, 6) 会吃掉有效数字；
        # 舍入统一在 metrics 层对合计值做
        turns.append(TurnUsage(turn=t, prompt_tokens=P, tiers=tiers, cost_usd=cost))

        cached = cr + cw
        prev_request = request

    return turns
