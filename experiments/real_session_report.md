# 真实会话解剖报告（本机 ~/.claude/projects 实测）

> 数据：本机 Claude Code 会话日志（jsonl），全部数字为实测聚合，不包含任何消息文本。token 为 API usage 里的权威数字；「构成桶」一栏为启发式估算（CJK≈1 token/字，已在表内标注）。

## 1. 全量概览

- 会话文件：321 个，API 调用：16,667 次
- 新输入 49.17M / 缓存写 0 （本机网关不报告缓存写）/ 缓存读 1055.4M / 输出 16.94M
- **缓存读占输入总量 95.5%**——缓存命中/未命中价差 30-50 倍，这个比例决定了成本结构

## 2. 成本：官方价推算 vs harness 自报

- 按 DeepSeek 官方价推算：**$33.91**（空闲时段价）
- harness（Claude Code）自报累计：**$478.92**（其自身标记 hasUnknownModelCost=True，口径不明）
- 差距：推算值比自报低 **92.9%**——即通过第三方网关实际付费约为官方价的 **14 倍**。这是实测差距，只报告不做归因。
- 未定价模型（跳过）：['<synthetic>']

## 3. 最大的 10 个会话

| 会话 | 调用 | 新输入 | 缓存读 | 缓存读占比 | 常驻开销(估算) |
|---|---|---|---|---|---|
| 066ce6ce-db2f-4704-812a-603d732c2291.jso | 3,112 | 13.60M | 356.7M | 96.3% | 26,610 |
| agent-a15c4f1015a1e53a6.jsonl | 355 | 0.28M | 32.0M | 99.1% | 8,857 |
| agent-afbf72f5cfd12b48c.jsonl | 246 | 0.47M | 19.6M | 97.6% | 2,208 |
| agent-ab75d588f7e97c307.jsonl | 202 | 0.18M | 13.2M | 98.7% | 2,295 |
| agent-af6313ac016eff8ad.jsonl | 193 | 0.10M | 11.4M | 99.1% | 10,950 |
| agent-ab75368e1107ce941.jsonl | 166 | 0.23M | 14.4M | 98.4% | 4,527 |
| agent-a68aabad9ebcd028c.jsonl | 163 | 0.32M | 9.2M | 96.6% | 16,639 |
| agent-a04fb42a6f39739cd.jsonl | 157 | 0.20M | 10.8M | 98.2% | 5,415 |
| agent-a46c0af2dc370a1ab.jsonl | 150 | 0.21M | 7.8M | 97.4% | 3,028 |
| agent-a1ebb57e1a0f5a8a9.jsonl | 143 | 0.60M | 13.9M | 95.9% | 14,989 |

## 4. 最大会话的上下文构成（估算桶）

- 会话：066ce6ce-db2f-4704-812a-603d732c2291.jsonl（3,112 次调用，语义：new_only）
- 常驻开销：**26,610 token**（方法：first_call_minus_user_text）——会话一启动、每次调用都重发的部分
- 用户文本 11,309 / 工具输出 669,784 / 思考 1,113,179 / 压缩摘要 0
- 模型分布：deepseek-v4-pro 1,067 次、deepseek-v4-flash 2,039 次、<synthetic> 6 次

## 5. 价格表（可溯源，可用 JSON 覆盖）

- claude-sonnet-5-5：输入 2.0 / 缓存写5m 2.5 / 缓存读 0.2 / 输出 10.0（USD/百万 token）——Anthropic 官方定价（claude-api 技能内置价格快照 2026-10-06）；缓存写 5m=1.25×、1h=2×输入价
- claude-sonnet-5：输入 2.0 / 缓存写5m 2.5 / 缓存读 0.2 / 输出 10.0（USD/百万 token）——Anthropic 官方定价（claude-api 技能内置价格快照 2026-10-06）；缓存写 5m=1.25×、1h=2×输入价
- claude-opus-5-5：输入 4.0 / 缓存写5m 5.0 / 缓存读 0.2 / 输出 20.0（USD/百万 token）——Anthropic 官方定价（claude-api 技能内置价格快照 2026-10-06）；缓存写 5m=1.25×、1h=2×输入价
- claude-fable-5-1：输入 10.0 / 缓存写5m 12.5 / 缓存读 0.25 / 输出 50.0（USD/百万 token）——Anthropic 官方定价（claude-api 技能内置价格快照 2026-10-06）；缓存写 5m=1.25×、1h=2×输入价
- deepseek-v4-pro：输入 0.66 / 缓存写5m 0.66 / 缓存读 0.022 / 输出 1.98（USD/百万 token）——DeepSeek 官方定价页（空闲时段，2026-09-10 起峰谷分时）https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
- deepseek-v4-pro[1m]：输入 0.66 / 缓存写5m 0.66 / 缓存读 0.022 / 输出 1.98（USD/百万 token）——DeepSeek 官方定价页（空闲时段，2026-09-10 起峰谷分时）https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
- deepseek-v4-flash：输入 0.15 / 缓存写5m 0.15 / 缓存读 0.003 / 输出 0.6（USD/百万 token）——DeepSeek 官方定价页（空闲时段，2026-09-10 起峰谷分时）https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
- deepseek-v4.1-flash：输入 0.15 / 缓存写5m 0.15 / 缓存读 0.003 / 输出 0.6（USD/百万 token）——DeepSeek 官方定价页（空闲时段，2026-09-10 起峰谷分时）https://api-docs.deepseek.com/zh-cn/quick_start/pricing/

## 复现

```
python -m ctxautopsy analyze ~/.claude/projects --out real_report.md
```