# 上下文压缩对照实验报告

> 所有 token 数为 chars/4 估算口径，所有美元数为缓存模型估算值（假设清单见第 3 节），只用于策略间相对对比，不等于真实账单。

## 1. 概览（每格均值）

| 任务 | 策略 | 压缩率 | 约束损坏 | 否定词损坏 | 工具链损坏 | 数值损坏 | 账单差(USD) | 账单差% |
|---|---|---|---|---|---|---|---|---|
| t_short | baseline | 0.0% | 0.0 | 0.0 | 0.0 | 0.0 | +0.000000 | +0.0% |
| t_short | truncate_head | 50.1% | 1.0 | 0.0 | 0.0 | 27.3 | -0.000100 | -0.6% |
| t_short | sliding_window | 83.6% | 2.0 | 0.0 | 1.0 | 46.0 | -0.000200 | -2.2% |
| t_short | freq_drop | 27.8% | 0.7 | 46.3 | 0.0 | 0.0 | +0.000000 | +0.2% |
| t_short | rule_summary | 28.9% | 0.0 | 0.0 | 0.0 | 45.3 | +0.000000 | +0.2% |
| t_tool_heavy | baseline | 0.0% | 0.0 | 0.0 | 0.0 | 0.0 | +0.000000 | +0.0% |
| t_tool_heavy | truncate_head | 50.0% | 1.0 | 0.0 | 0.0 | 27.0 | -0.000100 | -0.6% |
| t_tool_heavy | sliding_window | 83.6% | 2.0 | 0.0 | 1.0 | 45.3 | -0.000200 | -2.2% |
| t_tool_heavy | freq_drop | 26.6% | 0.0 | 15.7 | 0.0 | 0.0 | +0.000000 | +0.2% |
| t_tool_heavy | rule_summary | 28.9% | 0.0 | 0.0 | 0.0 | 44.3 | +0.000000 | +0.2% |
| t_numeric | baseline | 0.0% | 0.0 | 0.0 | 0.0 | 0.0 | +0.000000 | +0.0% |
| t_numeric | truncate_head | 50.1% | 1.0 | 0.0 | 0.0 | 27.0 | -0.000100 | -0.6% |
| t_numeric | sliding_window | 83.6% | 2.0 | 0.0 | 1.0 | 42.3 | -0.000200 | -2.2% |
| t_numeric | freq_drop | 26.6% | 0.0 | 16.0 | 0.0 | 0.0 | +0.000000 | +0.2% |
| t_numeric | rule_summary | 28.8% | 0.0 | 0.0 | 0.0 | 44.3 | +0.000000 | +0.2% |

## 2. 关键发现

**省 token 反涨价**：6 个任务×策略组合压缩率 > 0 且账单反而上涨：
- t_short × freq_drop
- t_short × rule_summary
- t_tool_heavy × freq_drop
- t_tool_heavy × rule_summary
- t_numeric × freq_drop
- t_numeric × rule_summary

**各策略主导损坏类型**（该策略在全部任务上的平均损坏计数最大值）：
- truncate_head: numeric
- sliding_window: numeric
- freq_drop: negation
- rule_summary: numeric

## 3. 缓存建模假设清单（诚实建模，逐条原文）

| 编号 | 假设 |
|---|---|
| A1 | 口径假设：全项目 token 用 chars/4 估算，非真实 tokenizer；中文内容建议 chars_per_token 调至 1.5~2 重跑做敏感性对比。 |
| A2 | 缓存对象假设：每轮请求整条 prompt 打缓存标记；只有完整断点段可写入/命中，低于 min_cache 的片段按新输入计。 |
| A3 | 前缀语义假设：默认单断点严格前缀（悲观下界）——历史任何改写使从首个差异点起全部内容失效、以 cache_write 价（≈新 token 价）重写；多断点（乐观上界）——内容未变的断点段即使位置移动也可命中。模型不声称复刻真实缓存，只保证变化方向与量级对比'结构对'。 |
| A4 | 压缩时点与节奏：压缩只作用于历史轮（默认不动常驻前缀与当前轮）；one-shot 与 every_turn 两种节奏分别可跑。 |
| A5 | 重放成本：工具链损坏后 agent 是否真的重放无法离线确知，默认不计入账单；include_replay_cost=True 时输出推测上界并标注'非实测'。 |
| A6 | 输出口径：压缩不改变本轮 output（压缩对象是历史）。 |
| A7 | 模型不含真实计费细节（最低消费、长上下文附加费等），所有美元数字是'模型估算值'，只用于策略间相对对比，绝对数字不等于真实账单。 |

本次运行的参数：chars_per_token=4.0、breakpoint_tokens=1024、min_cache_tokens=1024、multi_breakpoint=False、every_turn=False。

## 4. 价格表与来源

- 计价模型：deepseek-v4-flash
- 价格行（USD/百万 token）：输入 0.15 / 缓存写5m 0.15 / 缓存写1h 0.15 / 缓存读 0.003 / 输出 0.6
- 来源：DeepSeek 官方定价页（空闲时段，2026-09-10 起峰谷分时）https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
- 备注：高峰时段×2（北京时间工作日 9-12、14-18）；无 5m/1h 缓存写分层，两档同价

## 5. 复现方法

```
python -m ctxautopsy.experiment            # 默认演示矩阵（3 任务 × 5 策略 × 3 重复）
python -m ctxautopsy.experiment --config config.json --out-dir results
```
- 矩阵：3 任务 × 5 策略 × 3 重复 = 45 格
- seed 派生：crc32("task|strategy|repeat") % 2**31，全链路确定性，任何机器可复现
- 压缩时点：第 3 轮起，every_turn=False
