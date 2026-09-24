# AI 助手接 DeepSeek，加审计留痕闭环

## 决定

`PatienceQuant` 已有的 `/api/ai/explain/trade`（规则解释器 + 可选 OpenAI 兼容
LLM）直接复用——DeepSeek 网关本来就是 OpenAI 兼容接口，只需把 `.env` 的
`OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL` 指向 DeepSeek 网关即可，不用
新写调用代码。真机验证过：`provider` 从 `rules` 变成 `openai-compatible`，
生成的中文研究摘要是真实模型输出，不是模板。

新增：
- **策略助手**（`app/ai/strategy_assistant.py` + `POST /api/ai/strategy-assistant`）：
  自然语言 → 策略参数建议。比 PRD 9.1 节"生成代码默认不可直接实盘"更保守——
  连代码都不生成，只产出一份参数 JSON，且严格收敛到白名单字段
  （`ALLOWED_FIELDS`）、越界值直接丢弃，不自动创建或修改任何策略，需要人工
  在策略中心确认后手动创建。真机验证过用真实自然语言描述换来了合理的参数组合。
- **审计留痕**：`AIExplanation` 表加了 `model_version`/`confidence`/`adopted`/
  `rolled_back`/`decided_at` 字段（SQLite 用项目已有的 lightweight-migration
  模式加列，不引入 Alembic）；`explain_trade`/`strategy_assistant` 两个接口
  都会落一条审计记录并把 `audit_id` 回传给前端；新增 `POST /api/ai/audit/{id}/decision`
  给人工标记"采纳"或"回滚"，`GET /api/ai/audit` 查审计列表。`confidence`
  允许为空——模型没有真实可用的数值置信度（没有拿到 logprobs）时宁可留空，
  不伪造一个看起来精确的数字。

前端 `AIResearch.tsx` 加了：LLM 开关、审计信息展示（model_version/confidence/
audit_id）、采纳/回滚按钮、策略助手的输入框和结果展示。

## 理由

DeepSeek/Qwen 是 OpenAI 兼容端点，复用现成的 `openai` SDK 调用路径比新写一套
专用客户端更省事，也符合"少造轮子"的一贯原则。策略助手刻意比 PRD 要求更保守
（不生成代码、不自动应用），因为这套系统目前没有对生成代码做沙箱/静态扫描的
能力，把风险控制在"只给参数建议"这一层最安全。
