"""策略助手：自然语言 → 策略参数建议（不是可执行代码，也不自动创建/覆盖
任何策略）。PRD 9.1 节："生成代码默认不可直接实盘"——这里更保守，连代码都
不生成，只产出一份参数建议，由人工在策略中心确认后手动创建/修改策略。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.ai.credentials import get_ai_credentials

# 只允许这些字段进入建议结果——把 LLM 的自由文本收窄成结构化、可校验的
# 参数子集，避免模型编出策略配置里不存在的字段，或者返回可执行代码。
ALLOWED_FIELDS: Dict[str, tuple] = {
    "holdings_count": (int, 5, 30),
    "max_weight": (float, 0.05, 0.5),
    "rebalance_frequency": (str, None, None),
    "cash_buffer": (float, 0.0, 0.3),
    "trend_filter": (bool, None, None),
    "stop_loss": (float, 0.05, 0.4),
    "target_volatility": (float, 0.05, 0.5),
    "max_drawdown_budget": (float, 0.05, 0.5),
    "universe": (str, None, None),
}
ALLOWED_FREQUENCIES = {"weekly", "monthly", "quarterly"}
ALLOWED_UNIVERSES = {"a_share", "hs300", "csi_a500", "large_cap", "custom"}


def _clamp(field: str, value: Any) -> Optional[Any]:
    kind, lo, hi = ALLOWED_FIELDS[field]
    try:
        if kind is bool:
            return bool(value)
        if kind is int:
            value = int(value)
        elif kind is float:
            value = float(value)
        elif kind is str:
            value = str(value)
    except (TypeError, ValueError):
        return None
    if field == "rebalance_frequency" and value not in ALLOWED_FREQUENCIES:
        return None
    if field == "universe" and value not in ALLOWED_UNIVERSES:
        return None
    if lo is not None and value < lo:
        return None
    if hi is not None and value > hi:
        return None
    return value


def _sanitize(raw: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for field in ALLOWED_FIELDS:
        if field in raw:
            value = _clamp(field, raw[field])
            if value is not None:
                clean[field] = value
    return clean


class StrategyAssistantService:
    def _rule_fallback(self, description: str) -> Dict[str, Any]:
        """没有可用的 LLM 时，给一个保守的规则回退——不猜测用户意图的细节，
        只给一组通用、低风险的默认参数，明确标注这是模板而非针对性建议。
        """
        return {
            "content": "未接入语言模型，返回通用保守参数模板，请结合自己的判断调整后再在策略中心创建策略。",
            "provider": "rules",
            "model_version": "rules",
            "confidence": None,
            "suggested_params": {
                "holdings_count": 10, "max_weight": 0.15, "rebalance_frequency": "monthly",
                "cash_buffer": 0.05, "trend_filter": True, "stop_loss": 0.18,
            },
        }

    def suggest(self, description: str, db: Session) -> Dict[str, Any]:
        credentials = get_ai_credentials(db)
        if not credentials.api_key:
            return self._rule_fallback(description)
        try:
            from openai import OpenAI

            client = OpenAI(api_key=credentials.api_key, base_url=credentials.base_url)
            schema_hint = {field: kind.__name__ for field, (kind, _, _) in ALLOWED_FIELDS.items()}
            response = client.chat.completions.create(
                model=credentials.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是量化策略参数助手。只输出策略参数建议，不输出任何可执行代码，"
                            "不承诺收益。先用一段中文说明理由，再另起一行输出一个 JSON 对象，"
                            f"字段只能从这些里选（不要编造其它字段）：{json.dumps(schema_hint, ensure_ascii=False)}。"
                            "rebalance_frequency 只能是 weekly/monthly/quarterly，"
                            "universe 只能是 a_share/hs300/csi_a500/large_cap/custom。"
                        ),
                    },
                    {"role": "user", "content": description},
                ],
            )
            content = response.choices[0].message.content or ""
            raw_params: Dict[str, Any] = {}
            # 找输出里最后一个花括号 JSON 块；找不到就不给结构化参数，只给文字说明。
            start = content.rfind("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    raw_params = json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    raw_params = {}
            return {
                "content": content,
                "provider": "openai-compatible",
                "model_version": credentials.model,
                "confidence": None,
                "suggested_params": _sanitize(raw_params),
            }
        except Exception:
            return self._rule_fallback(description)
