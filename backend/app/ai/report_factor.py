"""功能②：如果研报对因子/建模有参考价值，把它翻译成一个可以直接接进回测
系统的线性因子——只允许 LLM 从已知特征白名单里选、给权重，不执行 LLM 生成
的任意代码（那是策略执行沙箱要解决的问题，这套系统目前没有）。
"""
from __future__ import annotations

import json
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.ai.credentials import get_ai_credentials
from app.quant_v3.model_training import FEATURE_COLUMNS

MIN_WEIGHT, MAX_WEIGHT = -1.0, 1.0


def sanitize_feature_weights(raw: Dict[str, Any]) -> Dict[str, float]:
    """只接受白名单里的特征名、范围内的数值权重；模型编出来的字段或越界值
    直接丢弃，不报错也不悄悄放行。"""
    clean: Dict[str, float] = {}
    for name, weight in raw.items():
        if name not in FEATURE_COLUMNS:
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if not (MIN_WEIGHT <= value <= MAX_WEIGHT):
            continue
        clean[name] = value
    return clean


class ReportFactorService:
    def _rule_fallback(self, report_text: str) -> Dict[str, Any]:
        return {
            "has_reference_value": False,
            "rationale": "未接入语言模型，无法判断这份研报是否有因子建模参考价值。",
            "factor_weights": {},
            "provider": "rules",
            "model_version": "rules",
            "confidence": None,
        }

    def generate(self, report_text: str, db: Session) -> Dict[str, Any]:
        credentials = get_ai_credentials(db)
        if not credentials.api_key:
            return self._rule_fallback(report_text)
        try:
            from openai import OpenAI

            client = OpenAI(api_key=credentials.api_key, base_url=credentials.base_url)
            response = client.chat.completions.create(
                model=credentials.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是量化因子研究助手。判断这份研报的逻辑能不能对应到下面这组"
                            f"已有的量价特征上：{json.dumps(FEATURE_COLUMNS, ensure_ascii=False)}。"
                            "只输出一个 JSON 对象，字段固定为：has_reference_value（bool）、"
                            "rationale（string，中文说明理由，不承诺收益）、"
                            "factor_weights（object，键只能从上面那组特征名里选，"
                            "值是 -1 到 1 之间的权重，表示这个特征在你理解的研报逻辑里的方向和相对重要性；"
                            "如果研报和这些特征毫无关系，has_reference_value 给 false、factor_weights 给空对象）。"
                        ),
                    },
                    {"role": "user", "content": report_text},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json(content)
            raw_weights = parsed.get("factor_weights", {})
            return {
                "has_reference_value": bool(parsed.get("has_reference_value", False)),
                "rationale": parsed.get("rationale", content),
                "factor_weights": sanitize_feature_weights(raw_weights if isinstance(raw_weights, dict) else {}),
                "provider": "openai-compatible",
                "model_version": credentials.model,
                "confidence": None,
            }
        except Exception:
            return self._rule_fallback(report_text)

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return {}
