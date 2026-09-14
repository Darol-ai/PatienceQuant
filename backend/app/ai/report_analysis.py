"""功能①：研报总结 + 策略参考。只做解读，不生成可执行代码、不自动创建
策略——比"因子生成"（report_factor.py）更保守的一档能力。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.ai.credentials import get_ai_credentials


class ReportAnalysisService:
    def _rule_fallback(self, report_text: str) -> Dict[str, Any]:
        preview = report_text[:120].replace("\n", " ")
        return {
            "summary": f"未接入语言模型，无法生成真正的摘要。研报开头预览：{preview}…",
            "key_points": [],
            "strategy_reference": "配置语言模型后可获得针对性的策略参考，目前仅做原文预览。",
            "provider": "rules",
            "model_version": "rules",
            "confidence": None,
        }

    def analyze(self, report_text: str, db: Session) -> Dict[str, Any]:
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
                            "你是量化投研助手，负责解读用户上传的研报。只做总结和策略参考，"
                            "不承诺收益，不建议直接下单。只输出一个 JSON 对象，字段固定为："
                            "summary（string，300字以内中文摘要）、"
                            "key_points（string 数组，研报里和量化因子/选股逻辑相关的关键点）、"
                            "strategy_reference（string，基于研报内容给出的策略思路参考，"
                            "要说明这只是参考、不是可直接执行的策略）。"
                        ),
                    },
                    {"role": "user", "content": report_text},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json(content)
            return {
                "summary": parsed.get("summary", content),
                "key_points": parsed.get("key_points", []) if isinstance(parsed.get("key_points"), list) else [],
                "strategy_reference": parsed.get("strategy_reference", ""),
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
