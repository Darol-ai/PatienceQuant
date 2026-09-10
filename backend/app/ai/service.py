from __future__ import annotations

import json
from typing import Any, Dict

from app.ai.explainer import RuleExplainer
from app.config import get_settings


class AIResearchService:
    def __init__(self):
        self.rules = RuleExplainer()

    def explain(self, context: Dict[str, Any], use_llm: bool = False) -> Dict[str, Any]:
        baseline = self.rules.explain(context)
        settings = get_settings()
        if not use_llm or not settings.openai_api_key:
            return baseline
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
            response = client.chat.completions.create(
                model=settings.openai_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "你是量化投研解释助手，只解释已有量化信号，不得改变交易决定。输出简洁中文。"},
                    {"role": "user", "content": "请结合规则解释和量化上下文，生成研究摘要、交易原因和风险提示。上下文：%s；规则解释：%s" % (json.dumps(context, ensure_ascii=False), json.dumps(baseline, ensure_ascii=False))},
                ],
            )
            content = response.choices[0].message.content or baseline["content"]
            return {**baseline, "content": content, "provider": "openai-compatible"}
        except Exception:
            return baseline

