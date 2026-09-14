from __future__ import annotations

import json
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.ai.credentials import get_ai_credentials
from app.ai.explainer import RuleExplainer


class AIResearchService:
    def __init__(self):
        self.rules = RuleExplainer()

    def explain(self, context: Dict[str, Any], db: Session, use_llm: bool = False) -> Dict[str, Any]:
        baseline = {**self.rules.explain(context), "model_version": "rules", "confidence": None}
        credentials = get_ai_credentials(db)
        if not use_llm or not credentials.api_key:
            return baseline
        try:
            from openai import OpenAI

            client = OpenAI(api_key=credentials.api_key, base_url=credentials.base_url)
            response = client.chat.completions.create(
                model=credentials.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "你是量化投研解释助手，只解释已有量化信号，不得改变交易决定。输出简洁中文。"},
                    {"role": "user", "content": "请结合规则解释和量化上下文，生成研究摘要、交易原因和风险提示。上下文：%s；规则解释：%s" % (json.dumps(context, ensure_ascii=False), json.dumps(baseline, ensure_ascii=False))},
                ],
            )
            content = response.choices[0].message.content or baseline["content"]
            # 置信度：模型没有直接给出可信的数值置信度（没有 logprobs），
            # 宁可留空也不伪造一个看起来很精确的数字。
            return {**baseline, "content": content, "provider": "openai-compatible", "model_version": credentials.model, "confidence": None}
        except Exception:
            return baseline

