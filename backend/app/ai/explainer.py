from __future__ import annotations

from typing import Any, Dict


class RuleExplainer:
    def explain(self, context: Dict[str, Any]) -> Dict[str, Any]:
        symbol = context.get("symbol", "该股票")
        action = context.get("action", "HOLD")
        score = float(context.get("score", 0) or 0)
        rank = context.get("rank", "—")
        industry = context.get("industry", "所属行业")
        volatility = float(context.get("volatility", 0) or 0)
        drawdown = float(context.get("max_drawdown", 0) or 0)
        weight = float(context.get("target_weight", 0) or 0)
        if action == "BUY":
            headline = "%s：综合评分进入组合候选，建议关注买入执行。" % symbol
            body = "%s 在当前截面排名第 %s，综合评分 %.1f 分，目标权重 %.1f%%。基本面、盈利质量和动量因子共同提供支撑。" % (symbol, rank, score, weight * 100)
        elif action == "SELL":
            headline = "%s：综合评分或目标权重下降，触发减仓/卖出。" % symbol
            body = "%s 的目标权重已经下降，当前策略优先把资金分配给排名更高的股票。该股票近一年波动率约 %.1f%%，历史最大回撤约 %.1f%%，需要控制组合风险。" % (symbol, volatility * 100, drawdown * 100)
        else:
            headline = "%s：策略暂不产生明确交易动作。" % symbol
            body = "%s 当前属于 %s，综合评分 %.1f 分，排名第 %s。建议结合估值、动量和风险指标继续观察。" % (symbol, industry, score, rank)
        risks = []
        if volatility > .35:
            risks.append("波动率偏高，仓位不宜激进")
        if drawdown > .25:
            risks.append("历史回撤较深，需要设置风险预算")
        if not risks:
            risks.append("因子解释基于历史数据，不代表未来收益")
        return {"symbol": symbol, "action": action, "headline": headline, "content": body, "risks": risks, "provider": "rules"}

