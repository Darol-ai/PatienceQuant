"""智能体推荐（ADR-0047 第 8 条）。

选哪个策略完全由规则决定：按用户的市场、可接受回撤、持仓周期在成绩卡里筛选排序，
只推荐一个。大模型只负责把选择理由讲清楚，而且它写出来的百分比数字必须都能在
成绩卡里找到，否则丢弃它的回答、改用规则生成的解释。智能体不能下单，只给出去
「策略实践」回测的链接。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.ai.credentials import get_ai_credentials

HOLDING_TO_FREQUENCY = {"short": "weekly", "medium": "monthly", "long": "quarterly"}
HOLDING_LABEL = {"short": "短（每周调仓）", "medium": "中（每月调仓）", "long": "长（每季度调仓）", "any": "不限"}
FREQUENCY_LABEL = {"weekly": "每周", "monthly": "每月", "quarterly": "每季度"}
MARKET_LABEL = {"a_share": "A股"}


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _facts(card: Dict[str, Any]) -> Dict[str, Any]:
    """交给大模型的事实：只有成绩卡里的数字（换算成百分比、保留一位小数）。"""
    m = card["metrics"]
    return {
        "策略": card["name"], "说明": card["description"], "股票池": card["universe"],
        "调仓频率": FREQUENCY_LABEL.get(card["rebalance_frequency"], card["rebalance_frequency"]),
        "回测区间": f"{card['standard']['start_date']} 至 {card['standard']['end_date']}",
        "年化收益": _pct(m.get("annual_return")), "累计收益": _pct(m.get("total_return")),
        "最大回撤": _pct(m.get("max_drawdown")), "年化波动": _pct(m.get("volatility")),
        "夏普比率": None if m.get("sharpe") is None else round(m["sharpe"], 2),
        "同期沪深300累计": _pct(m.get("benchmark_return")), "超额收益": _pct(m.get("excess_return")),
        "平均仓位": _pct(card.get("average_exposure")),
        "跑赢沪深300的年份": f"{card['years_beating_benchmark']}/{card['years']}",
        "最差年份": None if not card.get("worst_year") else f"{card['worst_year']['year']} 年 {_pct(card['worst_year']['strategy'])}",
        "最好年份": None if not card.get("best_year") else f"{card['best_year']['year']} 年 {_pct(card['best_year']['strategy'])}",
        "逐年收益": {str(y["year"]): _pct(y["strategy"]) for y in card.get("yearly", [])},
    }


def rank(cards: List[Dict[str, Any]], market: str, max_drawdown: Optional[float], holding: str) -> Dict[str, Any]:
    """返回 {chosen, alternatives, rules, notes}。max_drawdown 是正数（如 0.25 表示能接受 −25%），None 表示不限。"""
    rules = [f"只看市场为「{MARKET_LABEL.get(market, market)}」且已经有成绩卡的策略"]
    notes: List[str] = []
    pool = [c for c in cards if c["status"] == "ready" and c["market"] == market and c["metrics"].get("sharpe") is not None]
    if not pool:
        return {"chosen": None, "alternatives": [], "rules": rules, "notes": ["还没有可用的成绩卡，先在策略推荐页点「计算成绩卡」"]}

    if max_drawdown is not None:
        rules.append(f"最大回撤不超过 −{max_drawdown * 100:.0f}%")
        passing = [c for c in pool if c["metrics"]["max_drawdown"] >= -max_drawdown]
        if not passing:
            best = max(pool, key=lambda c: c["metrics"]["max_drawdown"])
            notes.append(f"没有策略的最大回撤在 −{max_drawdown * 100:.0f}% 以内，下面给的是回撤最小的一个"
                         f"（{_pct(best['metrics']['max_drawdown'])}），它超出了你能接受的范围")
            passing = [best]
        pool = passing

    if holding != "any":
        frequency = HOLDING_TO_FREQUENCY[holding]
        rules.append(f"优先调仓频率为{FREQUENCY_LABEL[frequency]}的策略")
        matching = [c for c in pool if c["rebalance_frequency"] == frequency]
        if matching:
            pool = matching
        else:
            notes.append(f"符合回撤要求的策略里没有{FREQUENCY_LABEL[frequency]}调仓的，忽略持仓周期这一条")

    rules.append("按夏普比率（单位波动换来的收益）从高到低排序，推荐第一个")
    ordered = sorted(pool, key=lambda c: c["metrics"]["sharpe"], reverse=True)
    return {"chosen": ordered[0], "alternatives": ordered[1:3], "rules": rules, "notes": notes}


def template_explanation(chosen: Dict[str, Any], prefs_text: str) -> str:
    f = _facts(chosen)
    return (
        f"按你的偏好（{prefs_text}），推荐「{f['策略']}」。在 {f['回测区间']} 的标准回测里，它年化 {f['年化收益']}、"
        f"最大回撤 {f['最大回撤']}、夏普 {f['夏普比率']}，同期沪深300累计 {f['同期沪深300累计']}；"
        f"{f['跑赢沪深300的年份']} 个年份跑赢沪深300，最差的是{f['最差年份']}。"
        f"平均仓位 {f['平均仓位']}，{f['调仓频率']}调仓。历史回测不代表未来收益，建议先在「策略实践」里换几个区间回测，再决定是否用到模拟盘。"
    )


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?\s*%")


def numbers_are_grounded(text: str, facts: List[Dict[str, Any]]) -> bool:
    """回答里每个百分比数字都要能在事实里找到（允许 0.1 个百分点的四舍五入差）。"""
    allowed = set()
    for fact in facts:
        for value in re.findall(r"-?\d+(?:\.\d+)?(?=%)", json.dumps(fact, ensure_ascii=False)):
            allowed.add(round(float(value), 1))
    for token in _NUMBER.findall(text):
        value = round(float(token.replace("%", "").strip()), 1)
        if not any(abs(value - a) <= 0.1 + 1e-9 or abs(-value - a) <= 0.1 + 1e-9 for a in allowed):
            return False
    return True


def explain(db: Session, result: Dict[str, Any], prefs_text: str, question: str) -> Dict[str, Any]:
    chosen = result["chosen"]
    template = template_explanation(chosen, prefs_text)
    credentials = get_ai_credentials(db)
    if not credentials.api_key:
        return {"content": template, "provider": "rules", "model_version": "rules", "grounded": True}
    facts = [_facts(chosen)] + [_facts(c) for c in result["alternatives"]]
    try:
        from openai import OpenAI

        client = OpenAI(api_key=credentials.api_key, base_url=credentials.base_url)
        response = client.chat.completions.create(
            model=credentials.model, temperature=0.2,
            messages=[
                {"role": "system", "content": (
                    "你是策略推荐解释助手。推荐哪个策略已经由规则选好了，你不能改选、不能推荐多个组合、不能建议下单。"
                    "你只能引用下面给出的事实里的数字，不能自己计算或编造任何数字，不承诺未来收益。"
                    "用简洁中文说明：为什么这个策略符合用户偏好、它的主要风险（最差年份、回撤、仓位），"
                    "以及和备选策略相比的取舍。最后提醒先去「策略实践」里回测验证。"
                    "输出纯文本，不要用 Markdown（不要 ** 加粗、不要 # 标题），分段用换行。")},
                {"role": "user", "content": json.dumps({
                    "用户偏好": prefs_text, "用户的问题": question or "无", "筛选排序规则": result["rules"],
                    "规则备注": result["notes"], "推荐策略的事实": facts[0], "备选策略的事实": facts[1:],
                }, ensure_ascii=False)},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return {"content": template, "provider": "rules", "model_version": "rules", "grounded": True,
                "llm_error": f"调用大模型失败，已改用规则生成的解释：{type(exc).__name__}"}
    if not content:
        return {"content": template, "provider": "rules", "model_version": "rules", "grounded": True}
    # 用户自己说的偏好（如"−25%"）和筛选规则里的数字也允许复述
    context = {"用户偏好": prefs_text, "规则": result["rules"], "备注": result["notes"]}
    if not numbers_are_grounded(content, facts + [context]):
        return {"content": template, "provider": "rules", "model_version": credentials.model, "grounded": False,
                "llm_error": "大模型的回答里出现了成绩卡以外的数字，已丢弃，改用规则生成的解释",
                "rejected_content": content}
    return {"content": content, "provider": "openai-compatible", "model_version": credentials.model, "grounded": True}
