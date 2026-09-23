"""策略规格：一个策略由哪几个插槽组成、每个插槽怎么设（ADR-0047 第 2、3 条）。

    本地行情库 → 因子 → 打分 → ① 选股规则 → ② 权重方案 → ③ × 择时仓位 → ④ 调仓

规格只描述"策略本身"——市场、打分方式、择时信号、选股规则、权重方案、调仓
频率。回测起止日期、初始资金、股票池、费用属于每次回测，不在这里。
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class FactorWeightScorerSpec(BaseModel):
    """因子权重打分：每组因子在股票池里的百分位排名按权重加总。"""

    type: Literal["factor_weights"] = "factor_weights"
    weights: Dict[str, float]

    @model_validator(mode="after")
    def _positive_total(self):
        if sum(max(v, 0.0) for v in self.weights.values()) <= 0:
            raise ValueError("因子权重合计必须大于 0")
        return self


class ModelScorerSpec(BaseModel):
    """模型打分：一个或多个模型库里的模型预测分数取平均。"""

    type: Literal["model"] = "model"
    models: List[str] = Field(min_length=1)


ScorerSpec = Union[FactorWeightScorerSpec, ModelScorerSpec]


class NoTimingSpec(BaseModel):
    type: Literal["none"] = "none"


class IndexTrendTimingSpec(BaseModel):
    """指数跌破 200 日均线降到 risk_off_exposure；跌破 50 日均线降到两者中间。"""

    type: Literal["index_trend"] = "index_trend"
    risk_off_exposure: float = Field(0.75, ge=0, le=1)


class RsrsTimingSpec(BaseModel):
    """RSRS 标准分择时（光大证券 2017）：高于 threshold 满仓，跌破 −threshold 空仓。"""

    type: Literal["rsrs"] = "rsrs"
    n: int = Field(18, ge=5, le=60)
    m: int = Field(600, ge=100, le=1200)
    threshold: float = Field(0.7, gt=0, le=3)


class IcuMaTimingSpec(BaseModel):
    """ICU 均线择时（中泰证券 2023）：收盘价在 n 日稳健回归均线之上满仓，之下空仓。"""

    type: Literal["icu_ma"] = "icu_ma"
    n: int = Field(5, ge=3, le=250)


class AlligatorTimingSpec(BaseModel):
    """鳄鱼线组合择时（招商证券 2024）：鳄鱼线 + AO + 分形 + MACD，参数沿用原书/研报。"""

    type: Literal["alligator"] = "alligator"


class LltTimingSpec(BaseModel):
    """低延迟趋势线 LLT（广发证券 2017）：LLT 上行时满仓。"""

    type: Literal["llt"] = "llt"
    d: int = Field(30, ge=5, le=120)


class MaChannelTimingSpec(BaseModel):
    """均线交叉结合通道突破（申万宏源 2018）。"""

    type: Literal["ma_channel"] = "ma_channel"
    short: int = Field(9, ge=2, le=60)
    long: int = Field(18, ge=5, le=250)
    n: int = Field(3, ge=1, le=20)


class OneWayVolTimingSpec(BaseModel):
    """单向波动差（国信证券 2015）。"""

    type: Literal["one_way_vol"] = "one_way_vol"
    window: int = Field(60, ge=5, le=250)


class RpsVolTimingSpec(BaseModel):
    """相对强弱 RPS 下的单向波动差（国信证券 2015）。"""

    type: Literal["rps_vol"] = "rps_vol"
    period: int = Field(13, ge=2, le=60)


class HighMomentTimingSpec(BaseModel):
    """指数高阶矩（广发证券 2015）：5 阶矩的 EMA 上升时满仓。"""

    type: Literal["high_moment"] = "high_moment"
    ema: int = Field(90, ge=10, le=250)


class VolumeResonanceTimingSpec(BaseModel):
    """价量共振（华创证券 2019），参数同研报。"""

    type: Literal["volume_resonance"] = "volume_resonance"


class QrsTimingSpec(BaseModel):
    """QRS（中金公司 2021）：RSRS 标准分 × R²。"""

    type: Literal["qrs"] = "qrs"
    n: int = Field(18, ge=5, le=60)
    m: int = Field(600, ge=100, le=1200)
    threshold: float = Field(0.7, gt=0, le=3)


TimingSpec = Union[NoTimingSpec, IndexTrendTimingSpec, RsrsTimingSpec, IcuMaTimingSpec, AlligatorTimingSpec,
                   LltTimingSpec, MaChannelTimingSpec, OneWayVolTimingSpec, RpsVolTimingSpec, HighMomentTimingSpec,
                   VolumeResonanceTimingSpec, QrsTimingSpec]


class SelectionSpec(BaseModel):
    """① 选股规则：前 N 名或前 x%。

    不提供"分数超过某值才买"——不同模型、不同因子组合的分数量纲不同，
    用户给不出有意义的阈值（ADR-0047）。
    """

    type: Literal["top_n", "top_pct"] = "top_n"
    n: Optional[int] = Field(None, ge=1)
    pct: Optional[float] = Field(None, gt=0, le=1)

    @model_validator(mode="after")
    def _has_size(self):
        if self.type == "top_n" and not self.n:
            raise ValueError("前 N 名需要给出 n")
        if self.type == "top_pct" and not self.pct:
            raise ValueError("前 x% 需要给出 pct")
        return self


class WeightingSpec(BaseModel):
    """② 权重方案：等权或按分数加权，再加单股上限（超出部分留作现金）。"""

    type: Literal["equal", "score"] = "equal"
    max_weight: float = Field(1.0, gt=0, le=1)


class RebalanceSpec(BaseModel):
    """④ 调仓：频率 + 换手阈值（目标与当前比例差小于它就不动）。"""

    frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    turnover_band: float = Field(0.0, ge=0, le=0.5)


class StrategySpec(BaseModel):
    market: Literal["a_share"] = "a_share"
    scorer: ScorerSpec = Field(discriminator="type")
    timing: TimingSpec = Field(default_factory=NoTimingSpec, discriminator="type")
    selection: SelectionSpec
    weighting: WeightingSpec = Field(default_factory=WeightingSpec)
    rebalance: RebalanceSpec = Field(default_factory=RebalanceSpec)
