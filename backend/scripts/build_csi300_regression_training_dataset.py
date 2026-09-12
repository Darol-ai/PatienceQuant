"""对齐主流做法：universe从自选30支候选池换成沪深300全部真实成分股。
用`csi300_universe_history.parquet`(baostock真实抓取)重建horizon=90的
回归训练样本表，供LightGBM/XGBoost两个策略共用同一份特征/标签。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/build_csi300_regression_training_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.regression_dataset import build_regression_sample

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "csi300_universe_history.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "csi300_regression_h90_training_samples.parquet"


def main() -> None:
    raw = pd.read_parquet(IN_PATH)
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    for i, (symbol, group_df) in enumerate(raw.groupby("symbol", sort=False)):
        group_name = group_df["group"].iloc[0]
        bars = group_df.to_dict(orient="records")
        for t_index in range(len(bars)):
            sample = build_regression_sample(bars, t_index, group=group_name, horizon=90)
            if sample is None:
                continue
            sample["symbol"] = symbol
            rows.append(sample)
        if (i + 1) % 30 == 0:
            print(f"{i + 1} 支已处理")

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n共 {len(frame)} 条样本，{frame['symbol'].nunique()} 支股票，落库：{OUT_PATH}")


if __name__ == "__main__":
    main()
