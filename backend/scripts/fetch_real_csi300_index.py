"""补一份真实沪深300指数点位，供 app/quant_v3/real_benchmark.py 用——
它和"候选池等权买入持有"那条基准线回答的是不同问题(见该模块顶部的
说明)，需要真实的官方指数点位(baostock `sh.000300`)，不是随便一条
近似曲线。

产出：data/real_csi300_index.parquet (列: date, close)

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_real_csi300_index.py
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "real_csi300_index.parquet"
START = "2010-01-01"
END = date.today().isoformat()


def main() -> None:
    bs.login()
    rs = bs.query_history_k_data_plus("sh.000300", "date,close", start_date=START, end_date=END, frequency="d")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        raise RuntimeError("baostock没有返回沪深300指数数据")
    frame = pd.DataFrame(rows, columns=["date", "close"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna()
    frame.to_parquet(OUT_PATH, index=False)
    print(f"落库完成：{OUT_PATH}，共 {len(frame)} 行，{frame['date'].min()} ~ {frame['date'].max()}")


if __name__ == "__main__":
    main()
