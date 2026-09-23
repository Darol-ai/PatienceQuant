"""补一份真实沪深300指数点位，供 app/quant_v3/real_benchmark.py 用——
它和"候选池等权买入持有"那条基准线回答的是不同问题(见该模块顶部的
说明)，需要真实的官方指数点位(tushare `index_daily('000300.SH')`)，不是
随便一条近似曲线。

产出：data/real_csi300_index.parquet (列: date, close)

连接层（限速/超时/失败重试）统一走 app.data.tushare_client，见
docs/adr/0046——这份脚本目前不会被运行（现有parquet保持不动），只是把
baostock时代的代码依赖换成tushare。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_real_csi300_index.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.tushare_client import run as run_tushare

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "real_csi300_index.parquet"
START = "20100101"
END = date.today().strftime("%Y%m%d")


def main() -> None:
    def _fetch(pro):
        frame = pro.index_daily(ts_code="000300.SH", start_date=START, end_date=END)
        if frame is None or frame.empty:
            raise RuntimeError("tushare index_daily(沪深300)返回空表")
        return frame

    frame = run_tushare(_fetch)
    frame = frame[["trade_date", "close"]].rename(columns={"trade_date": "date"})
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().sort_values("date").reset_index(drop=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"落库完成：{OUT_PATH}，共 {len(frame)} 行，{frame['date'].min()} ~ {frame['date'].max()}")


if __name__ == "__main__":
    main()
