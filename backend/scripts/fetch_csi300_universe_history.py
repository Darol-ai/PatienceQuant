"""对齐主流做法（ADR待补）：universe 从自选30支候选池换成沪深300全部
真实成分股（tushare `pro.index_weight('000300.SH')`，300支，取最近一期
快照作为universe定义）。全部300支代码都是tushare真实返回的、和公司名字
保证正确对应，不存在Demo那批填充数据"代码和名字瞎配"的问题。

跟baostock时代的`bs.query_hs300_stocks()`有一个语义差异：baostock一次
返回"当前成分股+各自的加入/剔除日期"，tushare的index_weight()则是
"某一个具体交易日的成分股快照"，需要指定trade_date——这里从今天开始
往前最多扫30天(index_weight不是每天都发布新快照，通常每月更新)，取到
第一个有数据的交易日为止。

逐支拉取2010年至今的真实日线（部分股票上市更晚，各自从实际上市日
起算），落到 data/csi300_universe_history.parquet。按symbol维度增量
写入+跳过已完成的，可以安全中断后重跑。

连接层（限速/超时/失败重试）统一走 app.data.tushare_client，见
docs/adr/0046——这份脚本目前不会被运行（现有parquet保持不动），只是把
baostock时代的代码依赖换成tushare。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_csi300_universe_history.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.tushare_client import TushareQueryFailed, run as run_tushare
from app.quant_v3.tushare_adapter import fetch_daily_bars

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "csi300_universe_history.parquet"
CONSTITUENTS_PATH = DATA_DIR / "csi300_constituents.parquet"

START_DATE = "20100101"
END_DATE = "20260911"


def fetch_constituents() -> pd.DataFrame:
    def _fetch(pro):
        for offset in range(30):
            probe_day = (date.today() - timedelta(days=offset)).strftime("%Y%m%d")
            frame = pro.index_weight(index_code="000300.SH", trade_date=probe_day)
            if frame is not None and not frame.empty:
                return frame
        raise RuntimeError("沪深300成分股查询最近30天都没有数据")

    weight_frame = run_tushare(_fetch)
    frame = weight_frame.rename(columns={"con_code": "ts_code"})[["ts_code"]].drop_duplicates()

    def _fetch_names(pro):
        basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
        if basic is None or basic.empty:
            raise RuntimeError("stock_basic返回空表，无法补充成分股名称")
        return basic

    names = run_tushare(_fetch_names)
    frame = frame.merge(names, on="ts_code", how="left")
    frame["symbol"] = frame["ts_code"]
    frame.to_parquet(CONSTITUENTS_PATH, index=False)
    return frame


def fetch_one(ts_code: str) -> list[dict]:
    def _fetch(pro):
        return fetch_daily_bars(pro, ts_code, START_DATE, END_DATE)

    return run_tushare(_fetch)


def main() -> None:
    constituents = fetch_constituents()
    print(f"沪深300成分股：{len(constituents)}支，已存 {CONSTITUENTS_PATH}")

    done = set()
    if OUT_PATH.exists():
        done = set(pd.read_parquet(OUT_PATH)["symbol"].unique())
    remaining = constituents[~constituents.symbol.isin(done)]
    print(f"已完成 {len(done)}，剩余 {len(remaining)}")

    batch: list[dict] = []
    for i, row in enumerate(remaining.itertuples()):
        print(f"[{i + 1}/{len(remaining)}] {row.symbol} {row.name} ...", end=" ", flush=True)
        try:
            bars = fetch_one(row.ts_code)
        except TushareQueryFailed as exc:
            # 已经在连接层重试过3次(每次重新登录)才到这里——如实记录
            # 失败，不拿demo/编造数据顶替，下次重跑脚本会自动跳过已完成的
            # symbol、只补这支。
            print(f"重试3次后仍失败: {exc}")
            continue
        for bar in bars:
            bar["group"] = "沪深300"
        batch.extend(bars)
        print(f"{len(bars)} 条" if bars else "无数据")

        if (i + 1) % 20 == 0 or i == len(remaining) - 1:
            if batch:
                new_data = pd.DataFrame(batch)
                if OUT_PATH.exists():
                    existing = pd.read_parquet(OUT_PATH)
                    new_data = pd.concat([existing[~existing.symbol.isin(new_data.symbol.unique())], new_data], ignore_index=True)
                new_data.to_parquet(OUT_PATH, index=False)
                batch = []
            print(f"--- 已保存至 {i + 1}/{len(remaining)} ---", flush=True)

    final = pd.read_parquet(OUT_PATH)
    print(f"\n落库完成：{OUT_PATH}，共 {len(final)} 行，{final['symbol'].nunique()} 支股票")


if __name__ == "__main__":
    main()
