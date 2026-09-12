"""对齐主流做法（ADR待补）：universe 从自选30支候选池换成沪深300全部
真实成分股（baostock `bs.query_hs300_stocks()`，300支，逐年成分股名单
本身会变化，这里取当前最新一期作为universe定义）。全部300支代码都是
baostock真实返回的、和公司名字保证正确对应，不存在Demo那批填充数据
"代码和名字瞎配"的问题。

逐支拉取2010年至今的真实日线（部分股票上市更晚，各自从实际上市日
起算），落到 data/csi300_universe_history.parquet。按symbol维度增量
写入+跳过已完成的，可以安全中断后重跑。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_csi300_universe_history.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.baostock_adapter import normalize_daily_bars
from scripts.fetch_a_phase_history import FIELDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "csi300_universe_history.parquet"
CONSTITUENTS_PATH = DATA_DIR / "csi300_constituents.parquet"

START_DATE = "2010-01-01"
END_DATE = "2026-09-11"


def fetch_constituents() -> pd.DataFrame:
    rs = bs.query_hs300_stocks()
    if rs.error_code != "0":
        raise RuntimeError(f"沪深300成分股查询失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    frame = pd.DataFrame(rows, columns=rs.fields)
    frame = frame.rename(columns={"code": "bao_code", "code_name": "name"})
    frame["symbol"] = frame["bao_code"].apply(lambda c: f"{c.split('.')[1]}.{'SH' if c.startswith('sh') else 'SZ'}")
    frame.to_parquet(CONSTITUENTS_PATH, index=False)
    return frame


def fetch_one(bao_code: str) -> list[dict]:
    rs = bs.query_history_k_data_plus(bao_code, FIELDS, start_date=START_DATE, end_date=END_DATE, frequency="d", adjustflag="2")
    if rs.error_code != "0":
        raise RuntimeError(f"{bao_code} 查询失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return normalize_daily_bars(rows)


def main() -> None:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    try:
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
                bars = fetch_one(row.bao_code)
            except Exception as exc:  # noqa: BLE001 - 单支失败不能中断整批
                print(f"失败: {exc}")
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
    finally:
        bs.logout()

    final = pd.read_parquet(OUT_PATH)
    print(f"\n落库完成：{OUT_PATH}，共 {len(final)} 行，{final['symbol'].nunique()} 支股票")


if __name__ == "__main__":
    main()
