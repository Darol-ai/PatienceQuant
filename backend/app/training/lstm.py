"""LSTM 模型（ADR-0053，第二批算法只做这一种）。

和树模型用同一套因子库、同一个训练股票池、同一条滚动规则（给 Y 年打分的模型只用
Y−1 年 6 月底前已揭晓的样本训练、Y−1 年下半年做早停验证），不同之处：

- 输入是每只股票过去 SEQ_LEN 个交易日的因子序列，不是当天一行；
- 每天的因子先换成当天在面板里的百分位再减 0.5（神经网络需要统一量纲），缺值填 0；
  打分时用同样的做法，保证训练和预测口径一致；
- 标签是未来收益在当天股票池里的百分位减 0.5；
- 相邻交易日的样本几乎一样（标签窗口 90 天大部分重叠），训练只每 SAMPLE_EVERY 天取一次样本。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SEQ_LEN = 20
HIDDEN = 32
SAMPLE_EVERY = 5
MAX_EPOCHS = 15
PATIENCE = 3
BATCH = 2048
LEARNING_RATE = 1e-3


def ranked_array(values: Dict[str, pd.DataFrame], keys: Sequence[str], suspended: pd.DataFrame) -> np.ndarray:
    """(交易日, 股票, 因子) 的百分位数组：每天在当天有交易的股票里的百分位 − 0.5，缺值为 0。

    停牌日不参与排名：训练面板里停牌日的价格沿用前收（因为知道它后来复牌），打分面板的最后一天
    停牌的股票却没有值；都排除才能让训练和打分的排名口径一致，也不借用"后来会复牌"这个未来信息。
    """
    trading = suspended.reindex_like(values[keys[0]]) == 0
    stacked = [values[k].where(trading).rank(axis=1, pct=True).sub(0.5).fillna(0.0).to_numpy(dtype=np.float32) for k in keys]
    return np.stack(stacked, axis=-1)


def all_present(values: Dict[str, pd.DataFrame], keys: Sequence[str]) -> np.ndarray:
    """(交易日, 股票)：当天全部输入因子都有值。和树模型一样，当天缺因子的股票不打分。"""
    return np.logical_and.reduce([values[k].notna().to_numpy() for k in keys])


def _net(n_features: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, HIDDEN, batch_first=True)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    torch.set_num_threads(4)
    return Net()


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _gather(arr, t_idx, n_idx):
    """从 (T, N, F) 数组取出以 t_idx 结尾的 SEQ_LEN 天序列 → (B, SEQ_LEN, F)。"""
    import torch

    offsets = torch.arange(-SEQ_LEN + 1, 1, device=arr.device)
    return arr[t_idx[:, None] + offsets[None, :], n_idx[:, None]]


def _predict(model, arr, t_idx, n_idx) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(t_idx), BATCH * 4):
            out.append(model(_gather(arr, t_idx[start:start + BATCH * 4], n_idx[start:start + BATCH * 4])).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def _fit(arr, target, train, calib, seed: int, cancelled: Callable[[], bool]):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _net(arr.shape[-1]).to(arr.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    t_train, n_train = train
    best, best_loss, stale = None, np.inf, 0
    use_calib = len(calib[0]) >= 60
    for _ in range(MAX_EPOCHS if use_calib else 5):
        if cancelled():
            raise InterruptedError("训练已取消")
        model.train()
        order = torch.as_tensor(rng.permutation(len(t_train)), device=arr.device)
        for start in range(0, len(order), BATCH):
            pick = order[start:start + BATCH]
            t, n = t_train[pick], n_train[pick]
            optimizer.zero_grad()
            loss = loss_fn(model(_gather(arr, t, n)), target[t, n])
            loss.backward()
            optimizer.step()
        if not use_calib:
            continue
        calib_pred = torch.as_tensor(_predict(model, arr, *calib), device=arr.device)
        calib_loss = float(loss_fn(calib_pred, target[calib[0], calib[1]]))
        if calib_loss < best_loss - 1e-6:
            best_loss, stale = calib_loss, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best is not None:
        model.load_state_dict(best)
    return model


def train_lstm(config, model_id: str, until: date, progress: Callable[[str], None],
               cancelled: Callable[[], bool] = lambda: False) -> dict:
    import torch

    from app.training.trainer import FIRST_TEST_YEAR, SEEDS, _pool_panels, evaluate_year, model_dir

    values, panels, membership = _pool_panels(config, until, progress)
    close = panels["close"]
    keys = list(config.factors)
    dates, symbols = close.index, list(close.columns)
    present = all_present(values, keys) & (panels["suspended"] == 0).to_numpy()
    progress("整理序列")
    arr_np = ranked_array(values, keys, panels["suspended"])

    forward = (close.shift(-config.horizon) / close - 1).replace([np.inf, -np.inf], np.nan)
    if membership is not None:
        member = np.array([[s in membership.at(d.date()) for s in symbols] for d in dates])
    else:
        member = close.notna().to_numpy()
    in_pool = forward.where(member)
    target_np = in_pool.rank(axis=1, pct=True).sub(0.5).to_numpy(dtype=np.float32)
    label_end = pd.Series(dates, index=dates).shift(-config.horizon)
    t_all = np.arange(len(dates))
    usable = member & present & np.isfinite(target_np) & (t_all[:, None] >= SEQ_LEN - 1)

    device = _device()
    arr = torch.as_tensor(arr_np, device=device)
    target = torch.as_tensor(np.nan_to_num(target_np), device=device)
    root = model_dir(model_id)
    results = {}

    def rows(mask: np.ndarray):
        t, n = np.nonzero(mask)
        return torch.as_tensor(t, device=device), torch.as_tensor(n, device=device)

    for year in range(FIRST_TEST_YEAR, until.year + 1):
        train_end = pd.Timestamp(f"{year - 1}-07-01")
        calib_start, calib_end = pd.Timestamp(f"{year - 1}-07-01"), pd.Timestamp(f"{year - 1}-12-31")
        sampled = (t_all % SAMPLE_EVERY == 0)[:, None]
        known_before = (label_end < train_end).to_numpy()[:, None]
        in_calib = np.asarray((dates >= calib_start) & (dates <= calib_end))[:, None] & (label_end < calib_end).to_numpy()[:, None]
        train_mask = usable & sampled & known_before
        if train_mask.sum() < 1000:
            results[str(year)] = {"skipped": f"训练样本只有 {int(train_mask.sum())} 条，不训练这一年"}
            continue
        calib_mask = usable & sampled & in_calib
        test_mask = usable & np.asarray(dates.year == year)[:, None]
        train, calib, test = rows(train_mask), rows(calib_mask), rows(test_mask)
        predictions = []
        for index, seed in enumerate(SEEDS[:config.seeds]):
            if cancelled():
                raise InterruptedError("训练已取消")
            progress(f"{year} 年 · 种子 {index + 1}/{config.seeds}")
            model = _fit(arr, target, train, calib, seed, cancelled)
            out = root / str(year) / f"seed{index}"
            out.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "lstm_model.pt")
            (out / "metadata.json").write_text(json.dumps({"feature_columns": keys, "group_categories": [], "framework": "lstm",
                                                           "seq_len": SEQ_LEN, "hidden": HIDDEN}, ensure_ascii=False), encoding="utf-8")
            if len(test[0]):
                predictions.append(_predict(model, arr, *test))
        summary = {"train_samples": int(train_mask.sum()), "calib_samples": int(calib_mask.sum())}
        if predictions:
            t_idx, n_idx = np.nonzero(test_mask)
            frame = pd.DataFrame({"date": dates[t_idx], "forward_return": forward.to_numpy()[t_idx, n_idx]})
            summary.update(evaluate_year(frame, np.mean(predictions, axis=0)))
        results[str(year)] = summary
    return results


def load_predictor(seed_dir: Path, n_features: int):
    """读一个种子的 LSTM，返回 predict(序列数组 (B, SEQ_LEN, F)) → 分数。打分在 CPU 上做。"""
    import torch

    model = _net(n_features)
    model.load_state_dict(torch.load(seed_dir / "lstm_model.pt", map_location="cpu"))
    model.eval()

    def predict(sequences: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.as_tensor(sequences, dtype=torch.float32)).numpy()
    return predict


def sequences_on(ranked: np.ndarray, day_index: int, columns: List[int]) -> Optional[np.ndarray]:
    """打分日的输入：第 day_index 天结尾的 SEQ_LEN 天序列，只取 columns 这些股票。"""
    if day_index < SEQ_LEN - 1 or not columns:
        return None
    window = ranked[day_index - SEQ_LEN + 1: day_index + 1, columns, :]  # (SEQ_LEN, B, F)
    return np.transpose(window, (1, 0, 2))
