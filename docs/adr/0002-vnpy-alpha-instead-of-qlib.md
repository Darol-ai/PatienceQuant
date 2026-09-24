# 用 vnpy 内置的 `vnpy.alpha` 模块替代 Qlib

[[ADR-0001]] 定为"Qlib 出信号 + vnpy 执行"。下载 vnpy 4.0 源码后发现它自带 `vnpy/alpha` 模块——
因子工程（含直接搬运自 Qlib 的 Alpha158）、LightGBM/Lasso/MLP 训练模板、`AlphaStrategy` +
`BacktestingEngine`，设计上明确"参照 Qlib"。

## 决定

选股研究与回测改在同一个 vnpy 生态里做，不再单独引入 Qlib。信号产出用
`vnpy.alpha.dataset`/`model`，执行仍是 vnpy 的 `AlphaStrategy` 模板 + `vnpy_portfoliostrategy`/
`vnpy_paperaccount`（独立安装包）。

## 理由

少一个框架、少一道"Qlib 信号 → vnpy 执行"的拼接缝，`vnpy.alpha` 依赖也轻（`polars`/
`lightgbm`，不需要 `torch` 除非用它的 MLP 模型，我们不用）。这不是重新评估后改变主意，是拿到
代码之前不知道这个模块存在。
