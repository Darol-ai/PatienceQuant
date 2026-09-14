#!/usr/bin/env bash
# 一条命令跑完沪深300策略集 + 30支候选池策略需要的全部真实数据准备
# （抓取历史行情 + 构建训练样本 + 训练模型）。详细说明见
# ../../docs/部署-真实数据准备.md。
#
# 用法：
#   cd PatienceQuant/backend
#   pip install -e '.[real-data]'    # 装baostock（如果还没装）
#   bash scripts/prepare_real_data.sh
#
# 每一步都是增量的、可以安全中断后重跑——脚本本身也是幂等的，重复运行
# 只会跳过已经生成好的文件对应的工作，不会重新算一遍全部。
# 全程预计几十分钟到数小时，取决于机器性能和网络状况，不适合放进
# docker build，所以设计成在宿主机/训练环境手动跑一次。

set -euo pipefail
cd "$(dirname "$0")/.."

run_step() {
    local desc="$1"
    shift
    echo
    echo "=== $desc ==="
    python "$@"
}

run_step "1/8 沪深300真实成分股 + 历史行情" scripts/fetch_csi300_universe_history.py
run_step "2/8 真实沪深300指数点位"          scripts/fetch_real_csi300_index.py
run_step "3/8 沪深300回归训练样本"          scripts/build_csi300_regression_training_dataset.py
run_step "4/8 LightGBM沪深300策略训练"      scripts/train_csi300_lightgbm_ensemble_walkforward.py
run_step "5/8 XGBoost沪深300策略训练"       scripts/train_csi300_xgboost_ensemble_walkforward.py
run_step "6/8 30支候选池历史行情(2010年起)"  scripts/fetch_broad_universe_history_extended.py
run_step "7/8 30支候选池回归训练样本"        scripts/build_broad_regression_h90_extended_training_dataset.py
run_step "8/8 30支候选池策略训练"            scripts/train_broad_regression_h90_ensemble_walkforward.py

echo
echo "全部完成。data/ 目录下的文件可以直接给 docker-compose.yml 的绑定挂载使用——"
echo "启动 docker compose up 之前，把这个 backend/data/ 目录整个复制到部署机器的同一位置即可。"
