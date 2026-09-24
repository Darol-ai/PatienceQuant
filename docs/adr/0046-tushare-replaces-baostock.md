# 换数据源：baostock → tushare（1.4.24，非官方中转）

## 背景

ADR-0045刚把baostock的连接层做完（统一登录/超时/重试），验证过程中又
发现baostock本身响应不稳定：同一个干净的单一登录、同一个查询，前一次
5秒完成，紧接着下一次卡到90秒以上无界——这是baostock服务器本身的问题
（免费社区服务，没有SLA），不是我们连接层的缺陷，但用户自己有一份可用
的tushare访问（token+5000积分），决定直接换掉baostock。

## 决定

1. **baostock作为代码依赖全部移除**：连接层
   (`app/data/tushare_client.py`替代`baostock_client.py`)、
   实时/Dashboard路径(`app/data/tushare_provider.py`替代
   `baostock_provider.py`)、9个`scripts/fetch_*.py`训练数据抓取脚本、
   `app/quant_v3/tushare_adapter.py`替代`baostock_adapter.py`、
   `pyproject.toml`依赖。
2. **已有的25个parquet训练数据文件和已训好的模型原封不动，不重新抓取、
   不重新训练**——用户明确要求"改代码依赖"和"不影响已有训练结果"两件事
   都要满足。8个训练脚本代码上已经换成tushare，但目前不会被运行；如果
   将来真的要重新生成训练数据，产出的parquot在"停牌日是否显式占一行"
   这一点上跟baostock版本不完全一致（tushare的`daily()`对停牌日直接
   不返回那一行，没有做用`trade_cal()`反查交易日历、插入零成交行的
   重建——这段重建逻辑没有真实数据能验证，与其写一段没测过的代码不如
   老实留空，见`app/quant_v3/tushare_adapter.py`模块docstring），重新
   训练前需要留意这个差异。
3. **连接地址是`https://tuaremax.top`，不是tushare官方的
   `api.waditu.com`**——这是用户自己已知情、明确要求使用的非官方中转
   服务，token会经过这个第三方。已经向用户说明风险（token可能被转卖/
   滥用、数据真实性没有官方渠道可以独立验证），用户确认这是他自己知情
   购买的渠道、愿意承担这个风险、坚持要用。这条风险记录在这里，不是
   代码里悄悄咽下去的假设。已实测确认这个中转能正常转发
   `stock_basic`/`daily`/`adj_factor`/`index_daily`/`index_weight`
   这5个接口，返回真实、合理的数据（比如茅台在沪深300权重6.17%这种
   细节对得上真实情况）。
4. **前复权价格自己算**：tushare官方`daily()`是不复权原始价格，用
   `adj_factor()`拿复权因子手动算：`前复权价 = 原始价 × 当日复权因子 /
   查询区间内最新一天的复权因子`——锚定在每次查询自己的最后一个交易日，
   不是锚定"今天"，因为一次查询内部的相对收益/动量计算不受锚点选择
   影响，只影响绝对价格水平这个不参与计算的量。没有用`ts.pro_bar(...,
   adj='qfq')`这个官方封装——它内部要连续发好几次请求，非官方中转不一定
   支持这种"多跳"调用，拆开自己算更可控。
5. **连接层设计跟baostock时代不一样**：tushare是无状态HTTP+token（每次
   `requests.post()`），不是baostock那种要维护一个独占全局socket的会话，
   所以不需要ADR-0045里那把进程级锁；改成每次请求间主动停顿0.3~0.5秒、
   失败重试3次指数退避、单次15秒超时（`requests`自带，不用像baostock
   那样自己给裸socket打补丁）——偏保守，因为非官方中转的限流规则未知，
   目标是"不触发对方限流"而不是"多快抓完"。"real优先、3次不行就如实
   失败、不用demo数据顶替"这条主线原则不变，从ADR-0045延续过来。
6. **`backend/data/`（parquet+训好的模型，333MB）改用Git LFS分发**——
   这个问题跟换不换tushare无关，是本来就存在的：这个目录之前完全没进
   git（两层`.gitignore`都排除），任何人`git clone`拿到的都是空目录，
   `prepare_real_data.sh`要真的重新跑一遍才有数据。换成Git LFS后正常
   `git clone`（装了`git-lfs`）就能直接拿到这批文件；`prepare_real_data.
   sh`默认会先检查这批"最终产出"文件是否已存在，存在就跳过整套抓取+
   训练，不再重复劳动，也不会不必要地消耗tushare积分配额。

## 影响范围之外（刻意不做的事）

- 不动`akshare`——它是另一个未启用的可选依赖，跟这次换baostock没关系。
- 不做"tushare数据 vs baostock数据"的一致性比对——用户已确认这一支
  parquot保持不动，没有新旧数据可比对，比对也没有意义。
- 不实现"整体请求墙钟超时"（只有单次recv/单次HTTP请求超时）——用户
  确认先观察现有方案(3次重试+单次超时)在真实使用中是否还会出现长时间
  卡顿的问题，视情况再决定要不要加。

## 验证

新代码（连接层、provider、9个抓取脚本、`tushare_adapter.py`）全部
`import`验证通过；`app/quant_v3/tushare_adapter.py`里可单元测试的纯
函数部分（`bars_from_merged_frame`）有新测试
(`tests/quant_v3/test_tushare_adapter.py`)覆盖前复权计算/字段映射/
缺失值归零/pbMRQ提取；全量测试套件224个测试通过（测试环境`DATA_MODE=
demo`，不触发真实tushare网络调用，所以不受tuaremax.top当时是否可用
影响）。**没有对`/api/dashboard`等真实网络路径做端到端live验证**——
按用户要求"先改完，然后再测试"，代码改完即止，真实网络验证留到用户
自己测试环境。
