"""tushare统一连接层（ADR-0046）——参考baostock时代的连接层设计
（docs/adr/0045），但连接模型不一样：tushare是无状态HTTP+token（每次
query()内部就是一次requests.post()），不像baostock要维护一个独占的全局
socket会话，所以不需要进程级锁；单次请求超时用requests自带的timeout
（不用像baostock那样自己给裸socket打补丁）。

- 这个token走的是非官方中转（https://tuaremax.top，不是tushare官方的
  api.waditu.com）——用户已知情并明确要求使用这个地址，见ADR-0046。
  官方的按积分分层限流规则对这个中转不一定适用/未知，所以主动加一个
  保守的请求间隔，降低触发对方限流的概率，不追求"多快抓完"。
- 失败重试3次，指数退避。3次都失败要如实抛出异常，调用方不能悄悄换成
  demo/虚构数据顶替真实结果——这条原则从ADR-0045延续到ADR-0046，没变。
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, TypeVar

import tushare as ts

T = TypeVar("T")

# token和中转地址从 backend/.env 读(TUSHARE_TOKEN / TUSHARE_HTTP_URL)，
# 不写进代码——这个仓库会推到GitHub，写在代码里等于公开token。
_REQUEST_TIMEOUT_SECONDS = 15
_MAX_ATTEMPTS = 3
_MIN_PACING_SECONDS = 0.3
_MAX_PACING_SECONDS = 0.5

_pacing_lock = threading.Lock()
_last_call_at = 0.0
_pro = None
_pro_lock = threading.Lock()


class TushareQueryFailed(Exception):
    """重试耗尽后的如实失败——调用方必须让这次真实数据请求可见地失败，
    不能捕获这个异常之后悄悄换成demo/虚构数据（ADR-0045/ADR-0046）。"""


def _get_pro():
    global _pro
    if _pro is None:
        with _pro_lock:
            if _pro is None:
                # baostock时代这台机器走本地代理对国内数据站点路由不稳定，
                # 直连更可靠（docs/setup.md）；tushare走的是requests，会
                # 默认读HTTP_PROXY等环境变量，这里延续同样的直连策略。
                import os

                from app.config import get_settings

                settings = get_settings()
                if not settings.tushare_token:
                    raise TushareQueryFailed("未配置 TUSHARE_TOKEN（在 backend/.env 里设置）")
                for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                    os.environ.pop(var, None)
                client = ts.pro_api(settings.tushare_token, timeout=_REQUEST_TIMEOUT_SECONDS)
                client._DataApi__token = settings.tushare_token
                client._DataApi__http_url = settings.tushare_http_url
                _pro = client
    return _pro


def _pace() -> None:
    """主动限速：距上次请求不足这个间隔就睡到够——见模块docstring。"""
    global _last_call_at
    with _pacing_lock:
        wait = _last_call_at + random.uniform(_MIN_PACING_SECONDS, _MAX_PACING_SECONDS) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def run(fn: Callable[[object], T]) -> T:
    """在配置好的tushare pro客户端上执行 fn(pro) -> T。

    fn拿到的是已经配置好token/中转地址的pro客户端，可以直接调用
    stock_basic/daily/adj_factor/index_daily/index_weight等任意接口——
    具体要哪些字段、怎么转换股票代码，完全由调用方决定，本函数不介入。
    失败（网络异常、超时、返回结构不对，由fn自己判断并抛异常）会整体
    重试，最多3次，指数退避（0.5s→1s→2s）。

    3次都失败会抛TushareQueryFailed，调用方必须让这次请求可见地失败，
    不能捕获后静默换成demo/虚构数据（ADR-0045/ADR-0046）。
    """
    pro = _get_pro()
    last_error: Exception = TushareQueryFailed("从未成功尝试过")
    for attempt in range(_MAX_ATTEMPTS):
        _pace()
        try:
            return fn(pro)
        except Exception as exc:  # noqa: BLE001 - 任何失败都要重试，不区分异常类型
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(0.5 * (2 ** attempt))
    raise TushareQueryFailed(f"tushare查询连续{_MAX_ATTEMPTS}次失败: {last_error}") from last_error
