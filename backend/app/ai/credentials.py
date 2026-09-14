"""OpenAI key/base_url/model的有效配置解析——数据库配置优先，没人在前端
配置过时才回退读.env。这样部署时改key不用重启Docker容器：DB配置随时
通过 POST /api/settings/ai 写入，下一次AI调用立即生效。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import AppSetting

KEY_API_KEY = "openai_api_key"
KEY_BASE_URL = "openai_base_url"
KEY_MODEL = "openai_model"
_ALL_KEYS = (KEY_API_KEY, KEY_BASE_URL, KEY_MODEL)


@dataclass
class AICredentials:
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    source: str  # "database" 或 "env"


def get_ai_credentials(db: Session) -> AICredentials:
    settings = get_settings()
    rows = {row.key: row.value for row in db.scalars(select(AppSetting).where(AppSetting.key.in_(_ALL_KEYS))).all()}
    db_api_key = rows.get(KEY_API_KEY) or None
    return AICredentials(
        api_key=db_api_key or settings.openai_api_key,
        base_url=rows.get(KEY_BASE_URL) or settings.openai_base_url,
        model=rows.get(KEY_MODEL) or settings.openai_model,
        source="database" if db_api_key else "env",
    )


def set_ai_credentials(
    db: Session, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None,
) -> None:
    """只更新传入的字段(非None)，其余保持原样——前端"留空则不修改"的语义。"""
    for key, value in ((KEY_API_KEY, api_key), (KEY_BASE_URL, base_url), (KEY_MODEL, model)):
        if value is None:
            continue
        row = db.get(AppSetting, key)
        if row:
            row.value = value
        else:
            db.add(AppSetting(key=key, value=value))
    db.commit()


def clear_ai_credentials(db: Session) -> None:
    """清掉数据库里的覆盖配置，回退到.env——三个key一起清，不做部分清除。"""
    for key in _ALL_KEYS:
        row = db.get(AppSetting, key)
        if row:
            db.delete(row)
    db.commit()
