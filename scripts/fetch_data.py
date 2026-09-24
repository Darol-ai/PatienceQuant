"""下载并解压数据包（本地行情库、每日指标、旧模型文件），只用 Python 标准库。

数据约 1.2GB，放在 GitHub Release 附件里（没有 LFS 的流量限制）。用法：

    python scripts/fetch_data.py            # 下载、校验 sha256、解压到 backend/data/
    python scripts/fetch_data.py --force    # 已经解压过也重新下载解压

下载中断后再运行会从断点续传。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

TAG = "data-2026-09-23"
NAME = "patiencequant-data-2026-09-23.tar"
SHA256 = "983a37ebe8b0cd4a183b3d14ef8561963bace15edaa835695b6b45b904875479"
URL = f"https://github.com/Darol-ai/PatienceQuant/releases/download/{TAG}/{NAME}"

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".data-cache"
MARKER = ROOT / "backend" / "data" / f".{TAG}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    done = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(URL, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(request) as resp:
        if done and resp.status != 206:
            done = 0  # 服务器不支持续传，从头下载
        total = done + int(resp.headers.get("Content-Length", 0))
        with target.open("ab" if done else "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r下载 {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="已经解压过也重新下载解压")
    args = parser.parse_args()
    if MARKER.exists() and not args.force:
        print(f"数据包 {TAG} 已经解压过（{MARKER.relative_to(ROOT)}），跳过。需要重来请加 --force。")
        return 0

    archive = CACHE / NAME
    if not archive.exists() or sha256(archive) != SHA256:
        print(f"从 {URL} 下载……")
        download(archive)
    print("校验 sha256……")
    actual = sha256(archive)
    if actual != SHA256:
        print(f"校验失败：期望 {SHA256}，实际 {actual}。删除 {archive} 后重试。", file=sys.stderr)
        return 1

    print("解压到 backend/data/ ……")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:  # 只接受 backend/data/ 下的普通文件，防止路径穿越
            if not member.name.startswith("backend/data/") or ".." in Path(member.name).parts or not member.isfile():
                print(f"数据包里有不该有的路径：{member.name}", file=sys.stderr)
                return 1
        tar.extractall(ROOT, members=members)
    MARKER.write_text(SHA256 + "\n", encoding="utf-8")
    print(f"完成：{len(members)} 个文件。可以删除 {CACHE.relative_to(ROOT)}/ 释放空间。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
