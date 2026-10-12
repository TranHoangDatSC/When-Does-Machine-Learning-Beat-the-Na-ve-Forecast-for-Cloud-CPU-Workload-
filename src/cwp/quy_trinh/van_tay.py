"""Vân tay của mã, cấu hình và dữ liệu, để biết khi nào được phép bỏ qua một tác vụ.

Bỏ qua một tác vụ chỉ vì tệp đầu ra đã tồn tại là cách chắc chắn để trộn kết quả của
hai lần chạy khác nhau. Vì vậy mỗi tác vụ ghi kèm một vân tay ba phần:

    ma      sha256 của toàn bộ src/cwp/**.py và scripts/*.py
    cau_hinh sha256 của config/*.yaml
    du_lieu  sha256 của tên, kích thước và thời điểm sửa của các tệp dữ liệu đầu vào

Lần chạy sau chỉ bỏ qua tác vụ khi cả ba phần trùng và tệp đầu ra đọc được. Khác một
phần nào là chạy lại, và kết quả cũ được chuyển sang thư mục `cu/` chứ không bị ghi
đè lặng lẽ.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def _bam_tep(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        if not p.exists():
            h.update(f"{p}|thieu".encode())
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.stat().st_size < 2_000_000 else
                 f"{p.stat().st_size}|{p.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _bam_meta(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        if p.exists():
            st = p.stat()
            h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
        else:
            h.update(f"{p.name}|thieu".encode())
    return h.hexdigest()[:16]


def van_tay_ma(root: Path) -> str:
    """Vân tay của mã nguồn: gói `cwp` và các script trong `scripts/`."""
    tep = list((root / "src" / "cwp").rglob("*.py"))
    tep += sorted((root / "scripts").glob("*.py"))
    return _bam_tep(tep)


def van_tay_cau_hinh(root: Path) -> str:
    """Vân tay của toàn bộ tệp cấu hình."""
    return _bam_tep(sorted((root / "config").glob("*.yaml")))


def van_tay_du_lieu(paths) -> str:
    """Vân tay của dữ liệu đầu vào, theo kích thước và thời điểm sửa."""
    return _bam_meta(paths)


def van_tay(root: Path, du_lieu=()) -> dict:
    """Ba phần vân tay, gói trong một dict để ghi vào meta của tác vụ."""
    return {
        "ma": van_tay_ma(root),
        "cau_hinh": van_tay_cau_hinh(root),
        "du_lieu": van_tay_du_lieu(du_lieu),
    }


def khop(a: dict, b: dict) -> bool:
    """Hai vân tay có khớp cả ba phần không."""
    return all(a.get(k) == b.get(k) for k in ("ma", "cau_hinh", "du_lieu"))


# --- Vân tay theo từng tác vụ (protocol v1.1) ------------------------------------
#
# `van_tay_ma` băm toàn bộ mã, nên sửa một dòng ở bất kỳ script nào cũng làm mọi tác vụ
# "lệch vân tay", kể cả tác vụ dựng dữ liệu không hề import dòng đó. Vân tay theo tác vụ chỉ
# băm những tệp mà tác vụ thực sự nạp: bao đóng import của tệp vào.


def _tim_mo_dun(root: Path, ten: str):
    """Đường dẫn tệp của mô-đun `ten` nếu nó thuộc `src/` hoặc `scripts/`, ngược lại None."""
    parts = ten.split(".")
    for goc in (root / "src", root / "scripts"):
        base = goc.joinpath(*parts)
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
    return None


def _ten_mo_dun_cua(root: Path, p: Path):
    for goc in (root / "src", root / "scripts"):
        try:
            rel = p.relative_to(goc).with_suffix("")
        except ValueError:
            continue
        parts = list(rel.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts), p.name == "__init__.py"
    return None, False


def bao_dong_import(root: Path, vao) -> list:
    """Mọi tệp `.py` thuộc dự án mà `vao` (tệp hoặc tên mô-đun `a.b`) nạp trực tiếp hay gián tiếp.

    Đọc cây cú pháp, không chạy mã. Gồm cả `__init__.py` của các gói cha. Bỏ qua thư viện ngoài.
    Import trong hàm cũng được tính (hơi rộng hơn thực tế, an toàn hơn là thiếu).
    """
    root = Path(root)
    if isinstance(vao, str) and not vao.endswith(".py"):
        p = _tim_mo_dun(root, vao)
    else:
        p = Path(vao)
        p = p if p.is_absolute() else root / p
    if p is None or not p.is_file():
        return []
    thay, cho = set(), [p.resolve()]
    while cho:
        f = cho.pop()
        if f in thay:
            continue
        thay.add(f)
        ten, la_goi = _ten_mo_dun_cua(root, f)
        try:
            cay = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        ung = set()
        for n in ast.walk(cay):
            if isinstance(n, ast.Import):
                ung.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    if ten is None:
                        continue
                    goc = ten.split(".") if la_goi else ten.split(".")[:-1]
                    goc = goc[: len(goc) - (n.level - 1)]
                    co = ".".join(goc + ([n.module] if n.module else []))
                else:
                    co = n.module or ""
                if co:
                    ung.add(co)
                    ung.update(f"{co}.{a.name}" for a in n.names)
        for m in ung:
            parts = m.split(".")
            for i in range(1, len(parts) + 1):    # gồm cả __init__ của các gói cha
                q = _tim_mo_dun(root, ".".join(parts[:i]))
                if q is not None and q.resolve() not in thay:
                    cho.append(q.resolve())
    return sorted(thay)


def tep_vao_cua_lenh(root: Path, lenh) -> list:
    """Tệp `.py` vào của một lệnh `[python, script.py, ...]` hoặc `[python, -m, mo.dun, ...]`."""
    lenh = [str(x) for x in lenh]
    if "-m" in lenh:
        i = lenh.index("-m")
        return bao_dong_import(root, lenh[i + 1])
    for x in lenh[1:]:
        if x.endswith(".py"):
            return bao_dong_import(root, x)
    return []


def van_tay_tac_vu(root: Path, lenh, du_lieu=()) -> dict:
    """Vân tay của một tác vụ: mã trong bao đóng import, toàn bộ cấu hình, dữ liệu vào."""
    root = Path(root)
    tep = tep_vao_cua_lenh(root, lenh)
    h = hashlib.sha256()
    for p in tep:    # đã sắp xếp
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return {
        "ma": h.hexdigest()[:16],
        "cau_hinh": van_tay_cau_hinh(root),
        "du_lieu": van_tay_du_lieu(du_lieu),
        "so_tep_ma": len(tep),
    }


def khop_tac_vu(a: dict, b: dict) -> bool:
    """Hai vân tay theo tác vụ có khớp `ma`, `cau_hinh` và `du_lieu` không."""
    return bool(a) and bool(b) and khop(a, b)


def doc_meta(p) -> dict:
    p = Path(p)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


__all__ = ["van_tay", "van_tay_ma", "van_tay_cau_hinh", "van_tay_du_lieu",
           "khop", "doc_meta", "bao_dong_import", "tep_vao_cua_lenh",
           "van_tay_tac_vu", "khop_tac_vu"]
