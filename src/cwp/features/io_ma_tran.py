"""Đọc và ghi ma trận đặc trưng kèm kiểm phiên bản lược đồ.

Một bản ma trận đặc trưng cũ có 22 cột và **không** có `lag_0`. Bản dùng ở đây có 23 cột.
Hai bản nằm ở hai thư mục khác nhau, nhưng tên tệp giống hệt nhau, nên đọc nhầm là
chuyện dễ xảy ra và **không báo lỗi**: mô hình vẫn khớp được, chỉ là thiếu một đặc
trưng và mọi con số lệch đi.

Vì vậy mỗi tệp mang một nhãn lược đồ trong metadata của parquet, và `doc()` dừng
ngay khi nhãn sai hoặc thiếu. Tệp không có nhãn, hoặc sinh bởi phiên bản mã cũ, luôn bị chặn.

    ghi(X, duong_dan)          ghi kèm nhãn
    doc(duong_dan)             đọc và kiểm nhãn, kiểm đủ 23 cột đúng thứ tự
    doc(duong_dan, cot=[...])  đọc một phần cột, vẫn kiểm nhãn

`bien_doi_n1_affine` đổi một ma trận N0 sang thang z-score mà không cần sinh lại đặc
trưng. Phép z-score là affine nên biến đổi này **chính xác**, không phải xấp xỉ:

    lag_k, roll_mean, roll_min, roll_max, target   (x − mu) / sd
    roll_std, diff_1                               x / sd
    bốn đặc trưng lịch                             giữ nguyên

Thí nghiệm B đổi thống kê chuẩn hoá theo từng fold, nên sinh lại đặc trưng cho mỗi
fold sẽ tốn gấp nhiều lần; đường affine cho đúng cùng kết quả. Thí nghiệm A vẫn dùng
ma trận sinh lại, và `scripts/check_dac_trung_va_baseline.py` đối chiếu hai đường.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cwp.features.spec import (
    CALENDAR_COLS,
    DIFF_COLS,
    LAG_COLS,
    MATRIX_COLS,
    ROLL_COLS,
    TARGET_COL,
)

# Nhãn phiên bản ghi vào metadata của parquet.
KHOA_PHIEN_BAN = b"cwp_schema"
PHIEN_BAN = b"v5-23cot-lag0"

# Cột dịch chuyển theo mu và co theo sd.
COT_AFFINE = LAG_COLS + [c for c in ROLL_COLS if not c.startswith("roll_std")] + [TARGET_COL]
# Cột chỉ co theo sd.
COT_CO = [c for c in ROLL_COLS if c.startswith("roll_std")] + DIFF_COLS


def ghi(X: pd.DataFrame, duong_dan) -> Path:
    """Ghi ma trận đặc trưng kèm nhãn phiên bản."""
    if list(X.columns) != MATRIX_COLS:
        raise ValueError(
            f"Lược đồ lệch khỏi MATRIX_COLS ({len(MATRIX_COLS)} cột). "
            f"Nhận {len(X.columns)} cột."
        )
    p = Path(duong_dan)
    p.parent.mkdir(parents=True, exist_ok=True)
    bang = pa.Table.from_pandas(X, preserve_index=False)
    meta = dict(bang.schema.metadata or {})
    meta[KHOA_PHIEN_BAN] = PHIEN_BAN
    pq.write_table(bang.replace_schema_metadata(meta), p)
    return p


def doc(duong_dan, cot: "list | None" = None) -> pd.DataFrame:
    """Đọc ma trận đặc trưng, dừng ngay nếu nhãn phiên bản sai."""
    p = Path(duong_dan)
    if not p.exists():
        raise FileNotFoundError(
            f"Thiếu {p}. Sinh lại bằng scripts/build_features.py."
        )
    meta = pq.read_schema(p).metadata or {}
    nhan = meta.get(KHOA_PHIEN_BAN)
    if nhan != PHIEN_BAN:
        raise ValueError(
            f"{p} mang nhãn lược đồ {nhan!r}, cần {PHIEN_BAN!r}. "
            "Tệp sinh bởi phiên bản mã cũ có thể thiếu cột lag_0. "
            "Sinh lại bằng scripts/build_features.py."
        )
    X = pd.read_parquet(p, columns=cot)
    if cot is None and list(X.columns) != MATRIX_COLS:
        raise ValueError(f"{p}: thứ tự cột lệch khỏi MATRIX_COLS.")
    return X


def co_nhan(duong_dan) -> bool:
    """Tệp có đúng nhãn lược đồ hay không; dùng cho bước kiểm tra đầu vào."""
    p = Path(duong_dan)
    if not p.exists():
        return False
    try:
        meta = pq.read_schema(p).metadata or {}
    except Exception:
        return False
    return meta.get(KHOA_PHIEN_BAN) == PHIEN_BAN


def bien_doi_n1_affine(X: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """Đổi ma trận N0 sang thang z-score theo từng chuỗi, bằng phép affine.

    Parameters
    ----------
    X : pd.DataFrame
        Ma trận N0, đủ `MATRIX_COLS`.
    mu, sd : pd.Series
        Trung bình và độ lệch chuẩn của **phần huấn luyện** của từng chuỗi, index là
        `series_id`. Chuỗi có `sd` không dương hoặc không xác định sẽ cho toàn NaN và
        bị luật dòng hợp lệ loại ở bước sau.

    Returns
    -------
    pd.DataFrame
        Ma trận cùng lược đồ, giá trị ở thang z.
    """
    thieu = [c for c in MATRIX_COLS if c not in X.columns]
    if thieu:
        raise ValueError(f"Thiếu cột {thieu}.")
    sid = X["series_id"].to_numpy()
    m = pd.Series(mu, dtype="float64").reindex(sid).to_numpy()
    s = pd.Series(sd, dtype="float64").reindex(sid).to_numpy()
    xau = ~np.isfinite(s) | (s <= 0)
    s = np.where(xau, np.nan, s)

    out = X.copy()
    for c in COT_AFFINE:
        out[c] = (X[c].to_numpy("float64") - m) / s
    for c in COT_CO:
        out[c] = X[c].to_numpy("float64") / s
    for c in CALENDAR_COLS:
        out[c] = X[c]
    return out


__all__ = ["KHOA_PHIEN_BAN", "PHIEN_BAN", "ghi", "doc", "co_nhan",
           "bien_doi_n1_affine", "COT_AFFINE", "COT_CO"]
