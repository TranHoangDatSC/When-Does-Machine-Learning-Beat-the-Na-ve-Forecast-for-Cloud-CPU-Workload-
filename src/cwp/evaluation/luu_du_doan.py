"""Lưu dự đoán của từng dòng, kèm đủ thông tin để tính lại mọi chỉ số về sau.

Bảng chỉ số theo từng chuỗi đủ cho các phép so đã định trước, nhưng không đủ cho một
chỉ số nghĩ ra sau khi chạy: muốn thêm một hàm sai số mới thì phải huấn luyện lại toàn
bộ. Vì vậy mỗi lần chấm test đều ghi ra một tệp parquet gồm giá trị thật và giá trị dự
đoán của từng dòng.

Mỗi lệnh gọi ghi **một tệp**, tên suy từ chính các khoá của lần chấm đó. Thư mục chứa
nhiều tệp đọc được như một bảng:

    pd.read_parquet("runs/<timestamp>/du_doan")

Mười ba cột
-----------

    nguon, dich       môi trường huấn luyện và môi trường được chấm
    mode              N0, N1 hoặc N2
    lich              co hoặc khong, tức có dùng bốn đặc trưng lịch hay không
    model, loss       tên mô hình và hàm mất mát đã huấn luyện
    h                 tầm dự báo, tính bằng bucket
    fold              test, hoặc tên fold của rolling-origin
    seed              hạt giống của lần chạy
    series_id         mã máy
    bucket_phat       thời điểm phát dự báo t
    bucket_muc_tieu   thời điểm cần dự báo t + h
    y, yhat           giá trị thật và giá trị dự đoán, trên thang CPU%

`bucket * 300` là số giây. Với E1 và E2 đó là epoch UTC thật; với E3 là giây kể từ
đầu trace, nên hai cột thời điểm của E3 chỉ so sánh được trong nội bộ E3. Quy ước đó
ghi trong `meta.json` của lần chạy, không lặp lại ở từng dòng.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COT = [
    "nguon", "dich", "mode", "lich", "model", "loss", "h", "fold", "seed",
    "series_id", "bucket_phat", "bucket_muc_tieu", "y", "yhat",
]


def ten_tep(nguon: str, dich: str, mode: str, lich: str, model: str,
            loss: str, h: int, fold: str) -> str:
    """Tên tệp suy từ khoá của lần chấm, để hai lần chấm khác nhau không đè nhau."""
    return f"{nguon}-{dich}_{mode}_{lich}_{model}_{loss}_h{h}_{fold}.parquet"


def ghi_du_doan(
    thu_muc: str | Path,
    *,
    series_id,
    bucket_phat,
    y,
    yhat,
    h: int,
    model: str,
    loss: str,
    nguon: str,
    dich: str | None = None,
    mode: str = "N0",
    lich: str = "co",
    fold: str = "test",
    seed: int = 42,
) -> Path:
    """Ghi một tệp parquet cho một lần chấm và trả về đường dẫn tệp.

    Parameters
    ----------
    thu_muc : str | Path
        Thư mục chứa, thường là `runs/<timestamp>/du_doan`.
    series_id, bucket_phat, y, yhat : array-like
        Bốn mảng cùng độ dài, đúng các dòng đã chấm.
    h : int
        Tầm dự báo tính bằng bucket.
    model, loss, nguon : str
        Tên mô hình, nhãn hàm mất mát, môi trường huấn luyện.
    dich : str, optional
        Môi trường được chấm. Bỏ trống thì lấy bằng `nguon`.
    mode, lich, fold, seed
        Các khoá còn lại của lần chấm.

    Returns
    -------
    Path
        Đường dẫn tệp vừa ghi.
    """
    dich = nguon if dich is None else dich
    sid = np.asarray(series_id)
    bp = np.asarray(bucket_phat, dtype="int64")
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    n = len(sid)
    if not (len(bp) == len(y) == len(yhat) == n):
        raise ValueError(
            f"Bốn mảng phải cùng độ dài: {n}, {len(bp)}, {len(y)}, {len(yhat)}."
        )
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")

    d = pd.DataFrame({
        "nguon": nguon, "dich": dich, "mode": mode, "lich": lich,
        "model": model, "loss": loss, "h": int(h), "fold": fold, "seed": int(seed),
        "series_id": sid, "bucket_phat": bp, "bucket_muc_tieu": bp + int(h),
        "y": y, "yhat": yhat,
    })[COT]

    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    p = thu_muc / ten_tep(nguon, dich, mode, lich, model, loss, h, fold)
    d.to_parquet(p, index=False)
    return p


def doc_du_doan(thu_muc: str | Path) -> pd.DataFrame:
    """Đọc cả thư mục dự đoán thành một bảng."""
    thu_muc = Path(thu_muc)
    if not thu_muc.exists():
        raise FileNotFoundError(f"Thiếu {thu_muc}.")
    return pd.read_parquet(thu_muc)


__all__ = ["COT", "ten_tep", "ghi_du_doan", "doc_du_doan"]
