"""Nạp dữ liệu, chuẩn hoá, dò siêu tham số, khớp và chấm — dùng chung cho mọi thí nghiệm.

Một "tổ hợp" là một lần huấn luyện và chấm: `(thí nghiệm, nguồn, đích, chế độ chuẩn
hoá, tầm dự báo, mô hình, hàm mất mát, seed, fold)`. Mọi thí nghiệm đều đi qua đúng
các bước dưới đây, nên chúng khác nhau ở phạm vi chứ không ở đường mã.

Bốn quy ước bắt buộc
--------------------

1. **Chỉ đọc ma trận N0.** Chế độ N1 dựng bằng phép affine từ chính ma trận đó, xem
   `cwp.features.io_ma_tran.bien_doi_n1_affine`. Nhờ vậy Thí nghiệm B đổi ranh giới
   phần train theo từng fold mà không phải sinh lại đặc trưng, và hai thí nghiệm dùng
   đúng một định nghĩa N1.

2. **`mu` và `sd` của N1 lấy trên phần train của chính chuỗi đó**, với ranh giới
   truyền từ ngoài vào. Chuỗi có `sd` không dương bị loại khỏi cả huấn luyện lẫn chấm
   ở chế độ N1, và số chuỗi bị loại được ghi ra.

3. **Mọi chỉ số tính trên thang CPU% gốc.** Dự đoán ở thang z được đưa về trước khi
   so. Giá trị thật luôn lấy từ cột `target` của ma trận N0.

4. **Điểm chọn siêu tham số cũng tính trên thang CPU%**, bằng đúng chỉ số chính của
   bài là trung vị MAE theo chuỗi. Chọn trên thang z sẽ ưu ái những chuỗi có `sd`
   nhỏ.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cwp.evaluation.luu_du_doan import ghi_du_doan
from cwp.evaluation.metrics import per_series_metrics
from cwp.evaluation.splits import (
    KHUNG_8N,
    Khung,
    doc_b0,
    fold_bounds,
    mask_khoang,
    offset,
)
from cwp.features import FEATURE_COLS, MATRIX_COLS
from cwp.features.io_ma_tran import bien_doi_n1_affine
from cwp.features.io_ma_tran import doc as doc_ma_tran
from cwp.models.baselines import BASELINE_NAMES, du_doan

# Hai mô hình cơ sở này tra thẳng `data/processed/`, tức đã ở thang CPU% gốc, nên
# **không** được đưa về thang lần nữa. Naive và trung bình trượt thì đọc từ ma trận
# đang ở thang của chế độ, nên vẫn phải đưa về.
BASELINE_DOC_CPU = ("seasonal", "seasonal_tuan")
from cwp.models.registry import REGISTRY

MODES = ("N0", "N1")


@dataclass
class KetQua:
    """Một dòng tóm tắt của một tổ hợp đã chạy."""

    thi_nghiem: str
    nguon: str
    dich: str
    mode: str
    h: int
    model: str
    loss: str
    seed: int
    fold: str
    n_train: int
    n_cham: int
    sieu_tham_so: str
    diem_val: float
    giay: float
    tep_du_doan: str
    ghi_chu: str = ""


# ------------------------------------------------------------- nạp dữ liệu

def thong_ke_n1(processed: pd.DataFrame, b0: int, train_end: int) -> pd.DataFrame:
    """`mu` và `sd` của phần train `[0, train_end)`, một dòng mỗi chuỗi.

    Bỏ qua `NaN` như `cwp.preprocess.normalize.thong_ke_train`, và dùng `ddof = 1`.
    """
    off = offset(processed["bucket"], b0)
    con = processed.loc[(off >= 0) & (off < int(train_end)), ["series_id", "y"]]
    g = con.groupby("series_id")["y"]
    out = pd.DataFrame({"mu": g.mean(), "sd": g.std(ddof=1), "n_diem": g.count()})
    return out


@dataclass
class DuLieu:
    """Ma trận đã sẵn sàng cho một `(env, h, mode)`."""

    env: str
    h: int
    mode: str
    X: pd.DataFrame           # ma trận ở thang của `mode`
    y_goc: np.ndarray         # target ở thang CPU% gốc
    sid: np.ndarray
    bucket: np.ndarray
    off: np.ndarray
    mu: np.ndarray            # theo dòng; NaN khi mode = N0
    sd: np.ndarray
    n_chuoi_tong: int
    processed: pd.DataFrame
    b0: int
    n_chuoi_loai_n1: int = 0
    chuoi_loai_n1: tuple = ()

    @property
    def A(self) -> np.ndarray:
        return self.X[FEATURE_COLS].to_numpy("float32")

    @property
    def y(self) -> np.ndarray:
        """Target ở thang của `mode`: dùng để huấn luyện."""
        return self.X["target"].to_numpy("float64")

    def ve_cpu(self, yhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Đưa dự đoán về thang CPU% gốc cho đúng các dòng của `mask`."""
        yhat = np.asarray(yhat, dtype="float64")
        if self.mode == "N0":
            return yhat
        return yhat * self.sd[mask] + self.mu[mask]


def chuan_bi(env: str, h: int, mode: str, *, feat_dir, proc_dir,
             train_end: int, X_goc=None, processed=None, dtype=None) -> DuLieu:
    """Nạp ma trận N0 của `(env, h)` rồi đưa về thang `mode`.

    Parameters
    ----------
    env, h, mode : str, int, str
        Môi trường, tầm dự báo, chế độ chuẩn hoá (`N0` hoặc `N1`).
    feat_dir, proc_dir : path-like
        Thư mục ma trận đặc trưng và thư mục bảng dài.
    train_end : int
        Ranh giới phần train tính bằng offset, dùng cho thống kê của N1.
    X_goc, processed : optional
        Bảng đã nạp sẵn, để Thí nghiệm B không đọc lại đĩa ở từng fold.
    dtype : optional
        Kiểu của 20 cột đặc trưng, ví dụ `"float32"` cho cửa sổ dài.
    """
    mode = str(mode).upper()
    if mode not in MODES:
        raise ValueError(f"Chế độ {mode!r} không thuộc {MODES}.")

    feat_dir, proc_dir = Path(feat_dir), Path(proc_dir)
    X = doc_ma_tran(feat_dir / f"{env}_h{h}.parquet") if X_goc is None else X_goc
    if dtype is not None:
        X = X.astype({c: dtype for c in FEATURE_COLS})
    if processed is None:
        processed = pd.read_parquet(proc_dir / f"{env}.parquet",
                                    columns=["series_id", "bucket", "y"])
    b0 = int(processed["bucket"].min())

    sid = X["series_id"].to_numpy()
    bucket = X["bucket"].to_numpy("int64")
    off = offset(bucket, b0)
    y_goc = X["target"].to_numpy("float64")
    n_chuoi_tong = int(processed["series_id"].nunique())

    if mode == "N0":
        return DuLieu(env=env, h=h, mode=mode, X=X, y_goc=y_goc, sid=sid,
                      bucket=bucket, off=off,
                      mu=np.full(len(X), np.nan), sd=np.full(len(X), np.nan),
                      n_chuoi_tong=n_chuoi_tong, processed=processed, b0=b0)

    tk = thong_ke_n1(processed, b0, train_end)
    xau = ~np.isfinite(tk["sd"]) | (tk["sd"] <= 0)
    chuoi_loai = tuple(tk.index[xau])
    Xz = bien_doi_n1_affine(X, tk["mu"], tk["sd"])
    giu = ~np.isin(sid, chuoi_loai)
    if not giu.all():
        Xz = Xz.loc[giu].reset_index(drop=True)
        y_goc = y_goc[giu]
        sid = sid[giu]
        bucket = bucket[giu]
        off = off[giu]
    mu = tk["mu"].reindex(sid).to_numpy("float64")
    sd = tk["sd"].reindex(sid).to_numpy("float64")

    return DuLieu(env=env, h=h, mode=mode, X=Xz, y_goc=y_goc, sid=sid,
                  bucket=bucket, off=off, mu=mu, sd=sd,
                  n_chuoi_tong=n_chuoi_tong, processed=processed, b0=b0,
                  n_chuoi_loai_n1=int(xau.sum()), chuoi_loai_n1=chuoi_loai)


# ------------------------------------------------------- chọn siêu tham số

def diem_val(d: DuLieu, mask: np.ndarray, yhat_thang_mode: np.ndarray) -> float:
    """Trung vị MAE theo chuỗi, trên thang CPU% gốc."""
    yhat = d.ve_cpu(yhat_thang_mode, mask)
    ps = per_series_metrics(d.sid[mask], d.y_goc[mask], yhat)
    v = ps["mae"].to_numpy("float64")
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def chon_sieu_tham_so(spec, d: DuLieu, khung: Khung, *, mau_con_ix=None,
                      log: "list | None" = None) -> tuple:
    """Rolling-origin trong vùng validation của `khung`. Không chạm test.

    Trả `(ứng viên tốt nhất, điểm trung bình qua các fold)`.
    """
    A, y = d.A, d.y
    if len(spec.luoi) == 1:
        return spec.luoi[0], float("nan")

    diem = []
    for i_c, cand in enumerate(spec.luoi):
        theo_fold = []
        for i_f, bien in enumerate(fold_bounds(khung=khung)):
            tr = mask_khoang(d.off, d.h, *bien["train"])
            va = mask_khoang(d.off, d.h, *bien["val"])
            if mau_con_ix is not None:
                tr = tr & mau_con_ix
            if not tr.any() or not va.any():
                continue
            t0 = time.time()
            mo = spec.dung(cand).fit(A[tr], y[tr])
            diem_f = diem_val(d, va, mo.predict(A[va]))
            theo_fold.append(diem_f)
            if log is not None:
                log.append({
                    "env": d.env, "h": d.h, "mode": d.mode, "model": spec.ten,
                    "ung_vien": json.dumps(cand, sort_keys=True), "fold": i_f,
                    "n_train": int(tr.sum()), "n_val": int(va.sum()),
                    "mae_p50_val": diem_f, "giay": round(time.time() - t0, 1),
                })
        diem.append(float(np.mean(theo_fold)) if theo_fold else float("inf"))
    tot = int(np.argmin(diem))
    return spec.luoi[tot], diem[tot]


# --------------------------------------------------------- khớp và chấm

def khop_va_cham(model: str, d: DuLieu, *, mask_fit: np.ndarray,
                 mask_cham: np.ndarray, cand: dict, seed: int,
                 thu_muc_du_doan, thi_nghiem: str, fold: str = "test",
                 dich: "DuLieu | None" = None, mau_con_ix=None) -> KetQua:
    """Khớp một mô hình rồi chấm, ghi dự đoán từng dòng và trả dòng tóm tắt.

    `dich` khác `None` nghĩa là chuyển giao: khớp trên `d`, chấm trên `dich`. Khi đó
    `mask_cham` là mặt nạ của `dich`.
    """
    t0 = time.time()
    d_cham = d if dich is None else dich

    if model in BASELINE_NAMES:
        tho = du_doan(model, d_cham.X, d_cham.processed, h=d_cham.h)[mask_cham]
        yhat = (tho if model in BASELINE_DOC_CPU
                else d_cham.ve_cpu(tho, mask_cham))
        n_train, loss, cand_s = 0, "khong", "{}"
    else:
        spec = REGISTRY[model]
        m_fit = mask_fit if mau_con_ix is None else (mask_fit & mau_con_ix)
        n_train = int(m_fit.sum())
        mo = spec.dung(cand).fit(d.A[m_fit], d.y[m_fit])
        yhat_mode = mo.predict(d_cham.A[mask_cham])
        yhat = d_cham.ve_cpu(yhat_mode, mask_cham)
        loss, cand_s = spec.loss, json.dumps(cand, sort_keys=True)

    p = ghi_du_doan(
        thu_muc_du_doan,
        series_id=d_cham.sid[mask_cham], bucket_phat=d_cham.bucket[mask_cham],
        y=d_cham.y_goc[mask_cham], yhat=yhat, h=d_cham.h, model=model,
        loss=loss, nguon=d.env, dich=d_cham.env, mode=d.mode, lich="co",
        fold=fold, seed=seed,
    )
    return KetQua(
        thi_nghiem=thi_nghiem, nguon=d.env, dich=d_cham.env, mode=d.mode,
        h=d_cham.h, model=model, loss=loss, seed=seed, fold=fold,
        n_train=n_train, n_cham=int(mask_cham.sum()), sieu_tham_so=cand_s,
        diem_val=float("nan"), giay=round(time.time() - t0, 1),
        tep_du_doan=Path(p).name,
    )


__all__ = ["MODES", "DuLieu", "KetQua", "chuan_bi", "chon_sieu_tham_so",
           "khop_va_cham", "thong_ke_n1", "diem_val"]
