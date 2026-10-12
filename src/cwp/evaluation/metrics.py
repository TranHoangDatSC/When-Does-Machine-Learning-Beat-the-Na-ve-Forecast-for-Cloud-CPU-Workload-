"""Năm chỉ số đánh giá, tính theo từng chuỗi rồi gộp bằng trung vị.

**MAE, RMSE, SMAPE, MASE, R². Không dùng MAPE** vì hơn 40% điểm của Bitbrains gần 0,
mẫu số tiến về 0 và MAPE nổ.

Cách gộp
--------

Tính chỉ số **riêng cho từng chuỗi trước**, rồi lấy trung vị và IQR trên *tập chuỗi*.
Không dồn mọi dòng của mọi chuỗi vào một dãy rồi tính một chỉ số.

Dồn mọi điểm vào một dãy làm ACF bậc 1 của E1 tăng từ 0,6674 lên 0,9586, vì phương sai
*giữa* các chuỗi bị trộn vào. Chỉ số lỗi cũng bị như vậy: chuỗi tải cao có sai số tuyệt
đối lớn hơn và sẽ chi phối con số chung. Trung vị theo chuỗi cho mỗi máy một phiếu
bằng nhau.

Ba trường hợp biên
------------------

- **SMAPE khi `|y| + |ŷ| = 0`**: số hạng đó bằng **0**, không phải NaN. Dự đoán đúng
  tuyệt đối tại một điểm bằng 0 thì sai số tương đối là 0. E1 và E2 có trung vị dưới
  2% nên quy ước này ảnh hưởng thật tới con số cuối.
- **MASE khi `d = 0`** (phần train phẳng hoàn toàn): **NaN**, chuỗi bị loại khỏi phần
  gộp và được đếm. Không thay bằng 0, không cộng epsilon, và **không trả `inf`**: một
  giá trị `inf` lọt vào trung vị là hỏng cả cột.
- **R² khi `SS_tot = 0`** (target hằng trong tập đang đánh giá): **NaN**, loại và đếm.

  Điều kiện này kiểm bằng `min == max`, **không** bằng `sum((y − ȳ)²) == 0`. Hai cách
  không tương đương trên số dấu phẩy động: chuỗi E1_830 có target hằng đúng
  `1,1333333333333333` ở cả 346 dòng test, nhưng `ȳ = sum/n` không rơi đúng vào giá
  trị đó nên tổng bình phương ra `1,7e-29` thay vì 0, và chuỗi lọt vào phần gộp với
  `R² = 1,0`. `min == max` nhận biết **chính xác** `SS_tot = 0`, không phụ thuộc thứ
  tự cộng dồn.

Đếm `n_loai`
------------

`n_chuoi + n_loai` = **tổng số chuỗi của môi trường**, không phải số chuỗi có mặt
trong tập đang đánh giá. Chuỗi `E3_m_2848` không có dòng hợp lệ nào trên test, nên nó
phải hiện ra ở `n_loai` chứ không được biến mất khỏi cả hai cột.

Mẫu số của MASE
---------------

`d` = trung bình `|y_t − y_{t−1}|` trên phần **train** của chính chuỗi đó, chỉ lấy cặp
mà **cả hai** đầu không NaN. Hai hệ quả:

- Tính **chỉ trên train**, cùng lý do `mu`, `sd` của chế độ N1 chỉ lấy trên train: không
  để thông tin của tập test lọt vào.
- **Không phụ thuộc horizon**: cùng một `d` dùng cho cả `h = 1, 6, 12`, nên MASE ở ba
  horizon so được với nhau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cwp.evaluation.splits import bucket_bounds, offset

METRIC_NAMES = ("mae", "rmse", "smape", "mase", "r2")

# Hai chỉ số có thể không xác định trên một chuỗi.
METRIC_CO_THE_LOAI = ("mase", "r2")


# ----------------------------------------------------------- chỉ số một chuỗi

def _sach(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    if y.shape != yhat.shape:
        raise ValueError(f"y và yhat lệch kích thước: {y.shape} vs {yhat.shape}.")
    giu = np.isfinite(y) & np.isfinite(yhat)
    return y[giu], yhat[giu]


def mae(y, yhat) -> float:
    """Sai số tuyệt đối trung bình. Chỉ số chính, cùng đơn vị với target."""
    y, yhat = _sach(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if y.size else float("nan")


def rmse(y, yhat) -> float:
    """Căn bậc hai của sai số bình phương trung bình. Luôn `>= mae` (bất đẳng thức Jensen)."""
    y, yhat = _sach(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if y.size else float("nan")


def smape(y, yhat) -> float:
    """`100 × mean(|y−ŷ| / ((|y|+|ŷ|)/2))`, số hạng `0/0` tính là **0**.

    Nằm trong `[0, 200]` theo định nghĩa.
    """
    y, yhat = _sach(y, yhat)
    if not y.size:
        return float("nan")
    mau = (np.abs(y) + np.abs(yhat)) / 2.0
    so_hang = np.where(mau == 0.0, 0.0, np.abs(y - yhat) / np.where(mau == 0.0, 1.0, mau))
    return float(100.0 * np.mean(so_hang))


def mase(y, yhat, d: float) -> float:
    """`MAE / d`, với `d` là mẫu số naive một bước trên **train** của chính chuỗi đó.

    `d = 0` hoặc `d` không hữu hạn thì trả **NaN** — không phải `inf`, xem đầu tệp.
    """
    if not np.isfinite(d) or d == 0.0:
        return float("nan")
    return mae(y, yhat) / float(d)


def r2(y, yhat) -> float:
    """`1 − SS_res/SS_tot`, `SS_tot` tính trên target của chuỗi đó **trong tập đang đánh giá**.

    `SS_tot = 0`, tức target **hằng**, thì trả **NaN**. Điều kiện hằng kiểm bằng
    `min == max`, xem phần đầu tệp về lý do.
    """
    y, yhat = _sach(y, yhat)
    if not y.size:
        return float("nan")
    if y.min() == y.max():
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot


# --------------------------------------------------------- mẫu số của MASE

def mase_denominator(y: np.ndarray) -> float:
    """`d` = trung bình `|y_t − y_{t−1}|`, chỉ lấy cặp mà **cả hai** đầu không NaN.

    Nhận đúng phần `y` thuộc **train** của một chuỗi, theo thứ tự bucket tăng dần và
    **liền mạch** trên lưới 5 phút (bước tiền xử lý bảo đảm điều đó).

    Trả NaN khi không còn cặp nào — chuỗi quá ngắn hoặc NaN xen kẽ từng điểm.
    """
    y = np.asarray(y, dtype="float64")
    if y.size < 2:
        return float("nan")
    sai = np.abs(np.diff(y))
    sai = sai[np.isfinite(sai)]  # cặp có một đầu NaN thì hiệu là NaN, bỏ đúng cặp đó
    return float(sai.mean()) if sai.size else float("nan")


def mase_denominators(
    df: pd.DataFrame, mask_train: np.ndarray | pd.Series,
    col_series: str = "series_id", col_y: str = "y",
) -> pd.Series:
    """Mẫu số MASE của mọi chuỗi, tính trên phần train của bảng dài `data/processed/`.

    Parameters
    ----------
    df : pd.DataFrame
        Bảng dài đã sắp theo `(series_id, bucket)`, có cột `series_id` và `y`.
    mask_train : array-like of bool
        Mặt nạ chọn đúng các bucket thuộc **train** (theo `splits.bucket_bounds()`).
        Mặt nạ chứ không phải `y` đã lọc, để hàm còn thấy được thứ tự gốc.

    Returns
    -------
    pd.Series
        Index là `series_id`, giá trị là `d`. Chuỗi có `d = 0` giữ nguyên giá trị 0 ở
        đây; việc loại chuỗi là của `gop()`, để chỗ đếm "bị loại" chỉ có một.
    """
    con = df.loc[np.asarray(mask_train, dtype=bool), [col_series, col_y]]
    return con.groupby(col_series, sort=True)[col_y].apply(
        lambda s: mase_denominator(s.to_numpy())
    ).rename("d")


def mase_denominators_train(processed: pd.DataFrame, b0: int) -> pd.Series:
    """Mẫu số MASE của mọi chuỗi, lấy đúng phần **train** của cửa sổ.

    Đây là nơi duy nhất quyết định `d` tính trên tập nào, và nó phải là **train**. Nếu
    tính trên test, MASE của naive ở `h = 1` sẽ lệch khỏi 1 mà không có dấu hiệu nào
    khác báo lỗi.

    Hàm không nhận `h`: cùng một `d` dùng cho cả `h = 1, 6, 12`.

    Parameters
    ----------
    processed : pd.DataFrame
        Bảng dài `data/processed/{env}.parquet`, đã sắp theo `(series_id, bucket)`.
    b0 : int
        Bucket đầu của môi trường, từ `splits.doc_b0()`.
    """
    lo, hi = bucket_bounds()["train"]
    off = offset(processed["bucket"], b0)
    return mase_denominators(processed, (off >= lo) & (off < hi))


# ------------------------------------------------------- chỉ số theo từng chuỗi

def per_series_metrics(
    series_id, y, yhat, d: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Năm chỉ số cho **từng chuỗi**, bước phải làm trước khi gộp.

    Dòng có `y` hoặc `yhat` không hữu hạn bị bỏ, và số dòng thực dùng của mỗi chuỗi
    ghi ở cột `n_dong`. Nhờ đó đọc được tỉ lệ dòng mà seasonal naive phải bỏ, thay vì
    lặng lẽ lấp giá trị.

    Parameters
    ----------
    series_id, y, yhat : array-like
        Cùng độ dài. Không cần sắp xếp trước.
    d : pd.Series | dict | None
        Mẫu số MASE theo `series_id`, từ `mase_denominators()`. Thiếu thì cột `mase`
        toàn NaN.

    Returns
    -------
    pd.DataFrame
        Một dòng mỗi chuỗi: `series_id, n_dong, mae, rmse, smape, mase, r2`.
    """
    df = pd.DataFrame({
        "series_id": np.asarray(series_id),
        "y": np.asarray(y, dtype="float64"),
        "yhat": np.asarray(yhat, dtype="float64"),
    })
    df = df[np.isfinite(df["y"]) & np.isfinite(df["yhat"])]

    sai = (df["y"] - df["yhat"]).abs()
    df["ae"] = sai
    df["se"] = sai ** 2
    mau = (df["y"].abs() + df["yhat"].abs()) / 2.0
    df["sm"] = np.where(mau == 0.0, 0.0, sai / mau.where(mau != 0.0, 1.0))

    g = df.groupby("series_id", sort=True)
    out = pd.DataFrame({
        "n_dong": g.size(),
        "mae": g["ae"].mean(),
        "rmse": np.sqrt(g["se"].mean()),
        "smape": 100.0 * g["sm"].mean(),
    })

    # R²: SS_tot trên target của chính chuỗi đó trong tập đang đánh giá. Chuỗi có
    # target hằng thì SS_tot = 0 — nhận biết bằng `min == max`, không bằng tổng bình
    # phương, xem phần đầu tệp.
    ss_res = g["se"].sum()
    ss_tot = g["y"].apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    hang = g["y"].min() == g["y"].max()
    out["r2"] = np.where(hang, np.nan, 1.0 - ss_res / ss_tot.where(~hang, 1.0))

    if d is None:
        out["mase"] = np.nan
    else:
        dd = pd.Series(d, dtype="float64").reindex(out.index)
        hop_le = np.isfinite(dd) & (dd != 0.0)
        out["mase"] = np.where(hop_le, out["mae"] / dd.where(hop_le, 1.0), np.nan)

    return out.reset_index()[["series_id", "n_dong", *METRIC_NAMES]]


# ------------------------------------------------------------------- gộp

def gop(per_series: pd.DataFrame, n_chuoi_tong: int | None = None,
        metrics=METRIC_NAMES) -> pd.DataFrame:
    """Trung vị và IQR **trên tập chuỗi**.

    Parameters
    ----------
    per_series : pd.DataFrame
        Đầu ra của `per_series_metrics()`.
    n_chuoi_tong : int, optional
        **Tổng số chuỗi của môi trường.** Mặc định lấy số dòng của `per_series`, chỉ
        đúng khi mọi chuỗi đều có ít nhất một dòng trong tập đang đánh giá. E3 thì
        không: `E3_m_2848` không có dòng hợp lệ nào trên test. Truyền tổng thật vào
        thì chuỗi ấy hiện ra ở `n_loai` thay vì biến mất khỏi cả hai cột.

    Returns
    -------
    pd.DataFrame
        Một dòng mỗi chỉ số: `metric, p25, p50, p75, iqr, n_chuoi, n_loai`, với
        `n_chuoi + n_loai = n_chuoi_tong`. `n_loai` gộp cả chuỗi vắng mặt lẫn chuỗi có
        chỉ số không xác định (`d = 0` với MASE, target hằng với R²).
    """
    dong = []
    tong = len(per_series) if n_chuoi_tong is None else int(n_chuoi_tong)
    if tong < len(per_series):
        raise ValueError(
            f"n_chuoi_tong = {tong} nhỏ hơn số chuỗi có mặt ({len(per_series)})."
        )
    for m in metrics:
        v = pd.to_numeric(per_series[m], errors="coerce")
        v = v[np.isfinite(v)]
        if len(v):
            p25, p50, p75 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        else:
            p25 = p50 = p75 = float("nan")
        dong.append({
            "metric": m, "p25": p25, "p50": p50, "p75": p75, "iqr": p75 - p25,
            "n_chuoi": int(len(v)), "n_loai": int(tong - len(v)),
        })
    return pd.DataFrame(dong)
