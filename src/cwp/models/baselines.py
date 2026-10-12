"""Ba mô hình cơ sở, mốc để so mọi mô hình học máy.

    naive           ŷ = y_t
    moving average  ŷ = trung bình y[t-5 .. t], cửa sổ 6, GỒM điểm hiện tại
    seasonal naive  ŷ = y_{t+h-288}   (đúng thời điểm cần dự báo, của ngày hôm trước)

**Không có yếu tố ngẫu nhiên nào**: không seed, không siêu tham số, không huấn luyện.
Mọi con số về sau đều so với các mô hình này, nên lệch một chữ số ở đây là **có lỗi**,
không phải nhiễu.

Hai mô hình đầu lấy thẳng từ ma trận đặc trưng
----------------------------------------------

`y_t` là cột `lag_0`. Moving average gồm cả điểm hiện tại nên dựng từ `roll_mean_6`,
vốn là trung bình của `y[t-6 .. t-1]`, bằng cách bỏ `y_{t-6}` và thêm `y_t`:

    ma6 = roll_mean_6 + (lag_0 − lag_6) / 6

Bản trước đây lấy thẳng `roll_mean_6`, tức bỏ mất điểm mới nhất, nên mốc so sánh yếu
hơn mức đáng có.

Lấy từ ma trận đặc trưng chứ không tính lại từ `data/processed/` là có chủ ý: nếu ma
trận đặc trưng sai thì dự báo naive cũng sai theo, và lỗi lộ ra ngay khi so với một
phép tính naive đọc thẳng từ `data/processed/`.

Mô hình thứ ba phải nối ngược về `data/processed/`
--------------------------------------------------

Dự báo phát tại `t` cho thời điểm `t+h`, nên giá trị cùng giờ của ngày hôm trước là
`y_{t+h-288}`, **không phải** `y_{t-288}`. Bản trước đây lấy `y_{t-288}`, tức lấy
đúng thời điểm phát dự báo thay vì thời điểm cần dự báo, và lệch đi `h` bước. Giá trị
đó không tính được từ `data/features/`: bộ đặc trưng sâu nhất chỉ tới `lag_24`. Nối
theo khoá `(series_id, bucket + h − 288)`.

Ba điều không làm:

- **Không** thêm `lag_288` vào bộ đặc trưng: không mô hình học máy nào dùng nó, chỉ
  mô hình cơ sở đó cần.
- **Không** siết luật dòng hợp lệ thành `t >= b0 + 288`: làm thế sẽ đổi số dòng của
  cả chín ma trận đặc trưng.
- Dòng không có `y_{t-288}` thì trả **NaN** và để phía gọi **báo tỉ lệ ra**. Không
  `ffill`, không lấp bằng `y_t`, không lặng lẽ bỏ.

Phần thiếu nằm gần như trọn trong train: `288 − h` bucket đầu của mỗi chuỗi rơi vào
khoảng 12% đầu cửa sổ. Tỉ lệ dòng test không có giá trị tương ứng được
`scripts/check_dac_trung_va_baseline.py` đo lại sau mỗi lần sinh đặc trưng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cùng giờ hôm trước trên lưới 5 phút: 24 * 12 = 288 bucket.
SEASONAL_LAG = 288

# Cùng giờ cùng thứ của tuần trước: 7 * 288 = 2016 bucket. Chỉ dùng được khi lịch sử
# đủ dài, tức ở Thí nghiệm B; cửa sổ 8 ngày không đủ một chu kỳ tuần.
SEASONAL_LAG_TUAN = 7 * SEASONAL_LAG

# Cửa sổ của moving average, trùng với đặc trưng `roll_mean_6`.
MA_WINDOW = 6

BASELINE_NAMES = ("naive", "ma6", "seasonal", "seasonal_tuan")


def y_hien_tai(X: pd.DataFrame) -> np.ndarray:
    """`y_t`, đọc thẳng cột `lag_0` của ma trận đặc trưng."""
    if "lag_0" not in X.columns:
        raise ValueError(
            "Ma trận đặc trưng thiếu cột lag_0. Sinh lại bằng scripts/build_features.py."
        )
    return X["lag_0"].to_numpy("float64")


def du_doan_naive(X: pd.DataFrame) -> np.ndarray:
    """`ŷ = y_t` (persistence)."""
    return y_hien_tai(X)


def du_doan_ma6(X: pd.DataFrame) -> np.ndarray:
    """`ŷ` = trung bình `y[t-5 .. t]`, gồm cả điểm hiện tại.

    Dựng từ `roll_mean_6` bằng cách bỏ `y_{t-6}` và thêm `y_t`, nên vẫn chỉ đọc ma
    trận đặc trưng và không cần tra lại `data/processed/`.
    """
    thieu = [c for c in ("roll_mean_6", "lag_0", "lag_6") if c not in X.columns]
    if thieu:
        raise ValueError(f"Ma trận đặc trưng thiếu cột {thieu}.")
    roll = X["roll_mean_6"].to_numpy("float64")
    moi = X["lag_0"].to_numpy("float64")
    cu = X["lag_6"].to_numpy("float64")
    return roll + (moi - cu) / MA_WINDOW


def du_doan_seasonal(
    X: pd.DataFrame, processed: pd.DataFrame, h: int, lag: int = SEASONAL_LAG
) -> np.ndarray:
    """`ŷ = y_{t+h-288}`, nối về `data/processed/` theo `(series_id, bucket + h − lag)`.

    Parameters
    ----------
    X : pd.DataFrame
        Ma trận đặc trưng, cần `series_id` và `bucket`.
    processed : pd.DataFrame
        Bảng dài `data/processed/{env}.parquet`, cần `series_id`, `bucket`, `y`.
    h : int
        Tầm dự báo tính bằng bucket. Dự báo phát tại `t` nhắm tới `t+h`, nên thời
        điểm tương ứng của ngày hôm trước là `t + h − lag`.
    lag : int, optional
        Mặc định 288 bucket = 24 giờ.

    Returns
    -------
    np.ndarray
        `NaN` ở dòng không có `y_{t+h-lag}`: hoặc vì thời điểm đó nằm trước đầu cửa
        sổ, hoặc vì chính điểm đó là `NaN` trong `data/processed/`. **Không lấp.**
    """
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")
    for c in ("series_id", "bucket"):
        if c not in X.columns:
            raise ValueError(f"Ma trận đặc trưng thiếu cột {c}.")
    for c in ("series_id", "bucket", "y"):
        if c not in processed.columns:
            raise ValueError(f"Bảng data/processed/ thiếu cột {c}.")

    tra_cuu = processed.set_index(["series_id", "bucket"])["y"]
    if not tra_cuu.index.is_unique:
        raise ValueError("data/processed/ có khoá (series_id, bucket) trùng lặp.")

    khoa = pd.MultiIndex.from_arrays(
        [X["series_id"].to_numpy(), X["bucket"].to_numpy("int64") + h - lag]
    )
    return tra_cuu.reindex(khoa).to_numpy("float64")


def du_doan(
    ten: str, X: pd.DataFrame, processed: pd.DataFrame | None = None,
    h: int | None = None,
) -> np.ndarray:
    """Gọi baseline theo tên. `seasonal` bắt buộc có `processed` và `h`."""
    if ten == "naive":
        return du_doan_naive(X)
    if ten == "ma6":
        return du_doan_ma6(X)
    if ten == "seasonal_tuan":
        if processed is None or h is None:
            raise ValueError(
                "seasonal naive theo tuần cần data/processed/ và h."
            )
        return du_doan_seasonal(X, processed, h, lag=SEASONAL_LAG_TUAN)
    if ten == "seasonal":
        if processed is None:
            raise ValueError(
                "seasonal naive cần data/processed/: bộ đặc trưng không có lag_288."
            )
        if h is None:
            raise ValueError(
                "seasonal naive cần h: dự báo nhắm tới t+h nên lấy y tại t+h−288."
            )
        return du_doan_seasonal(X, processed, h)
    raise ValueError(f"Baseline không rõ: {ten!r}. Có: {BASELINE_NAMES}.")
