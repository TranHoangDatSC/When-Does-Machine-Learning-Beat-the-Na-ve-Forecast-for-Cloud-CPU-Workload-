"""Ráp ma trận đặc trưng và lọc dòng hợp lệ.

Đầu vào là `data/processed/{env}.parquet`, bảng dài với `env, series_id, bucket, y,
is_interp`. Đầu ra là đúng 22 cột cho một horizon `h`: `series_id`, `bucket`, 19 đặc
trưng, và `target` (`y` tại `t+h`).

Luật dòng hợp lệ tách riêng khỏi việc sinh đặc trưng, vì hai thứ **không** tương đương:

> 19 đặc trưng chỉ chạm 15 điểm trong cửa sổ (`t-24`, `t-12..t-1`, và `t`); các điểm
> `t-23` đến `t-13` không đặc trưng nào dùng. Nhưng luật đòi *toàn bộ* `[t-24, t]`
> không NaN. Nên `dropna()` trên ma trận đặc trưng **lỏng hơn** luật: thừa 1.513 dòng
> ở E2 h=1 và 7.461 dòng ở E3 h=1, còn E1 thì trùng nên không lộ ra.

Vì vậy `make_feature_matrix` lọc bằng `valid_row_mask`, tính trên `y` của bảng gốc,
chứ không bằng `dropna` trên bảng đặc trưng.

Nội suy và tính nhân quả
------------------------

`data/processed/` giữ cột `is_interp`: điểm được nội suy tuyến tính từ hai đầu mút của
một lỗ hổng ngắn. Nội suy **dùng đầu mút bên phải**, tức một giá trị đứng sau. Với một
điểm nội suy `p`, đầu mút phải là điểm quan sát thật đầu tiên sau `p`. Suy ra:

- Nếu `y(t)` là giá trị quan sát thật thì mọi điểm nội suy nằm trước `t` đều có đầu
  mút phải **không vượt quá** `t`, nên chúng tính được tại thời điểm phát dự báo.
- Nếu `y(t)` là giá trị nội suy thì đầu mút phải của nó nằm **sau** `t`, tức dòng đó
  dùng thông tin chưa có. Dòng như vậy bị loại.

Nhãn cũng vậy: `y(t+h)` là giá trị nội suy thì đó không phải quan sát thật, nên không
được dùng làm mục tiêu để huấn luyện hay để chấm điểm.

Hai điều kiện trên nằm trong `valid_row_mask` và bật khi bảng đầu vào có cột
`is_interp`. Bản đối chứng `K = 0` không nội suy gì, nên cột đó toàn `False` và luật
không cắt thêm dòng nào.
"""

import pandas as pd

from cwp.features.calendar import add_calendar_features
from cwp.features.spec import INPUT_COLS, MATRIX_COLS, MAX_LAG
from cwp.features.windows import (
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    add_target,
    assert_contiguous_grid,
)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm cột bắt buộc, sắp theo (series_id, bucket), đánh lại index chạy.

    Sắp xếp ở đây để kết quả không phụ thuộc thứ tự dòng của tệp đầu vào, và để mọi
    chuỗi thành một khối liền — điều kiện để `groupby().rolling()` chạy đúng.
    """
    thieu = [c for c in INPUT_COLS if c not in df.columns]
    if thieu:
        raise ValueError(
            f"Bảng đầu vào thiếu cột {thieu}. Cần {INPUT_COLS}."
        )
    out = df.sort_values(["series_id", "bucket"], kind="stable").reset_index(drop=True)
    assert_contiguous_grid(out)
    return out


def valid_row_mask(df: pd.DataFrame, h: int, max_lag: int = MAX_LAG) -> pd.Series:
    """Luật dòng huấn luyện hợp lệ.

    Dòng tại `t` với horizon `h` hợp lệ khi và chỉ khi mọi điểm trong `[t - max_lag,
    t]` đều không NaN, **và** target tại `t + h` không NaN. Cả hai điều kiện tính
    trong phạm vi **một chuỗi**.

    Khi bảng có cột `is_interp`, thêm hai điều kiện nhân quả: `y(t)` và `y(t+h)` phải
    là quan sát thật, xem phần đầu tệp.

    Dòng không hợp lệ thì bỏ dòng đó, không cắt chuỗi, không lấp thêm. Hệ quả: một
    điểm NaN đơn lẻ làm mất 25 dòng.

    Parameters
    ----------
    df : pd.DataFrame
        Bảng dài đã qua `prepare_input`.
    h : int
        Horizon, tính bằng số bucket.
    max_lag : int, optional
        Độ sâu cửa sổ đặc trưng, mặc định 24 (`config/preprocess.yaml: row_validity`).

    Returns
    -------
    pd.Series
        Mặt nạ bool cùng index với `df`.
    """
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")

    sid = df["series_id"]
    co_nan = df["y"].isna().astype("float64")

    # Cửa sổ [t - max_lag, t] gồm max_lag + 1 điểm, và phải sạch NaN hoàn toàn.
    # min_periods để trống nên 24 dòng đầu mỗi chuỗi ra NaN: thiếu lịch sử cũng là
    # không hợp lệ, như thiếu dữ liệu.
    trong_cua_so = co_nan.groupby(sid, sort=False).rolling(max_lag + 1).max()
    trong_cua_so = trong_cua_so.droplevel(0).reindex(df.index)

    cua_so_sach = trong_cua_so.eq(0.0)
    target = df.groupby("series_id", sort=False)["y"].shift(-h)
    hop_le = cua_so_sach & target.notna()

    if "is_interp" in df.columns:
        noi = df["is_interp"].fillna(False).astype(bool)
        # Làm việc trên số thực để `shift` không sinh kiểu object: dòng cuối chuỗi
        # không có nhãn nên coi là "không phải quan sát thật" (1.0).
        noi_f = noi.astype("float64")
        noi_tai_dich = noi_f.groupby(sid, sort=False).shift(-h).fillna(1.0) > 0.5
        hop_le = hop_le & (~noi) & (~noi_tai_dich)

    return hop_le


def build_feature_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Sinh 22 cột cho horizon `h`, **chưa lọc**: dòng thiếu lịch sử còn NaN.

    Dùng cho kiểm tra: các phép kiểm rò rỉ cần nhìn thấy đúng những dòng NaN mà
    `make_feature_matrix` sẽ bỏ đi.

    Parameters
    ----------
    df : pd.DataFrame
        Bảng dài đã qua `prepare_input`.
    h : int
        Horizon, tính bằng số bucket.

    Returns
    -------
    pd.DataFrame
        22 cột theo thứ tự `MATRIX_COLS`, cùng index với `df`.
    """
    out = pd.DataFrame(index=df.index)
    out["series_id"] = df["series_id"].to_numpy()
    out["bucket"] = df["bucket"].to_numpy()

    add_lag_features(df, out)
    add_rolling_features(df, out)
    add_diff_features(df, out)
    add_calendar_features(df, out)
    add_target(df, out, h)

    return out[MATRIX_COLS]


def make_feature_matrix(
    df: pd.DataFrame, h: int, max_lag: int = MAX_LAG
) -> pd.DataFrame:
    """Ma trận đặc trưng đã lọc theo luật dòng hợp lệ.

    Parameters
    ----------
    df : pd.DataFrame
        Bảng dài đọc từ `data/processed/{env}.parquet`. Không cần sắp trước.
    h : int
        Horizon, tính bằng số bucket (nghiên cứu dùng 1, 6, 12).
    max_lag : int, optional
        Độ sâu cửa sổ đặc trưng, mặc định 24.

    Returns
    -------
    pd.DataFrame
        22 cột, chỉ giữ dòng hợp lệ, index đánh lại từ 0.
    """
    chuan = prepare_input(df)
    khung = build_feature_frame(chuan, h)
    hop_le = valid_row_mask(chuan, h, max_lag=max_lag)
    return khung.loc[hop_le].reset_index(drop=True)
