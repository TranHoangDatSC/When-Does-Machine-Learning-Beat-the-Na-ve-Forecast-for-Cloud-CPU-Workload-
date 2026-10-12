"""Đặc trưng cửa sổ quá khứ: lag, rolling, sai phân.

Bốn ràng buộc:

1. Rolling tính **chỉ trên quá khứ**, không gồm điểm hiện tại: chuỗi được dịch một
   bước (`shift(1)`) *trước khi* lăn cửa sổ, nên giá trị tại `t` lấy trên đúng
   `y[t-w] … y[t-1]`.
2. **Không** `min_periods=1`. Cửa sổ thiếu điểm hoặc dính NaN đều ra NaN, đúng mặc
   định của pandas khi để trống `min_periods`.
3. Mọi phép dịch và lăn cửa sổ đều đi qua `groupby("series_id")`. Một tệp
   `data/processed/` chứa hàng trăm chuỗi nối nhau; thiếu bước này thì đuôi chuỗi
   trước chảy vào đầu chuỗi sau.
4. Không đặc trưng nào chạm `y` tại `t+1` trở đi. Chỉ có `target` được phép, và nó
   không phải đặc trưng.
"""

import pandas as pd

from cwp.features.spec import LAGS, ROLL_DDOF, ROLL_STATS, ROLL_WINDOWS


def assert_contiguous_grid(df: pd.DataFrame) -> None:
    """Kiểm mỗi chuỗi là một dải bucket liên tục, bước đúng 1, đã sắp tăng dần.

    Lag và rolling ở đây tính theo **vị trí dòng**. Điều đó chỉ trùng với khoảng
    cách **thời gian** khi lưới không thủng. Bảng `data/processed/` thoả điều kiện
    này: `to_grid` sinh lưới liên tục và điểm thiếu vẫn giữ dòng với `y = NaN`. Lưới
    thủng nghĩa là bảng đầu vào đã bị lọc NaN từ trước; nếu đi tiếp, `lag_1` có thể
    thành khoảng cách 2 giờ thật mà không báo lỗi gì.

    Raises
    ------
    ValueError
        Nếu có chuỗi mà bucket không liên tục hoặc không tăng dần.
    """
    d = df.groupby("series_id", sort=False)["bucket"].diff()
    xau = d.notna() & (d != 1)
    if bool(xau.any()):
        sid = df.loc[xau, "series_id"].iloc[0]
        buoc = int(d[xau].iloc[0])
        raise ValueError(
            f"Lưới thời gian thủng: chuỗi {sid!r} có bước bucket {buoc} thay vì 1. "
            "Đầu vào phải là data/processed/ nguyên vẹn, chưa lọc NaN."
        )


def _past(df: pd.DataFrame) -> pd.Series:
    """Chuỗi `y` dịch một bước theo từng series_id, tức `y_{t-1}`."""
    return df.groupby("series_id", sort=False)["y"].shift(1)


def _flatten(s: pd.Series, index: pd.Index) -> pd.Series:
    """Bỏ mức khoá nhóm mà `groupby().rolling()` thêm vào, trả về theo index gốc."""
    return s.droplevel(0).reindex(index)


def add_lag_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Thêm `lag_0, lag_1, lag_2, lag_3, lag_6, lag_12, lag_24` vào `out`.

    `lag_0` là `y` tại chính `t`, tức giá trị đã biết tại thời điểm phát dự báo, và
    cũng là giá trị mà dự báo naive dùng. Nó không chạm `y` ở `t+1` trở đi nên không
    phải rò rỉ.

    Parameters
    ----------
    df : pd.DataFrame
        Bảng dài có `series_id` và `y`, đã sắp theo (series_id, bucket).
    out : pd.DataFrame
        Bảng đích, cùng index với `df`. Bị sửa tại chỗ.

    Returns
    -------
    pd.DataFrame
        Chính `out`, để nối chuỗi lời gọi.
    """
    g = df.groupby("series_id", sort=False)["y"]
    for k in LAGS:
        out[f"lag_{k}"] = g.shift(k)
    return out


def add_rolling_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Thêm 8 thống kê rolling của hai cửa sổ 6 và 12 vào `out`.

    Cửa sổ tại `t` là `[t-w, t-1]`: dịch trước rồi mới lăn, nên điểm hiện tại nằm
    ngoài. `roll_std_*` dùng `ddof = 1`.
    """
    past = _past(df)
    sid = df["series_id"]
    for w in ROLL_WINDOWS:
        # min_periods để trống = w: cửa sổ chưa đủ w điểm không NaN thì ra NaN.
        r = past.groupby(sid, sort=False).rolling(w)
        ket_qua = {
            "mean": r.mean(),
            "std": r.std(ddof=ROLL_DDOF),
            "min": r.min(),
            "max": r.max(),
        }
        for stat in ROLL_STATS:
            out[f"roll_{stat}_{w}"] = _flatten(ket_qua[stat], df.index)
    return out


def add_diff_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Thêm `diff_1` = `y_t` trừ `y_{t-1}` vào `out`.

    `y_t` đã biết tại thời điểm dự đoán `t`, nên dùng nó trong sai phân không phải
    rò rỉ.
    """
    out["diff_1"] = df["y"] - _past(df)
    return out


def add_target(df: pd.DataFrame, out: pd.DataFrame, h: int) -> pd.DataFrame:
    """Thêm `target` = `y` tại `t + h`, lấy trong **cùng chuỗi**."""
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")
    out["target"] = df.groupby("series_id", sort=False)["y"].shift(-h)
    return out
