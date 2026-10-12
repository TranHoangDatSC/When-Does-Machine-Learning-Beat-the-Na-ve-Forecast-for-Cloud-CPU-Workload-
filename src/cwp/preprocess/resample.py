"""Căn lưới thời gian, cắt cửa sổ và nội suy lỗ hổng ngắn."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def _load_preprocess_config(cfg_path: str | Path | None = None) -> dict:
    """Tải cấu hình tiền xử lý từ file yaml."""
    if cfg_path is not None:
        path = Path(cfg_path)
    else:
        path = Path("config/preprocess.yaml")
        if not path.exists():
            path = Path(__file__).resolve().parents[3] / "config" / "preprocess.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def to_grid(
    time_s: np.ndarray | pd.Series | list,
    cpu_pct: np.ndarray | pd.Series | list,
    grid: int | None = None,
    cfg: dict | None = None,
) -> pd.Series:
    """Căn mẫu về lưới thời gian cố định, lấy trung bình theo từng bucket.

    Parameters
    ----------
    time_s : array-like
        Mốc thời gian tính bằng giây.
    cpu_pct : array-like
        Mức sử dụng CPU (%).
    grid : int, optional
        Độ rộng bucket tính bằng giây (mặc định lấy grid_seconds từ config).
    cfg : dict, optional
        Cấu hình tiền xử lý.

    Returns
    -------
    pd.Series
        Chuỗi CPU% trung bình với index là số hiệu bucket floor(t / grid),
        liên tục từ bucket nhỏ nhất đến lớn nhất. Bucket rỗng nhận giá trị NaN.
    """
    if grid is None:
        if cfg is None:
            cfg = _load_preprocess_config()
        grid = cfg.get("grid_seconds", 300)

    time_arr = np.asarray(time_s)
    cpu_arr = np.asarray(cpu_pct, dtype=float)

    if len(time_arr) == 0:
        return pd.Series(dtype=float, index=pd.RangeIndex(0, 0))

    buckets = np.floor(time_arr / grid).astype(np.int64)
    s = pd.Series(cpu_arr, index=buckets)
    mean_s = s.groupby(s.index).mean()

    min_b = int(buckets.min())
    max_b = int(buckets.max())
    full_index = pd.RangeIndex(min_b, max_b + 1)

    return mean_s.reindex(full_index)


def apply_window(
    s: pd.Series,
    b0: int,
    n_buckets: int | None = None,
    cfg: dict | None = None,
) -> pd.Series:
    """Cắt chuỗi theo cửa sổ thời gian [b0, b0 + n_buckets).

    Parameters
    ----------
    s : pd.Series
        Chuỗi dữ liệu (index là số hiệu bucket).
    b0 : int
        Bucket bắt đầu của cửa sổ toàn cục.
    n_buckets : int, optional
        Số lượng bucket trong cửa sổ (mặc định window.days * 86400 / grid_seconds).
    cfg : dict, optional
        Cấu hình tiền xử lý.

    Returns
    -------
    pd.Series
        Chuỗi cắt theo các bucket nằm trong [b0, b0 + n_buckets).
        Trả về Series rỗng nếu chuỗi không có bucket nào trong cửa sổ.
    """
    if n_buckets is None:
        if cfg is None:
            cfg = _load_preprocess_config()
        days = cfg.get("window", {}).get("days", 8)
        grid_sec = cfg.get("grid_seconds", 300)
        n_buckets = int(days * 86400 // grid_sec)

    if s.empty:
        return s.iloc[0:0]

    mask = (s.index >= b0) & (s.index < b0 + n_buckets)
    out = s.loc[mask]
    if out.empty:
        return s.iloc[0:0]
    return out


def interpolate_short(
    s: pd.Series,
    k: int | None = None,
    cfg: dict | None = None,
) -> tuple[pd.Series, int]:
    """Nội suy tuyến tính các cụm NaN ngắn (độ dài <= k).

    Parameters
    ----------
    s : pd.Series
        Chuỗi đầu vào.
    k : int, optional
        Độ dài tối đa của cụm NaN được nội suy (mặc định gap.max_len từ config).
        Nếu k=0, không thực hiện nội suy.
    cfg : dict, optional
        Cấu hình tiền xử lý.

    Returns
    -------
    tuple[pd.Series, int]
        - Chuỗi Series sau khi nội suy.
        - Số điểm NaN đã được nội suy.
    """
    if k is None:
        if cfg is None:
            cfg = _load_preprocess_config()
        k = cfg.get("gap", {}).get("max_len", 2)

    if k <= 0 or s.empty:
        return s.copy(), 0

    arr = s.to_numpy(dtype=float, copy=True)
    n = len(arr)
    n_interp = 0
    i = 0
    while i < n:
        if np.isnan(arr[i]):
            start = i
            while i < n and np.isnan(arr[i]):
                i += 1
            end = i - 1
            gap_len = end - start + 1
            # Chỉ nội suy khi có điểm neo ở cả hai phía và cụm NaN dài <= k
            if start > 0 and end < n - 1 and gap_len <= k:
                left_val = arr[start - 1]
                right_val = arr[end + 1]
                steps = gap_len + 1
                for j in range(1, gap_len + 1):
                    arr[start + j - 1] = left_val + (right_val - left_val) * (j / steps)
                n_interp += gap_len
        else:
            i += 1

    return pd.Series(arr, index=s.index, name=s.name), n_interp


__all__ = ["to_grid", "apply_window", "interpolate_short"]
