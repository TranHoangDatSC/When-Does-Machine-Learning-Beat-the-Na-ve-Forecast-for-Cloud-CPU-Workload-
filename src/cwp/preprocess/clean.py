"""Làm sạch và khử giá trị bất thường (cắt CPU về [0, 100])."""

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


def clip_cpu(s: pd.Series, cfg: dict | None = None) -> tuple[pd.Series, int, int]:
    """Clip giá trị CPU% theo khoảng cấu hình (khoá `clip`).

    Parameters
    ----------
    s : pd.Series
        Chuỗi giá trị CPU%.
    cfg : dict, optional
        Cấu hình tiền xử lý (mặc định nạp từ config/preprocess.yaml).

    Returns
    -------
    tuple[pd.Series, int, int]
        - Chuỗi Series sau khi đã clip.
        - Số mẫu bị clip lên trên (s > max).
        - Số mẫu bị clip xuống dưới (s < min).
    """
    if cfg is None:
        cfg = _load_preprocess_config()

    clip_range = cfg.get("clip")
    if clip_range is None or len(clip_range) < 2:
        raise KeyError("Cấu hình thiếu hoặc sai định dạng cho khoá 'clip' (cần danh sách [min, max]).")

    lower, upper = clip_range[0], clip_range[1]

    n_high = int((s > upper).sum())
    n_low = int((s < lower).sum())
    s_clipped = s.clip(lower=lower, upper=upper)

    return s_clipped, n_high, n_low


def mask_sentinel(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Thay thế các giá trị sentinel bất thường thành NaN theo cấu hình (khoá `sentinel`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa các cột dữ liệu cần xử lý.
    cfg : dict, optional
        Cấu hình tiền xử lý (mặc định nạp từ config/preprocess.yaml).

    Returns
    -------
    pd.DataFrame
        Bản sao của DataFrame với các giá trị sentinel được chuyển thành NaN.
    """
    if cfg is None:
        cfg = _load_preprocess_config()

    sentinel_cfg = cfg.get("sentinel", {})
    if not sentinel_cfg:
        return df.copy()

    res = df.copy()
    for col, sentinels in sentinel_cfg.items():
        if col in res.columns:
            if not isinstance(sentinels, (list, tuple, set)):
                sentinels = [sentinels]
            res[col] = res[col].mask(res[col].isin(sentinels), np.nan)

    return res


def drop_unwanted_cols(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Bỏ các cột khai báo trong khoá `drop_cols` của cấu hình.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào.
    cfg : dict, optional
        Cấu hình tiền xử lý (mặc định nạp từ config/preprocess.yaml).

    Returns
    -------
    pd.DataFrame
        DataFrame sau khi đã bỏ các cột không cần thiết.
    """
    if cfg is None:
        cfg = _load_preprocess_config()

    drop_cols = cfg.get("drop_cols", [])
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df.copy()


__all__ = ["clip_cpu", "mask_sentinel", "drop_unwanted_cols"]
