"""Đọc và xử lý dữ liệu thô từ Bitbrains (E1 và E2)."""

from pathlib import Path
import pandas as pd
import yaml


def _load_config(cfg_path: str | Path | None = None) -> dict:
    """Tải cấu hình datasets từ file yaml."""
    if cfg_path is not None:
        path = Path(cfg_path)
    else:
        path = Path("config/datasets.yaml")
        if not path.exists():
            path = Path(__file__).resolve().parents[3] / "config" / "datasets.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_raw(path: str | Path, cfg: dict | None = None,
             env: str | None = None) -> pd.DataFrame:
    """Đọc tệp CSV Bitbrains và trả về 2 cột chuẩn: time_s và cpu_pct.

    Parameters
    ----------
    path : str hoặc Path
        Đường dẫn tới file CSV Bitbrains.
    cfg : dict, optional
        Cấu hình datasets (mặc định nạp từ config/datasets.yaml).

    Returns
    -------
    pd.DataFrame
        DataFrame chứa đúng 2 cột:
        - time_s: kiểu int64, giá trị timestamp tính bằng giây.
        - cpu_pct: kiểu float64, mức sử dụng CPU (%).
    """
    if cfg is None:
        cfg = _load_config()

    # Lấy cấu hình của đúng môi trường đang đọc. Bản cũ viết
    # `cfg.get("E1", {}) or cfg.get("E2", {})` nên luôn lấy E1 — chỉ chạy đúng nhờ
    # E1 và E2 tình cờ cùng sep/target, sẽ sai ngay khi hai bên khác nhau.
    if env:
        e_cfg = cfg.get(str(env).strip().upper(), {}) or {}
    else:
        e_cfg = cfg.get("E1", {}) or cfg.get("E2", {}) or {}
    sep = cfg.get("sep", e_cfg.get("sep", ";\t"))
    time_col = cfg.get("time_col", e_cfg.get("time_col", "Timestamp [ms]"))
    target_col = cfg.get("target", e_cfg.get("target", "CPU usage [%]"))

    df = pd.read_csv(path, sep=sep, engine="python")

    # Loại bỏ khoảng trắng và ký tự tab ở tên cột
    df.columns = [str(c).strip() for c in df.columns]
    time_col_clean = str(time_col).strip()
    target_col_clean = str(target_col).strip()

    if time_col_clean not in df.columns or target_col_clean not in df.columns:
        raise KeyError(
            f"Không tìm thấy cột '{time_col_clean}' hoặc '{target_col_clean}' trong {path}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    # QUAN TRỌNG: Giá trị cột Timestamp thực tế là GIÂY dù tên cột ghi ms.
    # Không nhân, không chia 1000.
    time_num = pd.to_numeric(df[time_col_clean], errors="coerce")
    cpu_num = pd.to_numeric(df[target_col_clean], errors="coerce")

    # Bỏ các dòng mà cpu_pct hoặc time_s không parse được thành số
    valid_mask = cpu_num.notna() & time_num.notna()

    if not valid_mask.any():
        return pd.DataFrame({
            "time_s": pd.Series(dtype="int64"),
            "cpu_pct": pd.Series(dtype="float64"),
        })

    return pd.DataFrame({
        "time_s": time_num[valid_mask].astype("int64").values,
        "cpu_pct": cpu_num[valid_mask].astype("float64").values,
    })


def make_series_id(env: str, path: str | Path, month: str | None = None) -> str:
    """Tạo mã định danh chuỗi (series_id).

    Parameters
    ----------
    env : str
        Môi trường ('E1' hoặc 'E2').
    path : str hoặc Path
        Đường dẫn file CSV hoặc tên VM.
    month : str, optional
        Tháng quan sát (bắt buộc đối với E2, ví dụ: '2013-8').

    Returns
    -------
    str
        Chuỗi series_id duy nhất (ví dụ: 'E1_137' hoặc 'E2_2013-8_1').
    """
    env_clean = str(env).strip()
    stem = Path(path).stem

    # Tên tệp của Rnd lặp lại giữa ba tháng, nên mọi môi trường lấy từ Rnd phải khoá
    # thêm phần tháng. Môi trường nào cần tháng thì truyền `month` vào; các môi
    # trường còn lại để trống.
    if month:
        return f"{env_clean}_{str(month).strip()}_{stem}"
    if env_clean.startswith("E2"):
        raise ValueError(
            f"Môi trường {env!r} lấy từ Bitbrains Rnd nên bắt buộc có thông tin tháng."
        )
    return f"{env_clean}_{stem}"


__all__ = ["load_raw", "make_series_id"]
