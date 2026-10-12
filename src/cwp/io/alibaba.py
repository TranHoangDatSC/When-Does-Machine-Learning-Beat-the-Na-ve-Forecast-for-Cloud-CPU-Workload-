"""Đọc dữ liệu Alibaba Cluster Trace (E3).

Đọc tệp 9 GB theo khối (chunk), chỉ giữ các máy nằm trong danh sách cố định.

**Không có hàm lấy mẫu ở đây, và đó là cố ý.** Danh sách 500 máy của E3 được cố
định ở `config/e3_machines.txt`, vì lấy mẫu tại chỗ cho kết quả phụ thuộc thứ tự
quét tệp: hai cách viết đều đúng về logic vẫn có thể ra hai tập máy khác nhau tới 89%.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

DEFAULT_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
]


def _load_e3_config(cfg_path: str | Path | None = None) -> dict:
    """Tải cấu hình cho môi trường E3 từ config/datasets.yaml."""
    if cfg_path is not None:
        p = Path(cfg_path)
    else:
        p = Path("config/datasets.yaml")
        if not p.exists():
            p = Path(__file__).resolve().parents[3] / "config" / "datasets.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg.get("E3", {})
    return {}


def _estimate_total_chunks(path: str | Path, chunksize: int) -> int | None:
    """Ước lượng số chunk dựa trên kích thước file để hiển thị tqdm thanh tiến trình."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        file_size = p.stat().st_size
        # machine_usage.csv có 246,934,820 dòng trên ~9.657 GB (~39.1 bytes/dòng)
        est_rows = file_size / 39.1
        return max(1, int(round(est_rows / chunksize)))
    except Exception:
        return None


def machine_means(
    path: str | Path,
    chunksize: int = 2_000_000,
    cfg: dict | None = None,
) -> pd.Series:
    """Tính CPU trung bình của từng máy. Một lượt quét toàn bộ tệp.

    **Không dùng hàm này để chọn mẫu cho E3.** Tập máy của E3 đã cố định ở
    `config/e3_machines.txt`, đọc bằng `load_machines_frozen()`. Hàm này cho CPU
    trung bình của toàn bộ 4.023 máy, là đầu vào của phép lấy mẫu phân tầng đã dùng
    để dựng danh sách đó.

    Đọc dữ liệu theo chunk, không nạp toàn bộ vào bộ nhớ.
    File không có header, gán đủ 9 tên cột theo config.

    Parameters
    ----------
    path : str hoặc Path
        Đường dẫn tới tệp machine_usage.csv.
    chunksize : int, default=2_000_000
        Kích thước chunk mỗi lần đọc.
    cfg : dict, optional
        Cấu hình datasets nếu có.

    Returns
    -------
    pd.Series
        Series có index là machine_id và giá trị là CPU% trung bình.
    """
    e3_cfg = cfg if cfg is not None else _load_e3_config()
    cols = e3_cfg.get("columns", DEFAULT_COLUMNS)
    col_mach = cols[0] if len(cols) > 0 else "machine_id"
    col_target = e3_cfg.get("target", cols[2] if len(cols) > 2 else "cpu_util_percent")

    total_chunks = _estimate_total_chunks(path, chunksize)

    reader = pd.read_csv(
        path,
        header=None,
        names=cols,
        usecols=[col_mach, col_target],
        chunksize=chunksize,
    )

    total_sums = pd.Series(dtype=float)
    total_counts = pd.Series(dtype=int)

    for chunk in tqdm(reader, total=total_chunks, desc="Lượt 1: Quét CPU trung bình (E3)"):
        grp = chunk.groupby(col_mach)[col_target].agg(["sum", "count"])
        total_sums = total_sums.add(grp["sum"], fill_value=0)
        total_counts = total_counts.add(grp["count"], fill_value=0)

    means = total_sums / total_counts
    means.name = "mean_cpu"
    return means


DEFAULT_FROZEN_PATH = "config/e3_machines.txt"


def load_machines_frozen(path: str | Path | None = None) -> list[str]:
    """Đọc danh sách cố định 500 máy của E3.

    Đây là cách duy nhất để xác định tập máy của E3. Không lấy mẫu lại tại chỗ:
    `.sample()` chọn theo vị trí, mà vị trí phụ thuộc thứ tự danh sách máy, nên hai
    cách viết đều đúng về logic vẫn ra hai tập khác nhau (đo được: chỉ trùng 56/500
    và 146/500 máy).

    Dùng danh sách cố định cũng bỏ được lượt quét thứ nhất trên tệp 9 GB.

    Parameters
    ----------
    path : str hoặc Path, optional
        Đường dẫn tệp danh sách. Mặc định `config/e3_machines.txt` ở gốc repo.

    Returns
    -------
    list[str]
        500 machine_id, giữ nguyên thứ tự trong tệp.

    Raises
    ------
    FileNotFoundError
        Khi không tìm thấy tệp danh sách.
    """
    if path is not None:
        p = Path(path)
    else:
        p = Path(DEFAULT_FROZEN_PATH)
        if not p.exists():
            p = Path(__file__).resolve().parents[3] / DEFAULT_FROZEN_PATH

    if not p.exists():
        raise FileNotFoundError(
            f"Không thấy danh sách máy E3 tại {p}. Tệp này có sẵn trong repo, "
            "lấy lại bằng `git checkout config/e3_machines.txt`."
        )

    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_machines(
    path: str | Path,
    machines: list[str] | set[str],
    chunksize: int = 2_000_000,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Lượt quét thứ hai: chỉ giữ các dòng của các máy đã chọn.

    Parameters
    ----------
    path : str hoặc Path
        Đường dẫn tới tệp machine_usage.csv.
    machines : list[str] hoặc set[str]
        Tập hợp machine_id cần giữ lại.
    chunksize : int, default=2_000_000
        Kích thước chunk mỗi lần đọc.
    cfg : dict, optional
        Cấu hình datasets nếu có.

    Returns
    -------
    pd.DataFrame
        DataFrame chứa 3 cột: machine_id, time_stamp, cpu_util_percent.
    """
    e3_cfg = cfg if cfg is not None else _load_e3_config()
    cols = e3_cfg.get("columns", DEFAULT_COLUMNS)
    col_mach = cols[0] if len(cols) > 0 else "machine_id"
    col_time = e3_cfg.get("time_col", cols[1] if len(cols) > 1 else "time_stamp")
    col_target = e3_cfg.get("target", cols[2] if len(cols) > 2 else "cpu_util_percent")
    usecols = [col_mach, col_time, col_target]

    target_set = set(machines)
    total_chunks = _estimate_total_chunks(path, chunksize)

    reader = pd.read_csv(
        path,
        header=None,
        names=cols,
        usecols=usecols,
        chunksize=chunksize,
    )

    chunks = []
    for chunk in tqdm(reader, total=total_chunks, desc="Lượt 2: Nạp dữ liệu máy đã chọn (E3)"):
        filtered = chunk[chunk[col_mach].isin(target_set)]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    df_result = pd.concat(chunks, ignore_index=True)
    return df_result


__all__ = ["machine_means", "load_machines_frozen", "load_machines"]
