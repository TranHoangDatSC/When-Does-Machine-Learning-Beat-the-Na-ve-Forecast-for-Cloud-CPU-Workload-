"""Tiền xử lý dữ liệu thô của ba môi trường E1, E2, E3."""

import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from cwp.io.bitbrains import load_raw, make_series_id
from cwp.preprocess.clean import clip_cpu
from cwp.preprocess.filter import count_valid_rows, judge
from cwp.preprocess.resample import apply_window, interpolate_short, to_grid


def load_yaml(path: str | Path) -> dict:
    """Đọc file YAML cấu hình."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def doc_n_train(window_len: int) -> int:
    """Số điểm đầu cửa sổ được phép dùng để quyết định giữ hay loại một chuỗi.

    Lấy tỉ lệ `train` của `config/split.yaml`, cùng con số mà bước chia dữ liệu dùng
    để đặt ranh giới. Quyết định chọn quần thể vì vậy nằm trọn trong phần lịch sử.
    """
    root = Path(__file__).resolve().parents[3]
    path = Path("config/split.yaml")
    if not path.exists():
        path = root / "config" / "split.yaml"
    cfg = load_yaml(path) if path.exists() else {}
    frac = float(cfg.get("train", 0.70))
    return int(window_len * frac)


def get_config_paths() -> tuple[dict, dict]:
    """Lấy cấu hình datasets và preprocess."""
    root = Path(__file__).resolve().parents[3]
    d_path = Path("config/datasets.yaml")
    if not d_path.exists():
        d_path = root / "config" / "datasets.yaml"
    p_path = Path("config/preprocess.yaml")
    if not p_path.exists():
        p_path = root / "config" / "preprocess.yaml"

    cfg_data = load_yaml(d_path)
    cfg_prep = load_yaml(p_path)
    return cfg_data, cfg_prep


# Một lần chạy = một dấu thời gian. Mọi dòng sinh trong cùng tiến trình mang cùng
# giá trị, nên nhìn catalog là biết nó có bị ghép từ nhiều lần chạy hoặc nhiều máy
# hay không.
_BUILT_ON = platform.node()
_BUILT_AT = datetime.now().isoformat(timespec="seconds")


def _stamp_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn nguồn gốc (máy nào, lúc nào) vào mọi dòng catalog."""
    df = df.copy()
    df["built_on"] = _BUILT_ON
    df["built_at"] = _BUILT_AT
    return df


def process_env_bitbrains(
    env: str, cfg_data: dict, cfg_prep: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Xử lý tiền xử lý cho môi trường Bitbrains (E1 hoặc E2)."""
    e_cfg = cfg_data[env]
    data_path = Path(e_cfg["path"])
    glob_pattern = e_cfg.get("glob", "*.csv")
    files = sorted(data_path.glob(glob_pattern))

    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file nào khớp '{glob_pattern}' tại: {data_path}"
        )

    grid_sec = cfg_prep.get("grid_seconds", 300)
    window_days = cfg_prep.get("window", {}).get("days", 8)
    window_len = int(window_days * 86400 // grid_sec)
    n_train = int(cfg_prep["_n_train"]) if cfg_prep.get("_n_train") else doc_n_train(window_len)
    k_gap = cfg_prep.get("gap", {}).get("max_len", 2)
    max_lag = cfg_prep.get("row_validity", {}).get("max_lag", 24)

    # Tháng của E2 lấy từ tên thư mục trong config, không ghi cứng: muốn đổi E2 sang
    # tháng khác thì chỉ cần sửa config/datasets.yaml.
    # Môi trường nào khoá theo tháng thì lấy tên thư mục làm phần tháng của series_id.
    month = data_path.name if "month" in (e_cfg.get("id_key") or []) else None

    # Lượt 1: Đọc raw, clip CPU, căn lưới và tìm min bucket toàn cục
    series_cache = []
    min_buckets = []
    total_raw_rows = 0
    total_clipped = 0

    print(f"\n--- Đang xử lý môi trường {env} ---")
    for f in tqdm(files, desc=f"Lượt 1 ({env}) - Căn lưới"):
        df = load_raw(f, cfg=cfg_data, env=env)
        total_raw_rows += len(df)

        s_clipped, hi, lo = clip_cpu(df["cpu_pct"], cfg=cfg_prep)
        total_clipped += hi

        g = to_grid(df["time_s"], s_clipped, grid=grid_sec, cfg=cfg_prep)
        if not g.empty:
            min_buckets.append(int(g.index.min()))
        series_cache.append((f, g, hi))

    if not min_buckets:
        raise ValueError(f"Không có dữ liệu hợp lệ trong môi trường {env}.")

    b0 = min(min_buckets)
    print(f"Mốc thời gian bắt đầu toàn cục b0 = {b0}")

    # Lượt 2: Cắt cửa sổ toàn cục [b0, b0 + window_len), nội suy, lọc và tính thống kê
    catalog_rows = []
    processed_records = []

    reasons = {"ngoai_cua_so": 0, "gan_chet": 0, "hang": 0, "it_dong": 0}
    kept_count = 0
    total_h1 = 0
    total_h6 = 0
    total_h12 = 0
    total_interp = 0
    total_pts = 0

    for f, g, hi in tqdm(
        series_cache, desc=f"Lượt 2 ({env}) - Cắt cửa sổ & lọc chuỗi"
    ):
        sid = make_series_id(env, f, month=month)

        w = apply_window(g, b0, window_len, cfg=cfg_prep)

        if w.empty:
            reason = "ngoai_cua_so"
            reasons[reason] += 1
            catalog_rows.append({
                "env": env,
                "series_id": sid,
                "kept": False,
                "reject_reason": reason,
                "n_points": 0,
                "n_interp": 0,
                "valid_rows_h1": 0,
                "valid_rows_h6": 0,
                "valid_rows_h12": 0,
                "mean": np.nan,
                "mean_train": np.nan,
                "p50": np.nan,
                "std": np.nan,
                "n_clipped": hi,
            })
            continue

        # Căn đầy đủ cửa sổ toàn cục 2304 bucket
        w_full = w.reindex(pd.RangeIndex(b0, b0 + window_len))
        was_nan = w_full.isna()

        w_interp, n_int = interpolate_short(w_full, k=k_gap, cfg=cfg_prep)
        is_interp = was_nan & w_interp.notna()

        reason = judge(w_interp, cfg=cfg_prep, n_train=n_train)

        n_pts = int(w_interp.notna().sum())
        noi_mask = is_interp.to_numpy(dtype=bool)
        vh1 = count_valid_rows(w_interp, max_lag=max_lag, h=1, cfg=cfg_prep,
                               noi_suy=noi_mask)
        vh6 = count_valid_rows(w_interp, max_lag=max_lag, h=6, cfg=cfg_prep,
                               noi_suy=noi_mask)
        vh12 = count_valid_rows(w_interp, max_lag=max_lag, h=12, cfg=cfg_prep,
                                noi_suy=noi_mask)

        clean_obs = w.dropna()
        train_obs = w_interp.iloc[:n_train].dropna()
        mean_train = float(train_obs.mean()) if len(train_obs) else np.nan
        mean_val = float(clean_obs.mean()) if len(clean_obs) else np.nan
        p50_val = float(clean_obs.median()) if len(clean_obs) else np.nan
        std_val = float(clean_obs.std()) if len(clean_obs) > 1 else 0.0

        if reason is not None:
            reasons[reason] += 1
            catalog_rows.append({
                "env": env,
                "series_id": sid,
                "kept": False,
                "reject_reason": reason,
                "n_points": n_pts,
                "n_interp": n_int,
                "valid_rows_h1": vh1,
                "valid_rows_h6": vh6,
                "valid_rows_h12": vh12,
                "mean": mean_val,
                "mean_train": mean_train,
                "p50": p50_val,
                "std": std_val,
                "n_clipped": hi,
            })
        else:
            kept_count += 1
            total_h1 += vh1
            total_h6 += vh6
            total_h12 += vh12
            total_interp += n_int
            total_pts += n_pts

            catalog_rows.append({
                "env": env,
                "series_id": sid,
                "kept": True,
                "reject_reason": "",
                "n_points": n_pts,
                "n_interp": n_int,
                "valid_rows_h1": vh1,
                "valid_rows_h6": vh6,
                "valid_rows_h12": vh12,
                "mean": mean_val,
                "mean_train": mean_train,
                "p50": p50_val,
                "std": std_val,
                "n_clipped": hi,
            })

            # Lưu vào bảng processed
            for b_idx, y_val, interp_flag in zip(
                w_interp.index, w_interp.values, is_interp.values
            ):
                processed_records.append({
                    "env": env,
                    "series_id": sid,
                    "bucket": int(b_idx),
                    "y": float(y_val) if not np.isnan(y_val) else np.nan,
                    "is_interp": bool(interp_flag),
                })

    ti_le_noi_suy_pct = (
        (total_interp / total_pts * 100) if total_pts > 0 else 0.0
    )
    ti_le_clip_pct = (
        (total_clipped / total_raw_rows * 100) if total_raw_rows > 0 else 0.0
    )

    print(f"\nChuỗi vào       : {len(files)}")
    print(f"Loại ngoai_cua_so: {reasons['ngoai_cua_so']}")
    print(f"Loại gan_chet   : {reasons['gan_chet']}")
    print(f"Loại hang       : {reasons['hang']}")
    print(f"Loại it_dong    : {reasons['it_dong']}")
    print(f"Chuỗi còn lại   : {kept_count}")
    print(f"Dòng hợp lệ h=12: {total_h12}")
    print(f"Tỉ lệ nội suy   : {ti_le_noi_suy_pct:.3f}%")
    print(f"Tỉ lệ clip      : {ti_le_clip_pct:.4f}%\n")

    cat_df = _stamp_provenance(pd.DataFrame(catalog_rows)).astype({
        "env": "object",
        "series_id": "object",
        "kept": "bool",
        "reject_reason": "object",
        "n_points": "int64",
        "n_interp": "int64",
        "valid_rows_h1": "int64",
        "valid_rows_h6": "int64",
        "valid_rows_h12": "int64",
        "mean": "float64",
        "mean_train": "float64",
        "p50": "float64",
        "std": "float64",
        "n_clipped": "int64",
        "built_on": "object",
        "built_at": "object",
    })

    if processed_records:
        proc_df = pd.DataFrame(processed_records).astype({
            "env": "object",
            "series_id": "object",
            "bucket": "int64",
            "y": "float64",
            "is_interp": "bool",
        })
    else:
        proc_df = pd.DataFrame({
            "env": pd.Series(dtype="object"),
            "series_id": pd.Series(dtype="object"),
            "bucket": pd.Series(dtype="int64"),
            "y": pd.Series(dtype="float64"),
            "is_interp": pd.Series(dtype="bool"),
        })

    return cat_df, proc_df


def process_env_alibaba(
    cfg_data: dict, cfg_prep: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Xử lý tiền xử lý cho môi trường Alibaba (E3)."""
    from cwp.io.alibaba import load_machines, load_machines_frozen

    e_cfg = cfg_data["E3"]
    data_path = Path(e_cfg["path"])

    print("\n--- Đang xử lý môi trường E3 (Alibaba) ---")

    # Danh sách máy cố định. Không lấy mẫu lại tại chỗ: kết quả phụ thuộc thứ tự
    # quét tệp. Nhờ vậy cũng bỏ được lượt quét thứ nhất trên tệp 9 GB.
    selected_machines = load_machines_frozen()
    selected_set = set(selected_machines)
    print(f"Dùng danh sách cố định: {len(selected_machines)} máy "
          f"(config/e3_machines.txt)")

    print("Nạp dữ liệu các máy đã chọn...")
    df_raw = load_machines(data_path, selected_machines)

    grid_sec = cfg_prep.get("grid_seconds", 300)
    window_days = cfg_prep.get("window", {}).get("days", 8)
    window_len = int(window_days * 86400 // grid_sec)
    n_train = int(cfg_prep["_n_train"]) if cfg_prep.get("_n_train") else doc_n_train(window_len)
    k_gap = cfg_prep.get("gap", {}).get("max_len", 2)
    max_lag = cfg_prep.get("row_validity", {}).get("max_lag", 24)

    # Căn lưới cho từng máy
    series_cache = []
    min_buckets = []
    total_raw_rows = len(df_raw)
    total_clipped = 0

    for m_id, group in tqdm(
        df_raw.groupby("machine_id"), desc="Căn lưới E3", total=len(selected_set)
    ):
        s_clipped, hi, lo = clip_cpu(group["cpu_util_percent"], cfg=cfg_prep)
        total_clipped += hi
        g = to_grid(
            group["time_stamp"], s_clipped, grid=grid_sec, cfg=cfg_prep
        )
        if not g.empty:
            min_buckets.append(int(g.index.min()))
        series_cache.append((m_id, g, hi))

    b0 = min(min_buckets)
    print(f"Mốc thời gian bắt đầu toàn cục E3 b0 = {b0}")

    catalog_rows = []
    processed_records = []
    reasons = {"ngoai_cua_so": 0, "gan_chet": 0, "hang": 0, "it_dong": 0}
    kept_count = 0
    total_h12 = 0
    total_interp = 0
    total_pts = 0

    for m_id, g, hi in tqdm(series_cache, desc="Lọc chuỗi E3"):
        sid = f"E3_{m_id}"
        w = apply_window(g, b0, window_len, cfg=cfg_prep)

        if w.empty:
            reason = "ngoai_cua_so"
            reasons[reason] += 1
            catalog_rows.append({
                "env": "E3",
                "series_id": sid,
                "kept": False,
                "reject_reason": reason,
                "n_points": 0,
                "n_interp": 0,
                "valid_rows_h1": 0,
                "valid_rows_h6": 0,
                "valid_rows_h12": 0,
                "mean": np.nan,
                "mean_train": np.nan,
                "p50": np.nan,
                "std": np.nan,
                "n_clipped": hi,
            })
            continue

        w_full = w.reindex(pd.RangeIndex(b0, b0 + window_len))
        was_nan = w_full.isna()
        w_interp, n_int = interpolate_short(w_full, k=k_gap, cfg=cfg_prep)
        is_interp = was_nan & w_interp.notna()

        reason = judge(w_interp, cfg=cfg_prep, n_train=n_train)

        n_pts = int(w_interp.notna().sum())
        noi_mask = is_interp.to_numpy(dtype=bool)
        vh1 = count_valid_rows(w_interp, max_lag=max_lag, h=1, cfg=cfg_prep,
                               noi_suy=noi_mask)
        vh6 = count_valid_rows(w_interp, max_lag=max_lag, h=6, cfg=cfg_prep,
                               noi_suy=noi_mask)
        vh12 = count_valid_rows(w_interp, max_lag=max_lag, h=12, cfg=cfg_prep,
                                noi_suy=noi_mask)

        clean_obs = w.dropna()
        train_obs = w_interp.iloc[:n_train].dropna()
        mean_train = float(train_obs.mean()) if len(train_obs) else np.nan
        mean_val = float(clean_obs.mean()) if len(clean_obs) else np.nan
        p50_val = float(clean_obs.median()) if len(clean_obs) else np.nan
        std_val = float(clean_obs.std()) if len(clean_obs) > 1 else 0.0

        if reason is not None:
            reasons[reason] += 1
            catalog_rows.append({
                "env": "E3",
                "series_id": sid,
                "kept": False,
                "reject_reason": reason,
                "n_points": n_pts,
                "n_interp": n_int,
                "valid_rows_h1": vh1,
                "valid_rows_h6": vh6,
                "valid_rows_h12": vh12,
                "mean": mean_val,
                "mean_train": mean_train,
                "p50": p50_val,
                "std": std_val,
                "n_clipped": hi,
            })
        else:
            kept_count += 1
            total_h12 += vh12
            total_interp += n_int
            total_pts += n_pts

            catalog_rows.append({
                "env": "E3",
                "series_id": sid,
                "kept": True,
                "reject_reason": "",
                "n_points": n_pts,
                "n_interp": n_int,
                "valid_rows_h1": vh1,
                "valid_rows_h6": vh6,
                "valid_rows_h12": vh12,
                "mean": mean_val,
                "mean_train": mean_train,
                "p50": p50_val,
                "std": std_val,
                "n_clipped": hi,
            })

            for b_idx, y_val, interp_flag in zip(
                w_interp.index, w_interp.values, is_interp.values
            ):
                processed_records.append({
                    "env": "E3",
                    "series_id": sid,
                    "bucket": int(b_idx),
                    "y": float(y_val) if not np.isnan(y_val) else np.nan,
                    "is_interp": bool(interp_flag),
                })

    ti_le_noi_suy_pct = (
        (total_interp / total_pts * 100) if total_pts > 0 else 0.0
    )
    ti_le_clip_pct = (
        (total_clipped / total_raw_rows * 100) if total_raw_rows > 0 else 0.0
    )

    print(f"\nChuỗi vào       : {len(selected_machines)}")
    print(f"Loại ngoai_cua_so: {reasons['ngoai_cua_so']}")
    print(f"Loại gan_chet   : {reasons['gan_chet']}")
    print(f"Loại hang       : {reasons['hang']}")
    print(f"Loại it_dong    : {reasons['it_dong']}")
    print(f"Chuỗi còn lại   : {kept_count}")
    print(f"Dòng hợp lệ h=12: {total_h12}")
    print(f"Tỉ lệ nội suy   : {ti_le_noi_suy_pct:.3f}%")
    print(f"Tỉ lệ clip      : {ti_le_clip_pct:.4f}%\n")

    cat_df = _stamp_provenance(pd.DataFrame(catalog_rows)).astype({
        "env": "object",
        "series_id": "object",
        "kept": "bool",
        "reject_reason": "object",
        "n_points": "int64",
        "n_interp": "int64",
        "valid_rows_h1": "int64",
        "valid_rows_h6": "int64",
        "valid_rows_h12": "int64",
        "mean": "float64",
        "mean_train": "float64",
        "p50": "float64",
        "std": "float64",
        "n_clipped": "int64",
        "built_on": "object",
        "built_at": "object",
    })

    proc_df = pd.DataFrame(processed_records).astype({
        "env": "object",
        "series_id": "object",
        "bucket": "int64",
        "y": "float64",
        "is_interp": "bool",
    })

    return cat_df, proc_df


def save_results(env: str, cat_df: pd.DataFrame, proc_df: pd.DataFrame,
                 proc_dir: str | Path = "data/processed",
                 catalog: str | Path = "data/catalog.parquet"):
    """Lưu kết quả parquet ra thư mục đã chọn và cập nhật catalog."""
    proc_dir = Path(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    proc_file = proc_dir / f"{env}.parquet"
    proc_df.to_parquet(proc_file, index=False)
    print(f"Đã lưu: {proc_file} ({len(proc_df)} dòng)")

    catalog_path = Path(catalog)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    if catalog_path.exists():
        old_cat = pd.read_parquet(catalog_path)
        old_other = old_cat[old_cat["env"] != env]
        final_cat = pd.concat([old_other, cat_df], ignore_index=True)
    else:
        final_cat = cat_df

    final_cat.to_parquet(catalog_path, index=False)
    print(
        f"Đã cập nhật: {catalog_path} (tổng {len(final_cat)} chuỗi từ các môi trường)"
    )


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Tiền xử lý dữ liệu thô: lưới 5 phút, cửa sổ 8 ngày, lọc chuỗi."
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Mã môi trường, phân tách bằng dấu phẩy, hoặc all cho E1, E2, E3",
    )
    parser.add_argument("--gap-max-len", type=int, default=None,
                        help="độ dài cụm NaN tối đa được nội suy; 0 là không nội suy")
    parser.add_argument("--window-days", type=float, default=None,
                        help="độ dài cửa sổ tính bằng ngày, ghi đè config")
    parser.add_argument("--n-train", type=int, default=None,
                        help="số điểm đầu dùng để quyết định giữ hay loại chuỗi; "
                             "bỏ trống thì lấy tỉ lệ train của config/split.yaml")
    parser.add_argument("--min-mean", type=float, default=None,
                        help="ngưỡng CPU trung bình phần train, dùng cho phần độ nhạy")
    parser.add_argument("--processed", default="data/processed",
                        help="thư mục ghi bảng dài")
    parser.add_argument("--catalog", default="data/catalog.parquet",
                        help="đường dẫn catalog")
    args = parser.parse_args()

    cfg_data, cfg_prep = get_config_paths()
    if args.gap_max_len is not None:
        cfg_prep.setdefault("gap", {})["max_len"] = int(args.gap_max_len)
    if args.window_days is not None:
        cfg_prep.setdefault("window", {})["days"] = float(args.window_days)
    if args.n_train is not None:
        cfg_prep["_n_train"] = int(args.n_train)
    if args.min_mean is not None:
        cfg_prep.setdefault("filter", {})["min_mean"] = float(args.min_mean)

    envs_to_run = (["E1", "E2", "E3"] if args.env == "all"
                   else [x.strip() for x in args.env.split(",")])

    print(f"n_train loc = {cfg_prep.get('_n_train') or 'theo tỉ lệ train'}")
    print(f"gap.max_len = {cfg_prep.get('gap', {}).get('max_len')}  "
          f"window.days = {cfg_prep.get('window', {}).get('days')}  "
          f"filter.min_mean = {cfg_prep.get('filter', {}).get('min_mean')}")
    print(f"ghi vào {args.processed} và {args.catalog}")

    for e in envs_to_run:
        if e not in cfg_data:
            raise SystemExit(
                f"config/datasets.yaml không có mục {e!r}. "
                f"Có: {sorted(k for k in cfg_data)}"
            )
        if cfg_data[e].get("entity") == "machine":
            cat_df, proc_df = process_env_alibaba(cfg_data, cfg_prep)
        else:
            cat_df, proc_df = process_env_bitbrains(e, cfg_data, cfg_prep)
        save_results(e, cat_df, proc_df, args.processed, args.catalog)


if __name__ == "__main__":
    main()
