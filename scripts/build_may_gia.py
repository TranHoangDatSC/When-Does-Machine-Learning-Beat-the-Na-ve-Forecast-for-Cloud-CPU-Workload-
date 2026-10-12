"""Dựng "máy giả": chia E1 thành E1a (VM đơn lẻ) và E1g (73 nhóm × 5 VM).

    python scripts/build_may_gia.py

Chỉ dựng dữ liệu, **không** chạy model nào. Sinh:

    config/may_gia_nhom_chot.csv                  danh sách nhóm, cố định, xem dưới
    data/processed/E1a.parquet, E1g.parquet  cùng schema data/processed/E1.parquet
    data/features/{E1a,E1g}_{N0,N1,N2}_h{1,6,12}.parquet
    results/tables/normalize_may_gia.csv       cùng schema normalize.csv
    results/tables/cv_may_gia.csv              cùng schema cv.csv

Ba chỗ dễ làm sai:

1. **Hai nửa rời nhau.** Hoán vị lấy trên danh sách `series_id` **đã sắp**, không trên
   thứ tự dòng của parquet: thứ tự dòng đổi thì nhóm đổi mà không ai biết.
2. **Gộp chặt.** Một thành viên `NaN` thì cả điểm gộp là `NaN`. `np.mean` trên hàng có
   `NaN` tự cho `NaN`, nên **không** được dùng `np.nanmean`: nó lấy trung bình phần
   còn lại và sinh bước nhảy mức tải giả mỗi lần một VM vắng mặt.
3. **Danh sách cố định.** Lần chạy đầu ghi `config/may_gia_nhom_chot.csv`. Lần sau chỉ
   **đối chiếu**: dựng lại từ hạt giống mà khác tệp thì thoát 1, không ghi đè. Cùng cách
   làm với danh sách máy E3 ở `config/e3_machines.txt`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.evaluation.tang import bang_cv  # noqa: E402
from cwp.features import MATRIX_COLS, make_feature_matrix  # noqa: E402
from cwp.features.io_ma_tran import ghi as ghi_ma_tran  # noqa: E402

from build_features import bien_doi_bang  # noqa: E402
from run_normalize import bang_thong_ke  # noqa: E402

SEED = 42
N_A = 368          # số VM giữ nguyên ở E1a, phần còn lại chia nhóm 5
K = 5
W = 2304
MODES = ("N0", "N1", "N2")
HORIZONS = (1, 6, 12)
ENV_A, ENV_G, ENV_BO = "E1a", "E1g", "bo"


# ---------------------------------------------------------------- chia nhóm

def chia_nhom(ids, seed: int = SEED, n_a: int = N_A, k: int = K) -> pd.DataFrame:
    """Bảng `env, series_id, vm` — một dòng mỗi VM của E1.

    `env` là `E1a` (VM giữ nguyên, `series_id == vm`), `E1g` (thành viên của một nhóm)
    hoặc `bo` (thừa ra khi chia nhóm, không dùng).
    """
    ids = sorted(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("series_id trùng")
    hv = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    a, b = hv[:n_a], hv[n_a:]
    n_g = len(b) // k
    dong = [{"env": ENV_A, "series_id": s, "vm": s} for s in a]
    for i in range(n_g):
        for s in b[i * k:(i + 1) * k]:
            dong.append({"env": ENV_G, "series_id": f"{ENV_G}_{i:02d}", "vm": s})
    for s in b[n_g * k:]:
        dong.append({"env": ENV_BO, "series_id": "", "vm": s})
    return pd.DataFrame(dong, columns=["env", "series_id", "vm"])


# --------------------------------------------------------------------- gộp

def ma_tran(df: pd.DataFrame, cot: str = "y") -> tuple[list[str], np.ndarray]:
    """Bảng dài → `(danh sách chuỗi đã sắp, mảng (số chuỗi, 2304))`."""
    d = df.sort_values(["series_id", "bucket"], kind="mergesort")
    ids = d["series_id"].drop_duplicates().tolist()
    if len(d) != len(ids) * W:
        raise ValueError(f"{len(d)} dòng không chia hết {len(ids)} × {W}")
    return ids, d[cot].to_numpy().reshape(len(ids), W)


def gop_nhom(e1: pd.DataFrame, nhom: pd.DataFrame) -> pd.DataFrame:
    """Bảng dài của E1g: trung bình **chặt** của các thành viên."""
    ids, Y = ma_tran(e1, "y")
    _, I = ma_tran(e1, "is_interp")
    Y = Y.astype("float64")
    I = I.astype(bool)
    vi_tri = {s: i for i, s in enumerate(ids)}
    buckets = np.sort(e1["bucket"].unique())
    if len(buckets) != W:
        raise ValueError(f"E1 có {len(buckets)} bucket, cần {W}")

    phan = []
    g = nhom[nhom["env"] == ENV_G]
    for sid, tv in g.groupby("series_id", sort=True):
        hang = [vi_tri[v] for v in tv["vm"]]
        y = Y[hang].mean(axis=0)                     # NaN nếu có thành viên NaN
        noi = I[hang].any(axis=0) & np.isfinite(y)
        phan.append(pd.DataFrame({"env": ENV_G, "series_id": sid, "bucket": buckets,
                                  "y": y, "is_interp": noi}))
    return pd.concat(phan, ignore_index=True)


def bang_cv_may_gia(env: str, df: pd.DataFrame) -> pd.DataFrame:
    """CV của máy giả, gọi thẳng `cwp.evaluation.tang.bang_cv`.

    Tầng CV chỉ dùng để lấy mẫu con của SVR, nên phải tính trên phần train như mọi
    môi trường khác.
    """
    return bang_cv(df, env, int(df["bucket"].min()))


# -------------------------------------------------------------------- chạy

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--gop", default="config/may_gia_nhom_chot.csv")
    a = ap.parse_args()
    proc, feat, tab = ROOT / a.processed, ROOT / a.features, ROOT / a.tables
    p_gop = ROOT / a.gop
    t0 = time.time()

    e1 = pd.read_parquet(proc / "E1.parquet")
    nhom = chia_nhom(e1["series_id"].unique())
    if p_gop.exists():
        cu = pd.read_csv(p_gop, keep_default_na=False)
        if not cu.equals(nhom):
            print(f"CHƯA ĐẠT — {p_gop} khác bản dựng lại từ hạt giống {SEED}. "
                  "Không ghi đè: danh sách nhóm là cố định.")
            return 1
        print(f"Đối chiếu {p_gop.name}: khớp bản dựng lại từ hạt giống {SEED}.")
    else:
        nhom.to_csv(p_gop, index=False)
        print(f"Ghi danh sách nhóm {p_gop.relative_to(ROOT)}")
    dem = nhom["env"].value_counts()
    print(f"   E1a {dem.get(ENV_A, 0)} VM · E1g {nhom.loc[nhom.env == ENV_G, 'series_id'].nunique()}"
          f" nhóm × {K} = {dem.get(ENV_G, 0)} VM · bỏ {dem.get(ENV_BO, 0)}: "
          f"{', '.join(nhom.loc[nhom.env == ENV_BO, 'vm'])}")

    ea = e1[e1["series_id"].isin(nhom.loc[nhom.env == ENV_A, "vm"])].copy()
    ea["env"] = ENV_A
    eg = gop_nhom(e1, nhom)
    bang = {ENV_A: ea.reset_index(drop=True), ENV_G: eg}

    tk, cv = [], []
    for env, df in bang.items():
        df = df[["env", "series_id", "bucket", "y", "is_interp"]]
        df.to_parquet(proc / f"{env}.parquet", index=False)
        ids, Y = ma_tran(df, "y")
        tk.append(bang_thong_ke(env, ids, Y.astype("float64")))
        cv.append(bang_cv_may_gia(env, df))
        print(f"{env}: {len(ids)} chuỗi, NaN {100 * float(np.isnan(Y.astype(float)).mean()):.2f}%")
        for mode in MODES:
            dfm = bien_doi_bang(df, mode)
            for h in HORIZONS:
                X = make_feature_matrix(dfm, h=h)
                assert list(X.columns) == MATRIX_COLS, "schema lệch khỏi MATRIX_COLS"
                # N0 mang tên không hậu tố, đúng quy ước của build_features.py, để
                # cwp.quy_trinh.chay đọc được. Ghi kèm nhãn lược đồ.
                ten = (f"{env}_h{h}.parquet" if mode in (None, "N0")
                       else f"{env}_{mode}_h{h}.parquet")
                ghi_ma_tran(X, feat / ten)
                print(f"   {mode} h={h:<2} {len(X):>9,} dòng  "
                      f"{X['series_id'].nunique()} chuỗi")

    pd.concat(tk, ignore_index=True).to_csv(tab / "normalize_may_gia.csv", index=False)
    pd.concat(cv, ignore_index=True).to_csv(tab / "cv_may_gia.csv", index=False)
    print(f"\nĐã ghi normalize_may_gia.csv, cv_may_gia.csv · {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
