"""Thống kê chuẩn hoá từng chuỗi, và ba bất biến kiểm tra trên DỮ LIỆU THẬT.

Sinh hai bảng:

    results/tables/normalize.csv    env, mode, series_id, mu, sd, n_dong_train
    results/tables/invariants.csv   env, bat_bien, lech_toi_da, nguong, dat

Chạy:
    python scripts/run_normalize.py

Ba bất biến là ba tính chất phải đúng về mặt toán học (notebook 05): naive và trung
bình trượt qua N1 rồi đưa về thang gốc phải trùng N0, và naive ở N2 với `Δ̂ = 0` phải
trùng naive ở N0. Script kiểm chúng trên **toàn bộ chuỗi thật của cả ba môi trường**,
nơi có lỗ hổng `NaN`, có chuỗi chạm trần 100, và có chuỗi gần hằng.

Chỉ đọc `data/processed/`, không sửa gì.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.preprocess.normalize import (  # noqa: E402
    TRAIN_END,
    bien_doi,
    map_nguoc,
    thong_ke_train,
)

ENVS = ("E1", "E2", "E3")
W = 2304
MODES = ("N0", "N1", "N2")

# Ngưỡng của ba bất biến. Cái thứ ba phải bằng 0 TUYỆT ĐỐI: nó chỉ là phép cộng
# `y_t + 0`, sai số dấu phẩy động không có chỗ chen vào.
NGUONG = {"naive_N0_vs_N1": 1e-9, "ma6_N0_vs_N1": 1e-9, "naive_N0_vs_N2": 0.0}


def nap(env: str) -> tuple[list[str], np.ndarray]:
    """Đọc một môi trường thành ma trận `(số chuỗi, 2304)`."""
    d = pd.read_parquet(ROOT / "data" / "processed" / f"{env}.parquet",
                        columns=["series_id", "bucket", "y"])
    d = d.sort_values(["series_id", "bucket"], kind="mergesort")
    ids = d["series_id"].drop_duplicates().tolist()
    if len(d) != len(ids) * W:
        raise ValueError(f"{env}: {len(d)} dòng không chia hết {len(ids)} × {W}")
    return ids, d["y"].to_numpy(dtype="float64").reshape(len(ids), W)


def ma6(a: np.ndarray) -> np.ndarray:
    """Trung bình 6 điểm quá khứ `a[t-5..t]`, vector hoá theo hàng."""
    n, m = a.shape
    out = np.full((n, m), np.nan)
    sw = np.lib.stride_tricks.sliding_window_view(a, 6, axis=1)
    out[:, 5:] = sw.mean(axis=2)
    return out


def bang_thong_ke(env: str, ids: list[str], Y: np.ndarray) -> pd.DataFrame:
    """Một dòng mỗi `(mode, chuỗi)`.

    `mu`/`sd` chỉ có nghĩa với **N1** — N0 không biến đổi gì, N2 map ngược bằng mốc
    neo `y_t` chứ không bằng tham số nào. Hai chế độ đó để `NaN`, **không** điền 0/1
    cho đủ ô: điền số giả vào chỗ không có tham số là mời người đọc hiểu nhầm.
    """
    dong = []
    for i, sid in enumerate(ids):
        mu, sd = thong_ke_train(Y[i])
        n_tr = int(np.isfinite(Y[i, :TRAIN_END]).sum())
        for mode in MODES:
            co = mode == "N1"
            dong.append({
                "env": env, "mode": mode, "series_id": sid,
                "mu": mu if co else np.nan,
                "sd": sd if co else np.nan,
                "n_dong_train": n_tr,
            })
    return pd.DataFrame(dong)


def bang_bat_bien(env: str, Y: np.ndarray) -> pd.DataFrame:
    """Ba bất biến, đo trên mọi chuỗi của môi trường, lấy lệch LỚN NHẤT."""
    n = Y.shape[0]
    Z = np.vstack([bien_doi(Y[i], "N1") for i in range(n)])

    # 1 & 2: naive và ma6 đi vòng qua N1 rồi map ngược.
    naive_n1 = np.vstack([map_nguoc(Z[i], Y[i], "N1") for i in range(n)])
    ma6_goc, ma6_z = ma6(Y), ma6(Z)
    ma6_n1 = np.vstack([map_nguoc(ma6_z[i], Y[i], "N1") for i in range(n)])

    # Chuỗi sd = 0 cho NaN trọn vẹn ở N1 — loại khỏi phép so, và đếm riêng.
    dung_duoc = np.isfinite(Z).any(axis=1)

    def lech(a: np.ndarray, b: np.ndarray) -> float:
        d = np.abs(a[dung_duoc] - b[dung_duoc])
        return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0

    # 3: N2 với Δ̂ = 0 phải trùng naive TUYỆT ĐỐI.
    n2 = np.vstack([map_nguoc(np.zeros(W), Y[i], "N2") for i in range(n)])
    d3 = np.abs(n2 - Y)
    l3 = float(np.nanmax(d3)) if np.isfinite(d3).any() else 0.0

    ket = {
        "naive_N0_vs_N1": lech(Y, naive_n1),
        "ma6_N0_vs_N1": lech(ma6_goc, ma6_n1),
        "naive_N0_vs_N2": l3,
    }
    return pd.DataFrame([
        {"env": env, "bat_bien": k, "lech_toi_da": v,
         "nguong": NGUONG[k], "dat": bool(v <= NGUONG[k])}
        for k, v in ket.items()
    ])


def main() -> int:
    tab = ROOT / "results" / "tables"
    tab.mkdir(parents=True, exist_ok=True)

    tk, bb = [], []
    print("\nChuẩn hoá và ba bất biến trên dữ liệu thật\n")
    for env in ENVS:
        ids, Y = nap(env)
        t = bang_thong_ke(env, ids, Y)
        b = bang_bat_bien(env, Y)
        tk.append(t)
        bb.append(b)

        n1 = t[t["mode"] == "N1"]
        khong = int((~np.isfinite(n1["sd"]) | (n1["sd"] == 0)).sum())
        print(f"{env}: {len(ids)} chuỗi | mu p50 = {n1['mu'].median():.6f} | "
              f"sd p50 = {n1['sd'].median():.6f} | sd = 0: {khong} chuỗi")
        for _, r in b.iterrows():
            print(f"   [{'  ok  ' if r['dat'] else 'TRƯỢT!'}] {r['bat_bien']:24}"
                  f" lệch tối đa {r['lech_toi_da']:.3e}  (ngưỡng {r['nguong']:.0e})")

    pd.concat(tk, ignore_index=True).to_csv(tab / "normalize.csv", index=False)
    bb_all = pd.concat(bb, ignore_index=True)
    bb_all.to_csv(tab / "invariants.csv", index=False)

    print(f"\nĐã ghi: {tab / 'normalize.csv'}")
    print(f"Đã ghi: {tab / 'invariants.csv'}")
    ok = bool(bb_all["dat"].all())
    print("\n" + "=" * 66)
    print("ĐẠT — ba bất biến đúng trên cả ba môi trường." if ok
          else "TRƯỢT — có bất biến hỏng trên dữ liệu thật. Dừng lại truy nguyên.")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
