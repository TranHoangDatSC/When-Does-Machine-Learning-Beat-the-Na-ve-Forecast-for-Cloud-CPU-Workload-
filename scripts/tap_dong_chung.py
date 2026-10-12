"""Tập dòng đánh giá chung cho các mô hình được so trực tiếp.

    python scripts/tap_dong_chung.py --env E1,E2,E3 --horizons 1,6,12
    python scripts/tap_dong_chung.py --env E1L,E2m07,E2m09 --horizons 1,12 --tuan

Không mô hình nào được chấm trên một tập dòng riêng. Ba điều kiện khác nhau về độ
khắt khe:

    bang        dòng hợp lệ của ma trận đặc trưng: cửa sổ [t-24, t] sạch, y(t) và
                y(t+h) là quan sát thật
    ngay        có y(t+h-288), tức seasonal naive theo ngày chấm được
    tuan        có y(t+h-2016), tức seasonal naive theo tuần chấm được
    chuoi       288 điểm [t-287, t] không có NaN, tức mô hình chuỗi chấm được

Tệp đầu ra giữ cả bốn cột, và `scripts/tong_hop.py` chọn giao của những cột ứng
với các mô hình thật sự có mặt trong phép so. Nhờ vậy một phép so chỉ giữa các mô
hình bảng không bị cắt bớt dòng vì DLinear cần lịch sử dài hơn, còn khi DLinear tham
gia thì mọi mô hình cùng chấm trên đúng một tập dòng.

Số dòng bị loại ở từng điều kiện ghi ra `tom_tat.csv` để báo trong bài.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.evaluation.splits import offset  # noqa: E402
from cwp.features.io_ma_tran import doc as doc_ma_tran  # noqa: E402
from cwp.models.baselines import SEASONAL_LAG, SEASONAL_LAG_TUAN  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Độ dài cửa sổ đầu vào của mô hình chuỗi, khớp với scripts/dl.py.
L_CHUOI = 288


def mot_to_hop(env: str, h: int, feat_dir: Path, proc_dir: Path,
               co_tuan: bool) -> tuple:
    X = doc_ma_tran(feat_dir / f"{env}_h{h}.parquet",
                    cot=["series_id", "bucket"])
    proc = pd.read_parquet(proc_dir / f"{env}.parquet")
    b0 = int(proc["bucket"].min())

    y = proc.set_index(["series_id", "bucket"])["y"]
    noi = proc.set_index(["series_id", "bucket"])["is_interp"]
    sid = X["series_id"].to_numpy()
    bk = X["bucket"].to_numpy("int64")

    def co_that(dich: np.ndarray) -> np.ndarray:
        k = pd.MultiIndex.from_arrays([sid, dich])
        v = y.reindex(k).to_numpy("float64")
        n = noi.reindex(k).fillna(True).to_numpy(dtype=bool)
        return np.isfinite(v) & (~n)

    out = pd.DataFrame({"series_id": sid, "bucket": bk})
    out["bang"] = True                      # đã lọc sẵn khi sinh ma trận
    out["ngay"] = co_that(bk + h - SEASONAL_LAG)
    out["tuan"] = co_that(bk + h - SEASONAL_LAG_TUAN) if co_tuan else False

    # Cửa sổ chuỗi: 288 điểm liên tiếp không NaN. Tính bằng tổng tích luỹ trên lưới
    # đã điền đủ bucket của data/processed/.
    p = proc.sort_values(["series_id", "bucket"])
    n_chuoi = p["series_id"].nunique()
    m = len(p) // n_chuoi
    if len(p) != n_chuoi * m:
        raise SystemExit(f"{env}: lưới không đều, {len(p)} dòng / {n_chuoi} chuỗi.")
    ids = p["series_id"].to_numpy()[::m]
    Y = p["y"].to_numpy("float64").reshape(n_chuoi, m)
    co = np.isfinite(Y).astype("int32")
    tich = np.cumsum(co, axis=1)
    du = np.zeros_like(co, dtype=bool)
    for t in range(L_CHUOI - 1, m):
        du[:, t] = (tich[:, t] - (tich[:, t - L_CHUOI] if t >= L_CHUOI else 0)) == L_CHUOI
    vi_tri = {s: i for i, s in enumerate(ids)}
    hang = np.array([vi_tri[s] for s in sid])
    out["chuoi"] = du[hang, offset(bk, b0)]

    tt = {
        "env": env, "h": h, "n_dong": len(out),
        "co_ngay": int(out["ngay"].sum()),
        "co_tuan": int(out["tuan"].sum()),
        "co_chuoi": int(out["chuoi"].sum()),
        "giao_bang_ngay": int((out["ngay"]).sum()),
        "giao_bang_ngay_chuoi": int((out["ngay"] & out["chuoi"]).sum()),
    }
    tt["bo_vi_ngay_pct"] = round(100 * (1 - tt["co_ngay"] / tt["n_dong"]), 4)
    tt["bo_vi_chuoi_pct"] = round(100 * (1 - tt["co_chuoi"] / tt["n_dong"]), 4)
    return out, tt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="E1,E2,E3")
    ap.add_argument("--horizons", default="1,6,12")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/eval_rows")
    ap.add_argument("--tuan", action="store_true",
                    help="tính thêm điều kiện seasonal theo tuần, cho Thí nghiệm B")
    a = ap.parse_args()

    envs = [x.strip() for x in a.env.split(",")]
    hs = [int(x) for x in a.horizons.split(",")]
    feat_dir, proc_dir = ROOT / a.features, ROOT / a.processed
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    print()
    print("TẬP DÒNG ĐÁNH GIÁ CHUNG")
    print(f"python: {sys.executable}")
    bang = []
    for env in envs:
        for h in hs:
            t0 = time.time()
            d, tt = mot_to_hop(env, h, feat_dir, proc_dir, a.tuan)
            d.to_parquet(out / f"{env}_h{h}.parquet", index=False)
            tt["giay"] = round(time.time() - t0, 1)
            bang.append(tt)
            print(f"{env} h={h:<2} {tt['n_dong']:>9,} dòng · thiếu ngày "
                  f"{tt['bo_vi_ngay_pct']:.3f}% · thiếu cửa sổ chuỗi "
                  f"{tt['bo_vi_chuoi_pct']:.3f}% ({tt['giay']}s)")
    p = out / "tom_tat.csv"
    df = pd.DataFrame(bang)
    if p.exists():
        cu = pd.read_csv(p)
        khoa = ["env", "h"]
        moi = set(map(tuple, df[khoa].astype(str).to_numpy()))
        cu = cu[~cu[khoa].astype(str).apply(tuple, axis=1).isin(moi)]
        df = pd.concat([cu, df], ignore_index=True)
    df.to_csv(p, index=False)
    print(f"Đã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
