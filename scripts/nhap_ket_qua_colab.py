"""Kiểm tra rồi nhập kết quả GRU tải về từ Colab.

    python scripts/nhap_ket_qua_colab.py --zip results/colab/ket_qua_gru.zip

Tệp zip tải về từ Colab phải có:

    du_doan/*.parquet   dự đoán từng dòng, 14 cột như mọi thí nghiệm khác
    tom_tat.csv         một dòng mỗi tổ hợp
    meta.json           cấu hình, phiên bản torch, thiết bị

Năm phép kiểm trước khi chép vào `results/dl/`:

    N01  đủ ba phần trong zip
    N02  lược đồ dự đoán đúng 14 cột
    N03  phạm vi đúng bốn tổ hợp đã chốt, hạt giống 42, chế độ N1
    N04  mọi dòng dự đoán nằm trong tập dòng đánh giá chung, và giá trị thật khớp
         data/processed/
    N05  không ghi đè tệp đã có mà khác nội dung; tệp trùng tên được chuyển sang
         results/dl/cu/ kèm dấu thời gian

Chỉ khi cả năm đạt thì tệp mới được chép vào.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.quy_trinh.kiem import y_khop  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COT = ["nguon", "dich", "mode", "lich", "model", "loss", "h", "fold", "seed",
       "series_id", "bucket_phat", "bucket_muc_tieu", "y", "yhat"]
PHAM_VI_MONG = {("E1", 1), ("E1", 12), ("E2", 1), ("E2", 12)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dich", default="results/dl")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--eval-rows", default="data/eval_rows")
    ap.add_argument("--chi-kiem", action="store_true",
                    help="chỉ kiểm, không chép")
    a = ap.parse_args()

    z = Path(a.zip)
    if not z.exists():
        raise SystemExit(f"Không thấy {z}.")
    tam = ROOT / "results" / "colab" / "_giai_nen"
    if tam.exists():
        shutil.rmtree(tam)
    tam.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tam)

    loi = []
    du = sorted((tam / "du_doan").glob("*.parquet")) if (tam / "du_doan").exists() else []
    if not du or not (tam / "tom_tat.csv").exists():
        loi.append("N01: zip thiếu du_doan/ hoặc tom_tat.csv")
    print(f"N01  {len(du)} tệp dự đoán, "
          f"tom_tat.csv {'có' if (tam / 'tom_tat.csv').exists() else 'THIẾU'}, "
          f"meta.json {'có' if (tam / 'meta.json').exists() else 'thiếu'}")

    d = pd.concat([pd.read_parquet(p) for p in du], ignore_index=True) if du else pd.DataFrame()
    if d.empty:
        loi.append("N02: không đọc được dòng dự đoán nào")
    else:
        if list(d.columns) != COT:
            loi.append(f"N02: lược đồ lệch, có {list(d.columns)}")
        else:
            print(f"N02  lược đồ đúng 14 cột, {len(d):,} dòng")

    if not d.empty:
        co = {(r.dich, int(r.h)) for r in d[["dich", "h"]].drop_duplicates().itertuples()}
        seeds = set(d["seed"].unique())
        modes = set(d["mode"].unique())
        ok = co == PHAM_VI_MONG and seeds == {42} and modes == {"N1"}
        print(f"N03  tổ hợp {sorted(co)} · seed {sorted(seeds)} · chế độ {sorted(modes)}")
        if not ok:
            loi.append(f"N03: phạm vi lệch, mong {sorted(PHAM_VI_MONG)}, seed 42, N1")

    if not d.empty:
        ngoai, lech, y_dat, n_thieu_y = 0, 0.0, True, 0
        for (env, h), g in d.groupby(["dich", "h"]):
            er = pd.read_parquet(ROOT / a.eval_rows / f"{env}_h{h}.parquet")
            hop = set(zip(er.loc[er["bang"] & er["chuoi"], "series_id"],
                          er.loc[er["bang"] & er["chuoi"], "bucket"]))
            ngoai += sum(1 for s, t in zip(g["series_id"], g["bucket_phat"])
                         if (s, t) not in hop)
            y = pd.read_parquet(ROOT / a.processed / f"{env}.parquet",
                                columns=["series_id", "bucket", "y"]).set_index(
                ["series_id", "bucket"])["y"]
            k = pd.MultiIndex.from_arrays([g["series_id"], g["bucket_muc_tieu"]])
            that = y.reindex(k).to_numpy("float64")
            ok, l, thieu = y_khop(that, g["y"].to_numpy())
            lech, y_dat, n_thieu_y = max(lech, l), y_dat and ok, n_thieu_y + thieu
        print(f"N04  dòng ngoài tập chung: {ngoai} · lệch giá trị thật tối đa "
              f"{lech:.3e} (np.allclose rtol=1e-6, atol=1e-5)"
              + (f" · {n_thieu_y} dòng trỏ tới điểm không có quan sát" if n_thieu_y else ""))
        if ngoai or not y_dat:
            loi.append("N04: dòng ngoài tập chung hoặc giá trị thật lệch")

    if loi:
        print("\nKHÔNG NHẬP. Lỗi:")
        for x in loi:
            print("  -", x)
        return 1

    if a.chi_kiem:
        print("\nChỉ kiểm, không chép. Năm phép kiểm đều đạt.")
        return 0

    dich = ROOT / a.dich
    (dich / "du_doan").mkdir(parents=True, exist_ok=True)
    cu = dich / "cu" / time.strftime("%Y%m%d-%H%M%S")
    for p in du:
        q = dich / "du_doan" / p.name
        if q.exists() and q.read_bytes() != p.read_bytes():
            cu.mkdir(parents=True, exist_ok=True)
            shutil.move(str(q), cu / p.name)
        shutil.copy2(p, q)
    moi = pd.read_csv(tam / "tom_tat.csv")
    tt = dich / "tom_tat.csv"
    if tt.exists():
        moi = pd.concat([pd.read_csv(tt), moi], ignore_index=True)
        moi = moi.drop_duplicates(subset=["nguon", "dich", "h", "model", "seed"],
                                  keep="last")
    moi.to_csv(tt, index=False)
    if (tam / "meta.json").exists():
        shutil.copy2(tam / "meta.json", dich / "meta_colab.json")
    print(f"\nĐã nhập {len(du)} tệp vào {dich}")
    if cu.exists():
        print(f"Tệp cũ khác nội dung đã chuyển sang {cu}")
    print("Bước tiếp: python scripts/tong_hop.py --thi-nghiem tn_a,dl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
