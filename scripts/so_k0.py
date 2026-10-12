"""So bản có nội suy với bản không nội suy (K = 0).

    python scripts/so_k0.py

Đọc dự đoán từng dòng của hai lần chạy:

    results/tn_a/du_doan   bản chính, có nội suy tối đa hai điểm
    results/k0/du_doan     bản K = 0, không nội suy

và báo ba thứ, theo từng `(môi trường, chế độ, tầm, mô hình)`:

    do_bao_phu   bao nhiêu dòng chấm được ở mỗi bản, và bao nhiêu dòng chung.
                 `pct_k0_trong_chinh` = |chung| / |K = 0|, `pct_chinh_trong_k0` =
                 |chung| / |chính|. Tên cột là "X nằm trong Y" nên mẫu số là số dòng của X.
    chi_so_chung MAE, RMSE trung vị theo máy, tính **chỉ trên các thời điểm quan sát
                 thật mà cả hai bản cùng chấm**, tức giao theo khoá
                 `(series_id, bucket_phat)`
    chenh_lech   hiệu MAE giữa hai bản trên đúng tập chung đó

Hai bản khớp cùng mô hình với **cùng siêu tham số** (K = 0 chạy với
`--chosen-co-dinh`), nên chênh lệch còn lại chỉ đến từ dữ liệu: bản chính có thêm các
dòng mà cửa sổ đặc trưng chứa điểm nội suy, còn K = 0 thì loại chúng. Trên tập chung
thì các dòng đó không có mặt ở bản nào, nên phép so là so cùng thời điểm.

Không suy ra gì về việc mô hình thắng hay thua naive: bảng này chỉ trả lời "kết quả có
phụ thuộc vào việc nội suy hay không".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.evaluation.metrics import per_series_metrics  # noqa: E402
from cwp.quy_trinh.nap import chi_muc_tep, nap_ho  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KHOA = ["dich", "mode", "h", "model", "seed"]
# Cột thực sự cần: khoá của vòng lặp đã nằm trong tên tệp nên không phải nạp lại.
COT = ["series_id", "bucket_phat", "y", "yhat"]


def trung_vi(g: pd.DataFrame) -> tuple:
    ps = per_series_metrics(g["series_id"], g["y"], g["yhat"])
    return float(ps["mae"].median()), float(ps["rmse"].median()), int(len(ps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chinh", default="results/tn_a/du_doan")
    ap.add_argument("--k0", default="results/k0/du_doan")
    ap.add_argument("--out", default="results/so_k0")
    a = ap.parse_args()

    p1, p0 = ROOT / a.chinh, ROOT / a.k0
    for p in (p1, p0):
        if not p.exists():
            raise SystemExit(f"Thiếu {p}. Chạy khối K4 (bản chính) và K7.4 (K = 0) trước.")

    print(f"python: {sys.executable}")
    # Chỉ mục tệp thay cho việc nạp cả hai kho: cộng lại là 36,6 triệu dòng và hơn 2 GiB
    # riêng cho các cột chuỗi, nên nạp hết thì hết bộ nhớ. Mỗi tệp là đúng một tổ hợp
    # `KHOA`, nên mỗi vòng chỉ cần một cặp tệp.
    i0 = chi_muc_tep(p0)
    if i0.empty:
        raise SystemExit(f"Không có tệp dự đoán nào trong {p0}.")
    mo_hinh = sorted(i0["model"].unique())
    i1 = chi_muc_tep(p1)
    i1 = i1[i1["model"].isin(mo_hinh)]
    tep_chinh = {tuple(r[c] for c in KHOA): r["tep"] for _, r in i1.iterrows()}
    print(f"mô hình so sánh: {mo_hinh}")

    bao_phu, chung = [], []
    for khoa, gi0 in i0.groupby(KHOA):
        m = dict(zip(KHOA, khoa))
        if tuple(khoa) not in tep_chinh:
            continue
        g0 = nap_ho(gi0["tep"], cot=COT)
        g1 = nap_ho([tep_chinh[tuple(khoa)]], cot=COT)
        k1 = pd.MultiIndex.from_arrays([g1["series_id"], g1["bucket_phat"]])
        k0 = pd.MultiIndex.from_arrays([g0["series_id"], g0["bucket_phat"]])
        ch = k1.intersection(k0)
        g1c = g1[k1.isin(ch)]
        g0c = g0[k0.isin(ch)]
        mae1, rmse1, n1 = trung_vi(g1c)
        mae0, rmse0, n0 = trung_vi(g0c)
        mae1_tat_ca, _, _ = trung_vi(g1)
        mae0_tat_ca, _, _ = trung_vi(g0)
        bao_phu.append({**m, "n_dong_chinh": len(g1), "n_dong_k0": len(g0),
                        "n_dong_chung": len(ch),
                        # `g1` là bản chính, `g0` là K = 0 (xem `p1, p0 = chinh, k0`).
                        # Tỉ lệ "X nằm trong Y" lấy mẫu số là số dòng của X.
                        "pct_k0_trong_chinh": round(100 * len(ch) / len(g0), 3),
                        "pct_chinh_trong_k0": round(100 * len(ch) / len(g1), 3)})
        chung.append({**m, "n_dong_chung": len(ch), "n_chuoi_chung": n1,
                      "mae_chinh_chung": mae1, "mae_k0_chung": mae0,
                      "chenh_mae": mae0 - mae1,
                      "chenh_mae_pct": 100 * (mae0 - mae1) / mae1 if mae1 else np.nan,
                      "rmse_chinh_chung": rmse1, "rmse_k0_chung": rmse0,
                      "mae_chinh_tat_ca": mae1_tat_ca, "mae_k0_tat_ca": mae0_tat_ca})
        del g0, g1, g0c, g1c

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    bp = pd.DataFrame(bao_phu)
    ch = pd.DataFrame(chung)
    bp.to_csv(out / "do_bao_phu.csv", index=False)
    ch.to_csv(out / "chi_so_chung.csv", index=False)
    if ch.empty:
        print("Không có tổ hợp chung nào để so.")
        return 1

    print()
    print(f"{len(ch)} tổ hợp. Độ bao phủ: dòng của K = 0 nằm trong bản chính "
          f"{bp['pct_k0_trong_chinh'].min():.2f}–{bp['pct_k0_trong_chinh'].max():.2f}%; "
          f"dòng của bản chính nằm trong K = 0 "
          f"{bp['pct_chinh_trong_k0'].min():.2f}–{bp['pct_chinh_trong_k0'].max():.2f}%.")
    print(f"Chênh MAE trên tập chung (K = 0 trừ chính), theo %: "
          f"min {ch['chenh_mae_pct'].min():+.3f}, "
          f"trung vị {ch['chenh_mae_pct'].median():+.3f}, "
          f"max {ch['chenh_mae_pct'].max():+.3f}")
    print(f"Đã ghi {out / 'do_bao_phu.csv'} và {out / 'chi_so_chung.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
