"""Chẩn đoán tập dòng chung: các mô hình có được chấm trên đúng cùng một tập dòng không.

    python scripts/chan_doan_tap_dong.py
    python scripts/chan_doan_tap_dong.py --out results/chan_doan

Chỉ đọc dự đoán và dữ liệu, chỉ ghi một tệp CSV mới vào `--out`. Không huấn luyện.

Với mỗi họ so sánh `(môi trường, chế độ, tầm)` của Thí nghiệm A, bảng gồm:

    n_dong_du_doan     số dòng dự đoán của mô hình naive trong họ (mọi mô hình bảng cùng số)
    n_huu_han_<m>      số dòng có `yhat` hữu hạn của từng mô hình, TRƯỚC khi cắt
    n_giao_huu_han     số dòng mà MỌI mô hình cùng có `yhat` hữu hạn
    n_tap_chung_trong_du_doan  số dòng dự đoán nằm trong tập chung của `tong_hop.py`
    n_tap_chung_ma_tran        kích thước tập chung trên TOÀN ma trận đặc trưng (gồm cả
                       train và validation), chỉ để tham khảo
    n_dung_<m>         số dòng mà `per_series_metrics` thực sự dùng để tính MAE của
                       từng mô hình, lấy từ chính hàm `mot_ho` của `tong_hop.py`
    cung_tap           đúng khi mọi `n_dung_<m>` bằng nhau
    tap_chung_con_nan  số dòng của tập chung mà một mô hình nào đó có `yhat` không
                       hữu hạn; phải bằng 0 thì MAE mới tính trên cùng một tập

Kết luận cần đọc từ cột `cung_tap` và `tap_chung_con_nan`, không từ số dòng NaN thô.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tong_hop as th  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--goc", default="results")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--eval-rows", default="data/eval_rows")
    ap.add_argument("--out", default="results/chan_doan")
    ap.add_argument("--co-dl", action="store_true",
                    help="thêm DLinear vào họ N1, đúng như tong_hop.py khi gộp tn_a,dl")
    a = ap.parse_args()

    print(f"python: {sys.executable}")
    d = th.nap_du_doan(ROOT / a.goc / "tn_a", "tn_a")
    if a.co_dl and (ROOT / a.goc / "dl" / "du_doan").exists():
        x = th.nap_du_doan(ROOT / a.goc / "dl", "dl")
        x["thi_nghiem"] = "tn_a"
        d = pd.concat([d, x], ignore_index=True)
    d["thi_nghiem"] = "tn_a"
    d["bucket_phat"] = d["bucket_phat"].astype("int64")

    proc, evr = ROOT / a.processed, ROOT / a.eval_rows
    d_mase, rng = {}, np.random.default_rng(42)
    dong = []
    for khoa, g in d.groupby(["nguon", "dich", "mode", "h"], sort=True):
        nguon, dich, mode, h = khoa
        if nguon != dich:
            continue
        ten = sorted(set(g["model"]))
        khoa_chung, dk = th.dong_chung(dich, int(h), set(ten), evr)

        # trước khi cắt
        ky = pd.MultiIndex.from_arrays([g["series_id"], g["bucket_phat"]])
        trong = ky.isin(list(khoa_chung))
        r = {"env": dich, "mode": mode, "h": int(h), "n_dong_du_doan": int((g["model"] == "naive").sum())}
        huu = {}
        for (m, s), x in g.groupby(["model", "seed"]):
            nm = m if m != "dlinear" else f"dlinear_s{s}"
            huu[nm] = int(np.isfinite(x["yhat"]).sum())
            r[f"n_huu_han_{nm}"] = huu[nm]
        # giao hữu hạn: khoá có yhat hữu hạn ở MỌI (mô hình, seed)
        n_nhom = g.groupby(["model", "seed"]).ngroups
        dem = (g.assign(ok=np.isfinite(g["yhat"]) & np.isfinite(g["y"]))
                .groupby(["series_id", "bucket_phat"])["ok"].sum())
        r["n_giao_huu_han"] = int((dem == n_nhom).sum())   # có mặt VÀ hữu hạn ở mọi nhóm
        # `trong` đếm dòng của mọi (mô hình, seed), nên chia cho số nhóm để ra số dòng
        # của MỘT mô hình.
        r["n_tap_chung_trong_du_doan"] = int(trong.sum() // g.groupby(["model", "seed"]).ngroups)
        r["n_tap_chung_ma_tran"] = int(len(khoa_chung))

        # tập dòng THỰC SỰ dùng khi tính MAE: chạy đúng mot_ho của tong_hop
        if (dich, th.MOC_TRAIN_MAC_DINH) not in d_mase:
            d_mase[(dich, th.MOC_TRAIN_MAC_DINH)] = th.mau_so_mase(
                dich, proc, th.MOC_TRAIN_MAC_DINH)
        ps, gop, kd, phat = th.mot_ho(g, khoa_chung, d_mase[(dich, th.MOC_TRAIN_MAC_DINH)],
                                      False, rng)
        dung = {}
        for _, row in gop.iterrows():
            nm = row["model"] if row["model"] != "dlinear" else f"dlinear_s{row['seed']}"
            dung[nm] = int(ps[(ps["model"] == row["model"]) & (ps["seed"] == row["seed"])]["n_dong"].sum())
            r[f"n_dung_{nm}"] = dung[nm]
        r["cung_tap"] = bool(len(set(dung.values())) == 1)

        # trong tập chung, còn dòng nào mà một mô hình có yhat không hữu hạn
        gc = g[trong]
        r["tap_chung_con_nan"] = int((~np.isfinite(gc["yhat"])).sum()
                                     + (~np.isfinite(gc["y"])).sum())
        # seasonal: các dòng NaN nằm ngoài tập chung hết không
        s = g[(g["model"] == "seasonal")]
        nan_s = s[~np.isfinite(s["yhat"])]
        kn = pd.MultiIndex.from_arrays([nan_s["series_id"], nan_s["bucket_phat"]])
        r["seasonal_nan"] = int(len(nan_s))
        r["seasonal_nan_trong_tap_chung"] = int(kn.isin(list(khoa_chung)).sum())
        # dòng seasonal hữu hạn nhưng bị loại khỏi tập chung (chỉ vì nhãn nội suy hoặc ngày)
        sk = pd.MultiIndex.from_arrays([s["series_id"], s["bucket_phat"]])
        huu_s = np.isfinite(s["yhat"].to_numpy())
        r["seasonal_huu_han_ngoai_tap_chung"] = int((huu_s & ~sk.isin(list(khoa_chung))).sum())
        dong.append(r)
        print(f"{dich} {mode} h={int(h):<2} dự đoán {r['n_dong_du_doan']:>7,} · "
              f"trong tập chung {r['n_tap_chung_trong_du_doan']:>7,} · cùng tập: {r['cung_tap']} · "
              f"NaN còn trong tập chung: {r['tap_chung_con_nan']}")

    bang = pd.DataFrame(dong)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tep = out / ("seasonal_tap_dong_chung_co_dl.csv" if a.co_dl else "seasonal_tap_dong_chung.csv")
    bang.to_csv(tep, index=False)
    print(f"\nĐã ghi {tep}")
    print(f"Họ cùng tập: {int(bang['cung_tap'].sum())}/{len(bang)} · "
          f"tổng NaN còn trong tập chung: {int(bang['tap_chung_con_nan'].sum())}")
    return 0 if bool(bang["cung_tap"].all()) and int(bang["tap_chung_con_nan"].sum()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
