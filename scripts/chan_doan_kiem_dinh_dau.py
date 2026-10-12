"""Chẩn đoán kiểm định dấu: kiểm định hạng có dấu và kiểm định dấu có cùng kết luận không.

    python scripts/chan_doan_kiem_dinh_dau.py
    python scripts/chan_doan_kiem_dinh_dau.py --thi-nghiem tn_b --out results/chan_doan

Chỉ đọc dự đoán và dữ liệu, chỉ ghi một tệp CSV mới vào `--out`. Không huấn luyện.

Kết luận chính của bài dùng kiểm định hạng có dấu Wilcoxon. Kiểm định này xếp hạng trị
tuyệt đối của hiệu MAE theo máy, nên một nhóm nhỏ máy thua đậm có thể kéo tổng hạng về
cân bằng dù phần lớn máy thắng. Script này đặt cạnh nó kiểm định dấu, chỉ đếm số máy
thắng và thua, trên ĐÚNG các cặp mà `tong_hop.py` đã kiểm định: họ so sánh, tập dòng
chung và MAE theo máy đều lấy từ chính `chi_muc_du_doan`, `dong_chung` và `mot_ho`.

Với mỗi (họ, mô hình), bảng gồm:

    n_chuoi            số máy ghép cặp được với naive
    n_thang, n_thua, n_hoa   số máy mô hình có MAE nhỏ hơn, lớn hơn, bằng naive
    hieu_p50           trung vị hiệu MAE (mô hình trừ naive), điểm phần trăm
    hang_mo_hinh_te_hon, hang_mo_hinh_tot_hon
                       tổng hạng của |hiệu| ở các máy mô hình tệ hơn / tốt hơn; máy hoà
                       bị bỏ, đúng như `scipy.stats.wilcoxon` mặc định
    tong_thua, tong_thang
                       tổng hiệu MAE ở các máy mô hình tệ hơn / độ lớn tổng ở máy tốt hơn
    te_nhat, may_te_nhat
                       hiệu MAE lớn nhất và máy tương ứng
    p_hang, p_hang_holm   p của kiểm định hạng có dấu, trước và sau Holm trong họ
    p_dau, p_dau_holm     p hai phía của kiểm định dấu, trước và sau Holm trong họ
    vuot_naive_hang, vuot_naive_dau
                       kết luận "vượt naive" theo từng kiểm định, cùng luật với bài:
                       p sau Holm < 0,05 và trung vị hiệu âm

`p_hang` được tính lại từ các cặp và so với `mae_p` của `mot_ho`; lệch thì script dừng,
vì khi đó bảng này không còn chấm trên cùng các cặp với kết quả chính.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tong_hop as th  # noqa: E402
from cwp.quy_trinh.nap import chi_muc_du_doan, nap_ho  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Trường hợp được nêu trong mục 4.6 của bài: cửa sổ đầu của E1L, h = 1, N1.
TIEU_DIEM = {"dich": "E1L", "mode": "N1", "h": 1, "fold": "o4032", "model": "xgb_mae"}


def mot_cap(c: pd.DataFrame) -> dict:
    """Hai kiểm định trên một bảng cặp `mae` (mô hình) và `mae_nv` (naive) theo máy."""
    a, b = c["mae"].to_numpy("float64"), c["mae_nv"].to_numpy("float64")
    hieu = a - b
    khac = hieu != 0
    hang = rankdata(np.abs(hieu[khac]))
    hk = hieu[khac]
    n_thang, n_thua = int((hieu < 0).sum()), int((hieu > 0).sum())
    i = int(np.argmax(hieu))
    return {
        "n_chuoi": len(c),
        "n_thang": n_thang,
        "n_thua": n_thua,
        "n_hoa": int((~khac).sum()),
        "hieu_p50": float(np.median(hieu)),
        "hang_mo_hinh_te_hon": float(hang[hk > 0].sum()),
        "hang_mo_hinh_tot_hon": float(hang[hk < 0].sum()),
        "tong_thua": float(hieu[hieu > 0].sum()),
        "tong_thang": float(-hieu[hieu < 0].sum()),
        "te_nhat": float(hieu[i]),
        "may_te_nhat": str(c.index[i]),
        "p_hang": 1.0 if np.allclose(hieu, 0) else float(wilcoxon(a, b).pvalue),
        "p_dau": (1.0 if n_thang + n_thua == 0
                  else float(binomtest(n_thang, n_thang + n_thua, 0.5).pvalue)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thi-nghiem", default="tn_b")
    ap.add_argument("--goc", default="results")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--eval-rows", default="data/eval_rows")
    ap.add_argument("--out", default="results/chan_doan")
    a = ap.parse_args()

    print(f"python: {sys.executable}")
    tn = a.thi_nghiem
    chi_muc = chi_muc_du_doan(ROOT / a.goc, [tn])
    if chi_muc.empty:
        raise SystemExit(f"Không có dự đoán ở {ROOT / a.goc / tn / 'du_doan'}.")
    proc, evr = ROOT / a.processed, ROOT / a.eval_rows
    moc = th.MOC_TRAIN_DAI if tn == "tn_b" else th.MOC_TRAIN_MAC_DINH
    d_mase, rng = {}, np.random.default_rng(42)

    dong = []
    for khoa, gi in chi_muc.groupby(th.KHOA_HO, sort=False):
        _, nguon, dich, mode, h, fold = khoa
        models = set(gi["model"])
        if nguon != dich or "naive" not in models:
            continue
        g = nap_ho(gi["tep"])
        khoa_chung, _ = th.dong_chung(dich, int(h), models, evr)
        if (dich, moc) not in d_mase:
            d_mase[(dich, moc)] = th.mau_so_mase(dich, proc, moc)
        ra = th.mot_ho(g, khoa_chung, d_mase[(dich, moc)], tn == "tn_b", rng)
        if ra is None:
            continue
        ps, _, kd, _ = ra
        nv = ps[ps["model"] == "naive"].set_index("series_id")[["mae"]]
        ho = []
        for (model, seed), x in ps.groupby(["model", "seed"]):
            if model == "naive":
                continue
            c = (x.set_index("series_id")[["mae"]]
                 .join(nv, rsuffix="_nv", how="inner").dropna())
            if len(c) < 10:
                continue
            r = {"thi_nghiem": tn, "dich": dich, "mode": mode, "h": int(h),
                 "fold": fold, "model": model, "seed": seed, **mot_cap(c)}
            goc_p = kd[(kd["model"] == model) & (kd["seed"] == seed)]["mae_p"]
            if len(goc_p) != 1 or not np.isclose(goc_p.iloc[0], r["p_hang"],
                                                  rtol=1e-12, atol=0):
                raise SystemExit(f"{khoa} {model}: p tính lại {r['p_hang']} khác "
                                 f"mae_p của tong_hop {goc_p.tolist()}.")
            ho.append(r)
        if not ho:
            continue
        ho = pd.DataFrame(ho)
        for p in ("p_hang", "p_dau"):
            ho[f"{p}_holm"] = th.holm(ho[p].to_numpy())
        for p in ("hang", "dau"):
            ho[f"vuot_naive_{p}"] = (ho[f"p_{p}_holm"] < th.ALPHA) & (ho["hieu_p50"] < 0)
        dong.append(ho)
        print(f"{dich} {mode} h={int(h):<2} {fold}: {len(ho)} mô hình")

    bang = pd.concat(dong, ignore_index=True)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tep = out / f"kiem_dinh_dau_{tn}.csv"
    bang.to_csv(tep, index=False)
    print(f"\nĐã ghi {tep}")

    lech = bang["vuot_naive_hang"] != bang["vuot_naive_dau"]
    print(f"Hai kiểm định khác kết luận ở {int(lech.sum())}/{len(bang)} (họ, mô hình).")

    td = bang
    for k, v in TIEU_DIEM.items():
        td = td[td[k] == v]
    if tn == "tn_b" and len(td) == 1:
        r = td.iloc[0]
        print(f"\nMục 4.6 — {TIEU_DIEM['dich']}, cửa sổ {TIEU_DIEM['fold']}, "
              f"h = {TIEU_DIEM['h']}, {TIEU_DIEM['mode']}, {TIEU_DIEM['model']}:")
        print(f"  máy                    {r['n_chuoi']}  (thắng {r['n_thang']}, "
              f"tỉ lệ {r['n_thang'] / r['n_chuoi']:.3f})")
        print(f"  trung vị hiệu MAE      {r['hieu_p50']:.3f}")
        print(f"  p hạng có dấu          {r['p_hang']:.4f}  (Holm {r['p_hang_holm']:.4f})")
        print(f"  tổng hạng tệ / tốt     {r['hang_mo_hinh_te_hon']:,.0f} / "
              f"{r['hang_mo_hinh_tot_hon']:,.0f}")
        print(f"  tổng thua / thắng      {r['tong_thua']:.1f} / {r['tong_thang']:.1f}")
        print(f"  máy tệ nhất            {r['te_nhat']:.2f}  ({r['may_te_nhat']})")
        print(f"  p kiểm định dấu        {r['p_dau']:.2e}  (Holm {r['p_dau_holm']:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
