"""Tổng hợp kết quả: chỉ số, kiểm định, chuyển giao và chi phí bất đối xứng.

    python scripts/tong_hop.py --thi-nghiem tn_a
    python scripts/tong_hop.py --thi-nghiem tn_a,tn_c,may_gia,tn_b
    python scripts/tong_hop.py --thi-nghiem tn_a --kiem-day-du

Đọc **dự đoán từng dòng** ở `results/*/du_doan/`, cắt về tập dòng đánh giá chung
rồi mới tính. Không mô hình nào được chấm trên tập dòng riêng của nó.

Tập dòng chung theo từng họ so sánh
-----------------------------------

Một họ là các mô hình cùng `(thí nghiệm, nguồn, đích, chế độ, tầm, fold)`. Điều kiện
lấy giao tuỳ theo những mô hình có mặt trong họ:

    luôn luôn                       cột `bang`
    có seasonal theo ngày           thêm cột `ngay`
    có seasonal theo tuần           thêm cột `tuan`
    có dlinear hoặc gru             thêm cột `chuoi`

Số dòng bị loại vì mỗi điều kiện ghi vào `tom_tat_dong.csv`.

Năm bảng đầu ra, dưới `results/tong_hop/`

    chi_so_theo_chuoi.csv   MAE, RMSE, MASE, SMAPE, R² của từng máy
    chi_so_gop.csv          trung vị và tứ phân vị trên tập máy, kèm `<chỉ số>_n_dung`
                            và `<chỉ số>_n_loai`: số máy thực sự vào phân vị và số máy bị
                            loại vì chỉ số đó không xác định (protocol mục 9)
    kiem_dinh.csv           Wilcoxon ghép cặp với naive, hiệu chỉnh Holm trong họ
    chuyen_giao.csv         hai tỉ số bắt buộc của Thí nghiệm C
    chi_phi_bat_doi_xung.csv  chi phí theo bốn tỉ lệ phạt thiếu so với dư
    tom_tat_dong.csv        số dòng chung và số dòng bị loại theo từng điều kiện

Kiểm định
---------

Wilcoxon ghép cặp theo từng máy, so **mỗi mô hình với naive** trong cùng một họ, hiệu
chỉnh Holm trên đúng họ đó. Hướng lấy từ trung vị hiệu ghép cặp. Kèm tỉ lệ máy thắng
và khoảng tin cậy bootstrap 95% của tỉ số trung vị, lấy mẫu lại **theo máy**; với
Thí nghiệm B lấy mẫu lại **theo khối ngày** để tính đến phụ thuộc giữa các ngày.

Chi phí bất đối xứng
--------------------

    chi phí = a · max(0, y − ŷ) + b · max(0, ŷ − y),  a : b ∈ {1:1, 2:1, 5:1, 10:1}

Đây là **phân tích theo kịch bản giả định**, không phải chi phí tài chính và không
phải mức giảm vi phạm cam kết chất lượng dịch vụ. Bốn tỉ lệ chốt trước khi chạy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.evaluation.metrics import METRIC_NAMES, per_series_metrics  # noqa: E402
from cwp.evaluation.metrics import mase_denominator  # noqa: E402
from cwp.evaluation.splits import KHUNG_8N, offset  # noqa: E402
from cwp.quy_trinh.nap import chi_muc_du_doan, nap_ho  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Thư mục của từng thí nghiệm nằm dưới một gốc chung, đổi bằng --goc.
TEN_THI_NGHIEM = ("tn_a", "tn_b", "tn_c", "may_gia", "dl")
# Tỉ lệ phạt thiếu so với phạt dư, chốt trước.
TI_LE_PHAT = [1, 2, 5, 10]
MOC_TRAIN_MAC_DINH = KHUNG_8N.n_train          # 1612
MOC_TRAIN_DAI = 14 * 288                        # Thí nghiệm B: 14 ngày
ALPHA = 0.05
N_BOOT = 1000
MO_HINH_CHUOI = ("dlinear", "gru")
MO_HINH_NGAY = ("seasonal",)
MO_HINH_TUAN = ("seasonal_tuan",)
KHOA_HO = ["thi_nghiem", "nguon", "dich", "mode", "h", "fold"]


def holm(p: np.ndarray) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni, trả p đã hiệu chỉnh, cắt tại 1."""
    p = np.asarray(p, dtype="float64")
    n = len(p)
    thu_tu = np.argsort(p)
    ra = np.empty(n)
    chay = 0.0
    for i, k in enumerate(thu_tu):
        chay = max(chay, (n - i) * p[k])
        ra[k] = min(chay, 1.0)
    return ra


def nap_du_doan(thu_muc: Path, thi_nghiem: str) -> pd.DataFrame:
    p = thu_muc / "du_doan"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    if "thi_nghiem" not in d.columns:
        d = d.assign(thi_nghiem=thi_nghiem)
    return d


def dong_chung(dich: str, h: int, models, eval_dir: Path) -> tuple:
    """Khoá `(series_id, bucket)` được phép chấm, và số dòng bị loại theo điều kiện."""
    p = eval_dir / f"{dich}_h{h}.parquet"
    if not p.exists():
        raise SystemExit(
            f"Thiếu {p}. Chạy scripts/tap_dong_chung.py trước."
        )
    er = pd.read_parquet(p)
    m = er["bang"].to_numpy(dtype=bool)
    dieu_kien = {"bang": int(m.sum())}
    if any(x in models for x in MO_HINH_NGAY):
        m = m & er["ngay"].to_numpy(dtype=bool)
        dieu_kien["ngay"] = int(m.sum())
    if any(x in models for x in MO_HINH_TUAN):
        m = m & er["tuan"].to_numpy(dtype=bool)
        dieu_kien["tuan"] = int(m.sum())
    if any(x in models for x in MO_HINH_CHUOI):
        m = m & er["chuoi"].to_numpy(dtype=bool)
        dieu_kien["chuoi"] = int(m.sum())
    khoa = set(zip(er.loc[m, "series_id"], er.loc[m, "bucket"]))
    return khoa, dieu_kien


def mau_so_mase(env: str, proc_dir: Path, moc_train: int) -> pd.Series:
    """Sai số naive một bước trên phần train của từng máy."""
    p = pd.read_parquet(proc_dir / f"{env}.parquet",
                        columns=["series_id", "bucket", "y"])
    b0 = int(p["bucket"].min())
    off = offset(p["bucket"], b0)
    con = p.loc[(off >= 0) & (off < moc_train)].sort_values(["series_id", "bucket"])
    return con.groupby("series_id")["y"].apply(lambda s: mase_denominator(s.to_numpy()))


def chi_phi_bat_doi_xung(y: np.ndarray, yhat: np.ndarray, a: float) -> float:
    """Trung bình của `a · thiếu + 1 · dư`, với thiếu là `y > ŷ`."""
    e = np.asarray(y, dtype="float64") - np.asarray(yhat, dtype="float64")
    return float(np.mean(a * np.clip(e, 0, None) + np.clip(-e, 0, None)))


def ti_so_theo_may(a, b) -> tuple:
    """Tỉ số `a / b` theo từng máy, chỉ trên máy mà tỉ số xác định (protocol mục 9).

    Máy có `b` (chỉ số của naive) bằng 0 thì tỉ số là 0/0 hoặc vô hạn: bị loại khỏi phần gộp
    của chỉ số này và được đếm riêng, không thay bằng 0 hay vô hạn. Trả `(tỉ số, số máy bị loại)`.
    """
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    return a[ok] / b[ok], int((~ok).sum())


def boot_ti_so(ti_so: np.ndarray, nhom: "np.ndarray | None", rng) -> tuple:
    """Khoảng tin cậy 95% của trung vị tỉ số, lấy mẫu lại theo máy hoặc theo khối."""
    v = np.asarray(ti_so, dtype="float64")
    ok = np.isfinite(v)
    v = v[ok]
    if v.size < 5:
        return float("nan"), float("nan")
    if nhom is None:
        mau = rng.choice(v, size=(N_BOOT, v.size), replace=True)
        tv = np.median(mau, axis=1)
    else:
        g = np.asarray(nhom)[ok]
        khoi = [v[g == k] for k in np.unique(g)]
        tv = np.empty(N_BOOT)
        for i in range(N_BOOT):
            lay = rng.integers(0, len(khoi), len(khoi))
            tv[i] = np.median(np.concatenate([khoi[j] for j in lay]))
    return float(np.percentile(tv, 2.5)), float(np.percentile(tv, 97.5))


def mot_ho(g: pd.DataFrame, khoa_chung: set, d_mase: pd.Series,
           theo_ngay: bool, rng) -> tuple:
    """Chỉ số theo chuỗi, bảng gộp, kiểm định và chi phí cho một họ so sánh.

    CHÚ Ý — sai lệch D2 (protocol v1.2, Phụ lục B). Tham số `theo_ngay` **không được
    dùng**: `boot_ti_so` dưới đây luôn được gọi với `nhom=None`, nên khoảng tin cậy
    bootstrap luôn lấy mẫu lại **theo máy**, kể cả với Thí nghiệm B mà mục 9 của protocol
    quy định lấy mẫu lại theo khối ngày. Quyết định ngày 27/09/2026 là giữ nguyên cách
    này và ghi nhận sai lệch, vì đoạn chấm của mỗi fold chỉ dài 7 ngày (fold cuối 1 đến 2
    ngày) nên lấy mẫu lại theo khối không đủ khối để có nghĩa. Tham số được giữ lại để
    chữ ký hàm khớp bộ gọi và để chỗ này còn dấu vết; đừng đọc nó như một nhánh đang chạy.
    """
    g = g[[(s, b) in khoa_chung
           for s, b in zip(g["series_id"], g["bucket_phat"])]]
    if g.empty:
        return None

    ps_all, gop, phat = [], [], []
    for (model, seed), x in g.groupby(["model", "seed"]):
        ps = per_series_metrics(x["series_id"], x["y"], x["yhat"], d=d_mase)
        ps.insert(0, "seed", seed)
        ps.insert(0, "model", model)
        ps_all.append(ps)
        r = {"model": model, "seed": seed, "n_chuoi": len(ps),
             "n_dong": int(len(x))}
        for m in METRIC_NAMES:
            v = pd.to_numeric(ps[m], errors="coerce")
            # Protocol mục 9: chuỗi có chỉ số không xác định bị loại khỏi phần gộp của
            # CHỈ SỐ ĐÓ và phải được đếm riêng. `n_chuoi` là tổng số chuỗi của họ, nên
            # nếu không có hai cột này thì phân vị của R² trông như tính trên `n_chuoi`.
            huu = np.isfinite(v)
            v = v[huu]
            r[f"{m}_n_dung"] = int(huu.sum())
            r[f"{m}_n_loai"] = int((~huu).sum())
            r[f"{m}_p25"] = float(np.percentile(v, 25)) if len(v) else np.nan
            r[f"{m}_p50"] = float(np.median(v)) if len(v) else np.nan
            r[f"{m}_p75"] = float(np.percentile(v, 75)) if len(v) else np.nan
        gop.append(r)
        for a in TI_LE_PHAT:
            cp = (x.assign(c=lambda z: a * np.clip(z["y"] - z["yhat"], 0, None)
                           + np.clip(z["yhat"] - z["y"], 0, None))
                  .groupby("series_id")["c"].mean())
            phat.append({"model": model, "seed": seed, "ti_le_phat": a,
                         "chi_phi_p50": float(cp.median()),
                         "chi_phi_tb": float(cp.mean())})

    ps_all = pd.concat(ps_all, ignore_index=True)
    gop = pd.DataFrame(gop)
    phat = pd.DataFrame(phat)

    # Kiểm định: mỗi mô hình so với naive, ghép cặp theo máy.
    kd = []
    if "naive" in set(ps_all["model"]):
        nv = ps_all[ps_all["model"] == "naive"].set_index("series_id")
        for (model, seed), x in ps_all.groupby(["model", "seed"]):
            if model == "naive":
                continue
            c = x.set_index("series_id")[["mae", "rmse", "mase"]].join(
                nv[["mae", "rmse", "mase"]], rsuffix="_nv", how="inner").dropna()
            if len(c) < 10:
                continue
            r = {"model": model, "seed": seed, "n_chuoi": len(c)}
            for m in ("mae", "rmse", "mase"):
                a_, b_ = c[m].to_numpy(), c[f"{m}_nv"].to_numpy()
                hieu = a_ - b_
                r[f"{m}_p"] = (1.0 if np.allclose(hieu, 0)
                               else float(wilcoxon(a_, b_).pvalue))
                r[f"{m}_hieu_p50"] = float(np.median(hieu))
                ti, n_loai = ti_so_theo_may(a_, b_)
                r[f"{m}_ti_so_p50"] = float(np.median(ti)) if len(ti) else float("nan")
                r[f"{m}_n_ti_so_loai"] = n_loai
                r[f"{m}_ti_le_thang"] = float(np.mean(a_ < b_))
                if m == "mae":
                    # `None`: lấy mẫu lại theo máy. Xem sai lệch D2 ở docstring.
                    lo, hi = boot_ti_so(ti, None, rng)
                    r["mae_ti_so_ci_lo"], r["mae_ti_so_ci_hi"] = lo, hi
            kd.append(r)
    kd = pd.DataFrame(kd)
    if not kd.empty:
        for m in ("mae", "rmse", "mase"):
            kd[f"{m}_p_holm"] = holm(kd[f"{m}_p"].to_numpy())
            kd[f"{m}_vuot_naive"] = ((kd[f"{m}_p_holm"] < ALPHA)
                                     & (kd[f"{m}_hieu_p50"] < 0))
    return ps_all, gop, kd, phat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thi-nghiem", default="tn_a")
    ap.add_argument("--goc", default="results",
                    help="thư mục gốc chứa tn_a/, tn_b/, tn_c/, may_gia/, dl/")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--eval-rows", default="data/eval_rows")
    ap.add_argument("--out", default="results/tong_hop")
    ap.add_argument("--kiem-day-du", action="store_true",
                    help="dừng nếu một họ thiếu mô hình naive")
    a = ap.parse_args()

    tns = [x.strip() for x in a.thi_nghiem.split(",")]
    proc_dir, eval_dir = ROOT / a.processed, ROOT / a.eval_rows
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print()
    print("TỔNG HỢP KẾT QUẢ")
    print(f"python     : {sys.executable}")
    print(f"thí nghiệm : {tns}")

    # Chỉ mục tệp thay cho việc nạp cả kho: Thí nghiệm B có ~162 triệu dòng, nạp hết
    # thì hết bộ nhớ. Mỗi tệp thuộc trọn một họ, nên nạp từng họ rồi giải phóng.
    # Thứ tự duyệt họ giữ nguyên như bản cũ (thứ tự tên tệp đã sắp), vì bootstrap
    # dùng chung một bộ sinh số ngẫu nhiên cho mọi họ.
    chi_muc = chi_muc_du_doan(ROOT / a.goc, tns)
    if chi_muc.empty:
        raise SystemExit("Không có dự đoán nào để tổng hợp.")
    for tn in tns:
        x = chi_muc[chi_muc["thu_muc"] == tn]
        if x.empty:
            print(f"  {tn}: chưa có dự đoán ở {ROOT / a.goc / tn / 'du_doan'}")
            continue
        print(f"  {tn}: {int(x['n_dong'].sum()):,} dòng dự đoán, "
              f"{x['model'].nunique()} mô hình")
    d_mase = {}
    ps_all, gop_all, kd_all, phat_all, dong_all = [], [], [], [], []
    n_cg_khong_naive = 0
    t0 = time.time()
    for khoa, gi in chi_muc.groupby(KHOA_HO, sort=False):
        tn, nguon, dich, mode, h, fold = khoa
        models = set(gi["model"])
        g = nap_ho(gi["tep"])
        if "naive" not in models:
            # Họ **chuyển giao** không có naive là đúng thiết kế: dự đoán naive không phụ
            # thuộc mô hình nguồn, nên `chuyen_giao.csv` lấy "naive tại đích" từ chiều tại
            # chỗ của cùng đích (protocol mục 9, hai tỉ số). Chỉ họ chấm **tại chỗ** mà
            # thiếu naive mới là lỗi. Phép kiểm `R02b` của `kiem_ket_qua.py` lo phần
            # tham chiếu của họ chuyển giao.
            if nguon == dich:
                msg = f"Họ {khoa} không có naive nên không kiểm định được."
                if a.kiem_day_du:
                    raise SystemExit(msg)
                print("  CẢNH BÁO:", msg)
            else:
                n_cg_khong_naive += 1
        khoa_chung, dieu_kien = dong_chung(dich, int(h), models, eval_dir)
        moc = MOC_TRAIN_DAI if tn == "tn_b" else MOC_TRAIN_MAC_DINH
        if (dich, moc) not in d_mase:
            d_mase[(dich, moc)] = mau_so_mase(dich, proc_dir, moc)
        ra = mot_ho(g, khoa_chung, d_mase[(dich, moc)], tn == "tn_b", rng)
        if ra is None:
            continue
        ps, gop, kd, phat = ra
        cot = dict(zip(KHOA_HO, khoa))
        for df, kho in ((ps, ps_all), (gop, gop_all), (kd, kd_all),
                        (phat, phat_all)):
            if df is not None and not df.empty:
                for k, v in cot.items():
                    df.insert(0, k, v)
                kho.append(df)
        dong_all.append(cot | dieu_kien | {
            "n_dong_du_doan": int(len(g)),
            "n_dong_cham": int(gop["n_dong"].max()) if not gop.empty else 0,
        })
        del g

    def ghi(ten, khung):
        if khung:
            df = pd.concat(khung, ignore_index=True)
            df.to_csv(out / ten, index=False)
            print(f"  {ten:28} {len(df):>8,} dòng")
            return df
        return pd.DataFrame()

    print()
    if n_cg_khong_naive:
        print(f"{n_cg_khong_naive} họ chuyển giao không có naive (đúng thiết kế): tỉ số so "
              f"với naive tại đích lấy từ chiều tại chỗ của cùng đích, xem chuyen_giao.csv "
              f"và phép kiểm R02b.")
    print("Đã ghi:")
    ghi("chi_so_theo_chuoi.csv", ps_all)
    gop_df = ghi("chi_so_gop.csv", gop_all)
    kd_df = ghi("kiem_dinh.csv", kd_all)
    ghi("chi_phi_bat_doi_xung.csv", phat_all)
    ghi("tom_tat_dong.csv", [pd.DataFrame(dong_all)])

    # Chuyển giao: ghép tỉ số với mô hình học tại đích và với naive tại đích.
    if not gop_df.empty and {"tn_c", "may_gia"} & set(gop_df["thi_nghiem"]):
        tai_cho = gop_df[gop_df["nguon"] == gop_df["dich"]]
        chuyen = gop_df[gop_df["nguon"] != gop_df["dich"]]
        k = ["dich", "mode", "h", "fold", "model", "seed"]
        nv = (gop_df[(gop_df["model"] == "naive")
                     & (gop_df["nguon"] == gop_df["dich"])]
              [["dich", "mode", "h", "fold", "mae_p50"]]
              .drop_duplicates()
              .rename(columns={"mae_p50": "mae_p50_naive_dich"}))
        m = chuyen.merge(tai_cho[k + ["mae_p50"]], on=k, how="left",
                         suffixes=("", "_tai_cho"))
        m = m.merge(nv, on=["dich", "mode", "h", "fold"], how="left")
        # Hai tỉ số bắt buộc báo cùng nhau: so với mô hình học tại đích, và so với
        # dự báo naive tại đích. Thiếu một trong hai thì cột đó là NaN và bước kiểm
        # kết quả sẽ báo ra, chứ không lặng lẽ bỏ.
        m["L"] = m["mae_p50"] / m["mae_p50_tai_cho"]
        m["so_voi_naive_tai_dich"] = m["mae_p50"] / m["mae_p50_naive_dich"]
        m.to_csv(out / "chuyen_giao.csv", index=False)
        print(f"  {'chuyen_giao.csv':28} {len(m):>8,} dòng")

    (out / "meta.json").write_text(json.dumps({
        "ngay": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv[1:], "python": sys.executable,
        "thi_nghiem": tns, "ti_le_phat": TI_LE_PHAT, "alpha": ALPHA,
        "n_boot": N_BOOT,
        "ho_kiem_dinh": "mỗi mô hình so với naive trong cùng "
                        "(thí nghiệm, nguồn, đích, chế độ, tầm, fold)",
        "giay": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nXong sau {time.time() - t0:.0f}s. Bảng ở {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
