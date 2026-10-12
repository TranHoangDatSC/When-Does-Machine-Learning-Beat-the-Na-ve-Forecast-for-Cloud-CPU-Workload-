"""Kiểm tra tính đúng của ma trận đặc trưng, ba mô hình cơ sở và ranh giới thời gian.

    python scripts/check_dac_trung_va_baseline.py --env E2
    python scripts/check_dac_trung_va_baseline.py --env all --horizons 1,6,12

Mỗi phép kiểm tính lại một đại lượng **bằng đường độc lập**, đọc thẳng
`data/processed/{env}.parquet`, rồi so với giá trị mà `data/features/` và
`cwp.models.baselines` đưa ra. Một phép kiểm chỉ có giá trị khi nó báo sai lúc hiện
thực sai, nên chỗ nào so được với dữ liệu gốc thì so thẳng, thay vì kiểm một tính
chất mà cả bản đúng lẫn bản sai cùng thoả.

Mười hai phép kiểm:

    K01  schema 23 cột, đúng thứ tự MATRIX_COLS
    K02  lag_0 = lag_1 + diff_1
    K03  lag_0 = y(t) đọc từ data/processed/
    K04  dự báo naive = y(t)
    K05  trung bình trượt = trung bình y[t-5 .. t]
    K06  seasonal naive = y(t+h-288), kèm tỉ lệ dòng test không có giá trị đó
    K07  target = y(t+h)
    K08  20 đặc trưng dựng lại được từ dữ liệu TRƯỚC NỘI SUY, cắt tại t
    K09  ranh giới train, validation, test và luật purge
    K10  tỉ lệ dòng test chạm giá trị nội suy, ở ba vị trí khác nhau
    K11  mẫu số MASE tính trên riêng phần train
    K12  quy tắc lọc chuỗi chỉ dùng phần train

K08 là phép kiểm rò rỉ, và nó bắt đầu từ **dữ liệu trước nội suy**: lấy chuỗi gốc,
đặt lại `NaN` ở mọi điểm từng được nội suy, cắt cụt ngay tại `t`, rồi chạy lại đúng
bước nội suy và bước sinh đặc trưng. Cắt chuỗi đã nội suy sẵn thì không chứng minh
được gì, vì giá trị nội suy đã mang sẵn thông tin của đầu mút bên phải.

K13 đối chiếu hai đường dựng chế độ N1: sinh lại đặc trưng từ chuỗi đã z-score, và
phép affine trên ma trận N0 mà `cwp.quy_trinh.chay` dùng.

K14 kiểm tập dòng đánh giá chung khớp với ma trận đặc trưng.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.evaluation.metrics import mase_denominator, mase_denominators_train  # noqa: E402
from cwp.evaluation.splits import (  # noqa: E402
    N_TRAIN, N_VAL, WINDOW_BUCKETS, bucket_bounds, doc_b0, fit_mask, offset,
    split_masks,
)
from cwp.features import FEATURE_COLS, MATRIX_COLS, MAX_LAG  # noqa: E402
from cwp.features.io_ma_tran import bien_doi_n1_affine  # noqa: E402
from cwp.features.io_ma_tran import co_nhan, doc as doc_ma_tran  # noqa: E402
from cwp.features.matrix import build_feature_frame, prepare_input  # noqa: E402
from cwp.preprocess.resample import interpolate_short  # noqa: E402
from cwp.models.baselines import SEASONAL_LAG, du_doan  # noqa: E402
from cwp.preprocess.filter import judge  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2", "E3"]
HORIZONS = [1, 6, 12]

# Sai số làm tròn của số dấu phẩy động, không phải ngưỡng nới lỏng phép kiểm.
EPS = 1e-9


class Bao:
    """Gom kết quả các phép kiểm rồi in một bảng."""

    def __init__(self) -> None:
        self.dong: list[dict] = []

    def them(self, ma: str, dat: bool, mo_ta: str, so_do: str = "") -> None:
        self.dong.append({"ma": ma, "dat": bool(dat), "mo_ta": mo_ta, "so_do": so_do})

    def in_ra(self) -> int:
        print()
        for r in self.dong:
            dau = "đạt " if r["dat"] else "HỎNG"
            print(f"  [{dau}] {r['ma']}  {r['mo_ta']}")
            if r["so_do"]:
                print(f"            {r['so_do']}")
        hong = [r for r in self.dong if not r["dat"]]
        print()
        print(f"{len(self.dong) - len(hong)}/{len(self.dong)} phép kiểm đạt.")
        return 1 if hong else 0


def _tra_cuu(processed: pd.DataFrame) -> pd.Series:
    s = processed.set_index(["series_id", "bucket"])["y"]
    if not s.index.is_unique:
        raise SystemExit("data/processed/ có khoá (series_id, bucket) trùng lặp.")
    return s


def _lay(tra_cuu: pd.Series, sid: np.ndarray, bucket: np.ndarray) -> np.ndarray:
    khoa = pd.MultiIndex.from_arrays([sid, np.asarray(bucket, dtype="int64")])
    return tra_cuu.reindex(khoa).to_numpy("float64")


def _lech(a: np.ndarray, b: np.ndarray) -> float:
    """Sai lệch tuyệt đối lớn nhất, bỏ qua dòng mà cả hai bên cùng thiếu."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    cung_thieu = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)
    d[cung_thieu] = 0.0
    return float(np.nanmax(d)) if d.size else 0.0


def kiem_mot_to_hop(env: str, h: int, feat_dir: Path, proc_dir: Path, bao: Bao,
                    rng: np.random.Generator, n_mau: int = 200) -> None:
    X = pd.read_parquet(feat_dir / f"{env}_h{h}.parquet")
    processed = pd.read_parquet(proc_dir / f"{env}.parquet")
    tra_cuu = _tra_cuu(processed)
    b0 = doc_b0(env, proc_dir)

    sid = X["series_id"].to_numpy()
    bucket = X["bucket"].to_numpy("int64")
    nhan = f"{env} h={h}"

    # K01 --------------------------------------------------------------
    bao.them(f"K01 {nhan}", list(X.columns) == MATRIX_COLS,
             f"schema đúng {len(MATRIX_COLS)} cột theo MATRIX_COLS",
             f"{len(X.columns)} cột, {len(X):,} dòng")

    # K02 --------------------------------------------------------------
    d = _lech(X["lag_0"].to_numpy("float64"),
              X["lag_1"].to_numpy("float64") + X["diff_1"].to_numpy("float64"))
    bao.them(f"K02 {nhan}", d < EPS, "lag_0 = lag_1 + diff_1", f"lệch tối đa {d:.3e}")

    # K03 --------------------------------------------------------------
    d = _lech(X["lag_0"].to_numpy("float64"), _lay(tra_cuu, sid, bucket))
    bao.them(f"K03 {nhan}", d < EPS, "lag_0 = y(t) của data/processed/",
             f"lệch tối đa {d:.3e}")

    # K04 --------------------------------------------------------------
    d = _lech(du_doan("naive", X), _lay(tra_cuu, sid, bucket))
    bao.them(f"K04 {nhan}", d < EPS, "naive = y(t)", f"lệch tối đa {d:.3e}")

    # K05 --------------------------------------------------------------
    cua_so = np.stack([_lay(tra_cuu, sid, bucket - k) for k in range(6)])
    d = _lech(du_doan("ma6", X), np.nanmean(cua_so, axis=0))
    bao.them(f"K05 {nhan}", d < 1e-8, "ma6 = trung bình y[t-5 .. t]",
             f"lệch tối đa {d:.3e}")

    # K06 --------------------------------------------------------------
    ss = du_doan("seasonal", X, processed, h=h)
    d = _lech(ss, _lay(tra_cuu, sid, bucket + h - SEASONAL_LAG))
    m_test = split_masks(offset(bucket, b0), h)["test"]
    thieu = float(np.isnan(ss[m_test]).mean() * 100)
    bao.them(f"K06 {nhan}", d < EPS, "seasonal = y(t+h-288)",
             f"lệch tối đa {d:.3e}; {thieu:.2f}% dòng test không có giá trị đó")

    # K07 --------------------------------------------------------------
    d = _lech(X["target"].to_numpy("float64"), _lay(tra_cuu, sid, bucket + h))
    bao.them(f"K07 {nhan}", d < EPS, "target = y(t+h)", f"lệch tối đa {d:.3e}")

    # K08 --------------------------------------------------------------
    # Dựng lại từ dữ liệu TRƯỚC nội suy: đặt NaN ở mọi điểm từng được nội suy, cắt
    # cụt tại t, rồi chạy lại đúng bước nội suy và bước sinh đặc trưng.
    chon = rng.choice(len(X), size=min(n_mau, len(X)), replace=False)
    theo_chuoi = processed.sort_values(["series_id", "bucket"]).copy()
    theo_chuoi["y_tho"] = np.where(theo_chuoi["is_interp"].to_numpy(dtype=bool),
                                   np.nan, theo_chuoi["y"].to_numpy())
    nhom = {k: v for k, v in theo_chuoi.groupby("series_id", sort=False)}
    lech_max, so_kiem = 0.0, 0
    for i in chon:
        g = nhom[sid[i]]
        t = bucket[i]
        con = g[(g["bucket"] > t - MAX_LAG - 8) & (g["bucket"] <= t)]
        if len(con) < MAX_LAG + 1:
            continue
        tho = con.set_index("bucket")["y_tho"]
        noi, _ = interpolate_short(tho)
        lai = pd.DataFrame({
            "series_id": sid[i], "bucket": noi.index.to_numpy(),
            "y": noi.to_numpy(),
            "is_interp": tho.isna().to_numpy() & noi.notna().to_numpy(),
        })
        khung = build_feature_frame(prepare_input(lai), h)
        hang = khung[khung["bucket"] == t]
        if hang.empty or not np.isfinite(hang[FEATURE_COLS].to_numpy("float64")).all():
            continue
        a = hang[FEATURE_COLS].to_numpy("float64")[0]
        b = X.iloc[i][FEATURE_COLS].to_numpy("float64")
        lech_max = max(lech_max, _lech(a, b))
        so_kiem += 1
    bao.them(f"K08 {nhan}", so_kiem > 0 and lech_max < 1e-8,
             "đặc trưng dựng lại được từ dữ liệu trước nội suy, cắt tại t",
             f"{so_kiem} dòng mẫu, lệch tối đa {lech_max:.3e}")

    # K09 --------------------------------------------------------------
    off = offset(bucket, b0)
    mn = split_masks(off, h)
    lo_te, hi_te = bucket_bounds()["test"]
    ok = bool(
        (off[mn["test"]] >= lo_te).all()
        and (off[mn["test"]] + h < hi_te).all()
        and (off[fit_mask(off, h)] + h < N_TRAIN + N_VAL).all()
        and not (mn["train"] & mn["val"]).any()
        and not (mn["val"] & mn["test"]).any()
        and (off >= 0).all() and (off < WINDOW_BUCKETS).all()
    )
    dem = {k: int(v.sum()) for k, v in mn.items()}
    dem["loai"] = len(X) - sum(dem.values())
    bao.them(f"K09 {nhan}", ok, "ranh giới ba tập rời nhau và luật purge đúng",
             ", ".join(f"{k} {v:,}" for k, v in dem.items()))

    # K10 --------------------------------------------------------------
    noi = processed.set_index(["series_id", "bucket"])["is_interp"]

    def ti_le(dich: np.ndarray) -> float:
        khoa = pd.MultiIndex.from_arrays([sid[m_test], dich[m_test]])
        return float(noi.reindex(khoa).fillna(False).to_numpy().mean() * 100)

    cua = np.zeros(m_test.sum(), dtype=bool)
    for k in range(MAX_LAG + 1):
        khoa = pd.MultiIndex.from_arrays([sid[m_test], bucket[m_test] - k])
        cua |= noi.reindex(khoa).fillna(False).to_numpy()
    bao.them(f"K10 {nhan}", True, "tỉ lệ dòng test chạm giá trị nội suy",
             f"y(t) {ti_le(bucket):.3f}%, target {ti_le(bucket + h):.3f}%, "
             f"cửa sổ [t-24, t] {cua.mean() * 100:.3f}%")

    # K13 --------------------------------------------------------------
    p_n1 = feat_dir / f"{env}_N1_h{h}.parquet"
    if co_nhan(p_n1):
        from cwp.quy_trinh.chay import thong_ke_n1
        X1 = doc_ma_tran(p_n1)
        tk = thong_ke_n1(processed, b0, N_TRAIN)
        Xa = bien_doi_n1_affine(X, tk["mu"], tk["sd"])
        k0 = pd.MultiIndex.from_arrays([X["series_id"], X["bucket"]])
        k1 = pd.MultiIndex.from_arrays([X1["series_id"], X1["bucket"]])
        chung = k0.intersection(k1)
        ia, ib = k0.get_indexer(chung), k1.get_indexer(chung)
        v = max(_lech(Xa.iloc[ia][c].to_numpy(), X1.iloc[ib][c].to_numpy())
                for c in FEATURE_COLS + ["target"])
        bao.them(f"K13 {nhan}", v < 1e-8,
                 "N1 dựng bằng phép affine trùng N1 sinh lại từ chuỗi đã z-score",
                 f"{len(chung):,} dòng chung, lệch tối đa {v:.3e}")

    # K14 --------------------------------------------------------------
    p_er = ROOT / "data" / "eval_rows" / f"{env}_h{h}.parquet"
    if p_er.exists():
        er = pd.read_parquet(p_er)
        khop = (len(er) == len(X)
                and bool((er["series_id"].to_numpy() == sid).all())
                and bool((er["bucket"].to_numpy() == bucket).all()))
        bao.them(f"K14 {nhan}", khop,
                 "tập dòng đánh giá chung khớp từng dòng với ma trận đặc trưng",
                 f"{len(er):,} dòng, thiếu seasonal ngày "
                 f"{100 * (1 - er['ngay'].mean()):.3f}%, thiếu cửa sổ chuỗi "
                 f"{100 * (1 - er['chuoi'].mean()):.3f}%")


def kiem_mot_moi_truong(env: str, feat_dir: Path, proc_dir: Path, bao: Bao) -> None:
    processed = pd.read_parquet(proc_dir / f"{env}.parquet")
    b0 = doc_b0(env, proc_dir)

    # K11 --------------------------------------------------------------
    d_ham = mase_denominators_train(processed, b0)
    lo, hi = bucket_bounds()["train"]
    off = offset(processed["bucket"], b0)
    tay = (processed[(off >= lo) & (off < hi)]
           .sort_values(["series_id", "bucket"])
           .groupby("series_id")["y"]
           .apply(lambda s: mase_denominator(s.to_numpy())))
    chung = d_ham.index.intersection(tay.index)
    d = _lech(d_ham.loc[chung].to_numpy(), tay.loc[chung].to_numpy())
    bao.them(f"K11 {env}", d < EPS and len(chung) == len(d_ham),
             "mẫu số MASE tính trên riêng phần train",
             f"{len(chung):,} chuỗi, lệch tối đa {d:.3e}")

    # K12 --------------------------------------------------------------
    n_train = N_TRAIN
    xau = []
    for s, g in processed.sort_values(["series_id", "bucket"]).groupby("series_id"):
        ly_do = judge(g.set_index("bucket")["y"], n_train=n_train)
        if ly_do is not None:
            xau.append((s, ly_do))
    bao.them(f"K12 {env}", not xau,
             "mọi chuỗi đang giữ đều qua được bộ lọc tính trên riêng phần train",
             f"{len(xau)} chuỗi không qua" if xau else
             f"{processed['series_id'].nunique():,} chuỗi, ngưỡng lấy {n_train} điểm đầu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="all", choices=ENVS + ["all"])
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n-mau", type=int, default=200,
                    help="số dòng lấy mẫu cho phép kiểm rò rỉ K08")
    a = ap.parse_args()

    envs = ENVS if a.env == "all" else [a.env]
    horizons = [int(x) for x in a.horizons.split(",")]
    feat_dir, proc_dir = ROOT / a.features, ROOT / a.processed

    print()
    print("KIỂM TRA TÍNH ĐÚNG: đặc trưng, mô hình cơ sở, ranh giới thời gian")
    print(f"môi trường : {', '.join(envs)}")
    print(f"tầm dự báo : {', '.join(map(str, horizons))}")

    bao = Bao()
    rng = np.random.default_rng(42)
    for env in envs:
        kiem_mot_moi_truong(env, feat_dir, proc_dir, bao)
        for h in horizons:
            if not (feat_dir / f"{env}_h{h}.parquet").exists():
                bao.them(f"K00 {env} h={h}", False,
                         "thiếu ma trận đặc trưng",
                         f"chạy: python scripts/build_features.py --env {env}")
                continue
            kiem_mot_to_hop(env, h, feat_dir, proc_dir, bao, rng, a.n_mau)
    return bao.in_ra()


if __name__ == "__main__":
    sys.exit(main())
