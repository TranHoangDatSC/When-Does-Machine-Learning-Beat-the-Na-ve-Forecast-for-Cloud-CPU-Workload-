"""Thí nghiệm A: yếu tố nào quyết định kết luận so với dự báo naive.

    python scripts/tn_a.py --env E2 --horizons 1 --modes N0 --models naive,xgb
    python scripts/tn_a.py --tat-ca                    # phạm vi đã chốt
    python scripts/tn_a.py --tat-ca --resume           # chạy tiếp sau gián đoạn

Phạm vi đã chốt: ba môi trường E1, E2, E3 × ba tầm dự báo 1, 6, 12 × hai chế độ chuẩn
hoá N0 và N1 × các mô hình `naive`, `ma6`, `seasonal`, `ridge`, `xgb`, `xgb_mae`,
`svr`. Lưới XGBoost là phương án B: hai ứng viên `max_depth` 4 và 8, cùng 300 cây, áp
giống nhau cho cả hai hàm mất mát.

Ranh giới thời gian là khung 8 ngày: train `[0, 1612)`, validation `[1612, 1957)`,
test `[1957, 2304)`. Siêu tham số chọn bằng rolling-origin 5 fold **trong vùng
validation**, điểm chọn là trung vị MAE theo chuỗi trên thang CPU%. Mô hình cuối khớp
trên train + validation rồi chấm test **đúng một lần**.

Đầu ra, dưới `results/tn_a/`:

    tom_tat.csv     một dòng mỗi tổ hợp đã chạy, kèm vân tay và thời gian
    cv_search.csv   điểm validation của từng ứng viên, từng fold
    chosen.csv      siêu tham số đã chốt của từng (env, mode, h, model)
    du_doan/        dự đoán từng dòng, 14 cột, xem cwp.evaluation.luu_du_doan
    meta.json       cấu hình, phiên bản thư viện, số lần chấm test, vân tay

Chỉ số và kiểm định **không** tính ở đây. `scripts/tong_hop.py` đọc `du_doan/`,
cắt về tập dòng đánh giá chung rồi mới tính. Nhờ vậy thêm một chỉ số về sau không
phải huấn luyện lại.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.evaluation.splits import KHUNG_8N, fit_mask, mask_khoang  # noqa: E402
from cwp.evaluation.tang import bang_tang, doc_cau_hinh_tang  # noqa: E402
from cwp.models.baselines import BASELINE_NAMES  # noqa: E402
from cwp.models.registry import REGISTRY, SEED, mau_con_phan_tang  # noqa: E402
from cwp.quy_trinh import van_tay as vt  # noqa: E402
from cwp.quy_trinh.chay import chon_sieu_tham_so, chuan_bi, khop_va_cham  # noqa: E402
from cwp.quy_trinh.kiem import KHOA_CV  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2", "E3"]
HORIZONS = [1, 6, 12]
MODES = ["N0", "N1"]
MODELS = ["naive", "ma6", "seasonal", "ridge", "xgb", "xgb_mae", "svr"]
KHOA = ["thi_nghiem", "nguon", "dich", "mode", "h", "model", "seed", "fold"]


def _phien_ban() -> dict:
    v = {"python": platform.python_version()}
    for t in ("pandas", "numpy", "pyarrow", "scipy", "sklearn", "xgboost"):
        try:
            v[t] = __import__(t).__version__
        except Exception:
            v[t] = "?"
    return v


def khoa_mau_con_svr(d, tang_env, n: int) -> np.ndarray:
    """Mặt nạ mẫu con của SVR, phân tầng theo ba tầng độ giật của phần train.

    Rút **một lần cho mỗi (env, mode)** trên vùng train + validation của `h = 1` rồi
    dùng lại cho mọi tầm dự báo, để ba tầm học trên cùng một mẫu.
    """
    vi_tri = np.flatnonzero(fit_mask(d.off, d.h, KHUNG_8N))
    tang = tang_env.reindex(d.sid[vi_tri]).to_numpy()
    chon = mau_con_phan_tang(tang, n, seed=SEED)
    m = np.zeros(len(d.sid), dtype=bool)
    m[vi_tri[chon]] = True
    return m


def da_chay(tom_tat: Path, van_tay_hien: dict) -> set:
    """Các tổ hợp đã có kết quả **và** vân tay còn khớp."""
    if not tom_tat.exists():
        return set()
    d = pd.read_csv(tom_tat)
    if d.empty or "van_tay_ma" not in d.columns:
        return set()
    giu = (
        (d["van_tay_ma"] == van_tay_hien["ma"])
        & (d["van_tay_cau_hinh"] == van_tay_hien["cau_hinh"])
        & (d["van_tay_du_lieu"] == van_tay_hien["du_lieu"])
    )
    return {tuple(r) for r in d.loc[giu, KHOA].astype(str).to_numpy()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default=",".join(ENVS))
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results/tn_a")
    ap.add_argument("--chosen-co-dinh", default=None,
                    help="dùng siêu tham số của tệp này, KHÔNG dò lại; dùng cho K = 0")
    ap.add_argument("--tat-ca", action="store_true",
                    help="phạm vi đã chốt, bỏ qua các tham số phạm vi khác")
    ap.add_argument("--resume", action="store_true",
                    help="bỏ qua tổ hợp đã chạy và còn khớp vân tay")
    a = ap.parse_args()

    envs = ENVS if a.tat_ca else [x.strip() for x in a.env.split(",")]
    hs = HORIZONS if a.tat_ca else [int(x) for x in a.horizons.split(",")]
    modes = MODES if a.tat_ca else [x.strip().upper() for x in a.modes.split(",")]
    models = MODELS if a.tat_ca else [x.strip() for x in a.models.split(",")]

    feat_dir, proc_dir = ROOT / a.features, ROOT / a.processed
    out = ROOT / a.out
    (out / "du_doan").mkdir(parents=True, exist_ok=True)

    du_lieu_tep = [proc_dir / f"{e}.parquet" for e in envs]
    du_lieu_tep += [feat_dir / f"{e}_h{h}.parquet" for e in envs for h in hs]
    dau_van = vt.van_tay(ROOT, du_lieu_tep)
    bo_qua = da_chay(out / "tom_tat.csv", dau_van) if a.resume else set()

    co_dinh = None
    if a.chosen_co_dinh:
        pc = ROOT / a.chosen_co_dinh
        if not pc.exists():
            raise SystemExit(f"Thiếu {pc}. Chạy Thí nghiệm A (khối K4) trước.")
        co_dinh = {(r.env, r.mode, int(r.h), r.model): json.loads(r.sieu_tham_so)
                   for r in pd.read_csv(pc).itertuples()}
    cv_bang = pd.read_csv(ROOT / "results" / "tables" / "cv.csv")
    st = doc_cau_hinh_tang(ROOT / "config" / "split.yaml")
    tang_tat = bang_tang(cv_bang, envs=tuple(envs), n_strata=int(st["n_strata"]))

    print()
    print("THÍ NGHIỆM A — yếu tố quyết định kết luận")
    print(f"python   : {sys.executable}")
    print(f"môi trường {envs} · tầm {hs} · chế độ {modes}")
    print(f"mô hình  : {models}")
    print(f"seed     : {a.seed} · khung: {KHUNG_8N.window_buckets} bucket, "
          f"train [0, {KHUNG_8N.n_train}), test "
          f"[{KHUNG_8N.n_train + KHUNG_8N.n_val}, {KHUNG_8N.window_buckets})")
    print(f"vân tay  : {dau_van}")
    print("siêu tham số: " + (f"cố định từ {a.chosen_co_dinh}, không dò lại"
                              if co_dinh is not None else "dò lại bằng rolling-origin"))
    if bo_qua:
        print(f"bỏ qua   : {len(bo_qua)} tổ hợp đã chạy và còn khớp vân tay")
    print()

    dong, cv_log, chon_log = [], [], []
    so_lan_cham = 0
    t_tong = time.time()

    for env in envs:
        tang_env = tang_tat[tang_tat["env"] == env].set_index("series_id")["tang"]
        for mode in modes:
            for h in hs:
                t0 = time.time()
                d = chuan_bi(env, h, mode, feat_dir=feat_dir, proc_dir=proc_dir,
                             train_end=KHUNG_8N.n_train)
                m_fit = fit_mask(d.off, h, KHUNG_8N)
                m_test = mask_khoang(d.off, h,
                                     KHUNG_8N.n_train + KHUNG_8N.n_val,
                                     KHUNG_8N.window_buckets)
                print(f"{env} {mode} h={h:<2} {len(d.X):>9,} dòng · "
                      f"fit {int(m_fit.sum()):>9,} · test {int(m_test.sum()):>8,}"
                      + (f" · N1 loại {d.n_chuoi_loai_n1} chuỗi"
                         if d.n_chuoi_loai_n1 else ""))

                mau_svr = None
                for ten in models:
                    khoa = ("tn_a", env, env, mode, str(h), ten, str(a.seed), "test")
                    if tuple(map(str, khoa)) in bo_qua:
                        print(f"   {ten:8} bỏ qua")
                        continue
                    spec = REGISTRY.get(ten)
                    cand, diem = {}, float("nan")
                    mau = None
                    if ten not in BASELINE_NAMES:
                        if spec.mau_con is not None:
                            if mau_svr is None:
                                mau_svr = khoa_mau_con_svr(d, tang_env, spec.mau_con)
                            mau = mau_svr
                        if co_dinh is not None:
                            cand = co_dinh.get((env, mode, h, ten))
                            if cand is None:
                                raise SystemExit(
                                    f"Thiếu siêu tham số ({env}, {mode}, h={h}, "
                                    f"{ten}) trong {a.chosen_co_dinh}.")
                            diem = float("nan")
                        else:
                            cand, diem = chon_sieu_tham_so(
                                spec, d, KHUNG_8N, mau_con_ix=mau, log=cv_log)
                    kq = khop_va_cham(ten, d, mask_fit=m_fit, mask_cham=m_test,
                                      cand=cand, seed=a.seed,
                                      thu_muc_du_doan=out / "du_doan",
                                      thi_nghiem="tn_a", fold="test",
                                      mau_con_ix=mau)
                    kq.diem_val = diem
                    r = kq.__dict__ | dau_van_cot(dau_van)
                    dong.append(r)
                    chon_log.append({
                        "env": env, "mode": mode, "h": h, "model": ten,
                        "sieu_tham_so": kq.sieu_tham_so, "mae_p50_val": diem,
                        "n_train": kq.n_train, "giay": kq.giay,
                    })
                    if ten not in BASELINE_NAMES:
                        so_lan_cham += 1
                    print(f"   {ten:8} {kq.sieu_tham_so:<34} "
                          f"n_train={kq.n_train:>9,} {kq.giay:>7.1f}s")
                    ghi_dan(out, dong, cv_log, chon_log)
                print(f"   → xong {env} {mode} h={h} sau {time.time() - t0:.0f}s")

    ghi_dan(out, dong, cv_log, chon_log)
    (out / "meta.json").write_text(json.dumps({
        "thi_nghiem": "tn_a",
        "ngay": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "python": sys.executable,
        "env": envs, "horizons": hs, "modes": modes, "models": models,
        "seed": a.seed,
        "khung": {"window_buckets": KHUNG_8N.window_buckets,
                  "n_train": KHUNG_8N.n_train, "n_val": KHUNG_8N.n_val,
                  "n_folds": KHUNG_8N.n_folds},
        "luoi": {t: REGISTRY[t].luoi for t in models if t in REGISTRY},
        "loss": {t: REGISTRY[t].loss for t in models if t in REGISTRY},
        "so_lan_cham_test": so_lan_cham,
        "van_tay": dau_van,
        "phien_ban": _phien_ban(),
        "giay": round(time.time() - t_tong, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Chấm test {so_lan_cham} lần cho mô hình học máy.")
    print(f"Xong sau {(time.time() - t_tong) / 60:.1f} phút. Kết quả ở {out}")
    return 0


def dau_van_cot(dv: dict) -> dict:
    return {f"van_tay_{k}": v for k, v in dv.items()}


def ghi_dan(out: Path, dong, cv_log, chon_log) -> None:
    """Ghi ngay sau mỗi tổ hợp, nối thêm vào tệp cũ thay vì ghi đè."""
    if dong:
        _noi(out / "tom_tat.csv", pd.DataFrame(dong), KHOA)
    if cv_log:
        _noi(out / "cv_search.csv", pd.DataFrame(cv_log), KHOA_CV)
    if chon_log:
        _noi(out / "chosen.csv", pd.DataFrame(chon_log),
             ["env", "mode", "h", "model"])


def _noi(p: Path, df: pd.DataFrame, khoa) -> None:
    """Nối `df` vào tệp cũ; dòng cũ trùng khoá với `df` bị thay bằng dòng mới.

    `ghi_dan` được gọi sau mỗi mô hình và `cv_log` tích luỹ trong bộ nhớ, nên nếu không có khoá
    thì mỗi lần gọi lại ghi nối toàn bộ nhật ký từ đầu (K4.1 kết thúc với 74.180 dòng cho 1.170
    khoá). Vì vậy `cv_search.csv` cũng phải có khoá.
    """
    if p.exists():
        cu = pd.read_csv(p)
        if khoa and all(c in cu.columns for c in khoa):
            moi = set(map(tuple, df[khoa].astype(str).to_numpy()))
            giu = ~cu[khoa].astype(str).apply(tuple, axis=1).isin(moi)
            cu = cu[giu]
        df = pd.concat([cu, df], ignore_index=True)
    if khoa and df.duplicated(khoa).any():
        df = df.drop_duplicates(khoa, keep="last")
    df.to_csv(p, index=False)


if __name__ == "__main__":
    sys.exit(main())
