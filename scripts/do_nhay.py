"""Phần độ nhạy: ngưỡng lọc quần thể, và kiểm tra Random Forest phạm vi hẹp.

    python scripts/do_nhay.py --nguong 0,0.5,2 --buoc du-lieu
    python scripts/do_nhay.py --nguong 0,0.5,2 --buoc mo-hinh
    python scripts/do_nhay.py --buoc rf

Hai phần rời nhau, chạy độc lập.

Ngưỡng lọc quần thể
-------------------

Quần thể chính dùng ngưỡng CPU trung bình phần train từ 1% trở lên. Phần độ nhạy chạy
lại toàn bộ đường dữ liệu ở các ngưỡng **chốt trước** 0%, 0,5% và 2%, trong đó 0%
nghĩa là không lọc theo mức tải nhưng vẫn giữ các phép kiểm chất lượng dữ liệu. Ngưỡng
**không** được chọn theo kết quả.

Mỗi ngưỡng có thư mục dữ liệu riêng `data/nguong_{x}/` và thư mục kết quả riêng
`results/do_nhay/nguong_{x}/`, nên không lần nào ghi đè lần nào. Mô hình chạy ở
mức rút gọn đã chốt: `naive`, `xgb`, `xgb_mae`, hai chế độ N0 và N1, tầm 1 và 12, siêu
tham số lấy từ Thí nghiệm A, không dò lại.

Random Forest
-------------

Chỉ một cấu hình định trước, đúng cấu hình đã chọn trước khi chạy: 50 cây, `max_depth`
16, `min_samples_leaf` 5. Chạy ở N0, tầm 1 và 12, ba môi trường, không dò siêu tham
số. Mục đích là làm đối chứng phạm vi hẹp, không phải để so họ thuật toán.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.evaluation.splits import KHUNG_8N, fit_mask, mask_khoang  # noqa: E402
from cwp.models.registry import REGISTRY, SEED  # noqa: E402
from cwp.quy_trinh import van_tay as vt  # noqa: E402
from cwp.quy_trinh.chay import chuan_bi, khop_va_cham  # noqa: E402
from tn_a import dau_van_cot, ghi_dan  # noqa: E402
from tn_c import doc_sieu_tham_so  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2", "E3"]
NGUONG = [0.0, 0.5, 2.0]        # 1.0 là quần thể chính, đã chạy ở Thí nghiệm A
HORIZONS = [1, 12]
MODES = ["N0", "N1"]
MODELS = ["naive", "xgb", "xgb_mae"]
# Cấu hình Random Forest định trước, cấu hình đã chọn trước khi chạy.
RF_CAND = {"max_depth": 16}


def ten_nguong(x: float) -> str:
    return f"nguong_{str(x).replace('.', 'p')}"


def chay_lenh(cmd: list) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"Lệnh thoát với mã {r.returncode}.")


def buoc_du_lieu(nguong: list, envs: list) -> None:
    """Dựng lại dữ liệu và ma trận đặc trưng cho từng ngưỡng."""
    for x in nguong:
        ten = ten_nguong(x)
        proc = ROOT / "data" / ten / "processed"
        feat = ROOT / "data" / ten / "features"
        proc.mkdir(parents=True, exist_ok=True)
        feat.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Ngưỡng {x}% → data/{ten}/")
        chay_lenh([sys.executable, "-m", "cwp.preprocess.build",
                   "--env", ",".join(envs), "--min-mean", str(x),
                   "--processed", str(proc.relative_to(ROOT)),
                   "--catalog", str((ROOT / "data" / ten / "catalog.parquet")
                                    .relative_to(ROOT))])
        chay_lenh([sys.executable, "scripts/build_features.py",
                   "--env", ",".join(envs),
                   "--processed", str(proc.relative_to(ROOT)),
                   "--out", str(feat.relative_to(ROOT)),
                   "--catalog", str((ROOT / "data" / ten / "catalog.parquet")
                                    .relative_to(ROOT))])
        chay_lenh([sys.executable, "scripts/tap_dong_chung.py",
                   "--env", ",".join(envs), "--horizons", "1,12",
                   "--processed", str(proc.relative_to(ROOT)),
                   "--features", str(feat.relative_to(ROOT)),
                   "--out", f"data/{ten}/eval_rows"])


def buoc_mo_hinh(nguong: list, envs: list, chosen: Path, seed: int) -> None:
    """Chạy bộ mô hình rút gọn trên từng ngưỡng."""
    stp = doc_sieu_tham_so(chosen)
    for x in nguong:
        ten = ten_nguong(x)
        proc = ROOT / "data" / ten / "processed"
        feat = ROOT / "data" / ten / "features"
        out = ROOT / "results" / "do_nhay" / ten
        (out / "du_doan").mkdir(parents=True, exist_ok=True)
        dau_van = vt.van_tay(ROOT, [proc / f"{e}.parquet" for e in envs])
        dong = []
        print(f"\n=== Ngưỡng {x}% → {out}")
        for env in envs:
            for mode in MODES:
                for h in HORIZONS:
                    d = chuan_bi(env, h, mode, feat_dir=feat, proc_dir=proc,
                                 train_end=KHUNG_8N.n_train)
                    m_fit = fit_mask(d.off, h, KHUNG_8N)
                    m_test = mask_khoang(d.off, h,
                                         KHUNG_8N.n_train + KHUNG_8N.n_val,
                                         KHUNG_8N.window_buckets)
                    print(f"{env} {mode} h={h:<2} {len(d.X):>9,} dòng · "
                          f"{d.n_chuoi_tong} chuỗi")
                    for m in MODELS:
                        cand = {} if m == "naive" else stp.get((env, mode, h, m))
                        if m != "naive" and cand is None:
                            raise SystemExit(
                                f"Thiếu siêu tham số ({env}, {mode}, h={h}, {m})."
                            )
                        kq = khop_va_cham(m, d, mask_fit=m_fit, mask_cham=m_test,
                                          cand=cand, seed=seed,
                                          thu_muc_du_doan=out / "du_doan",
                                          thi_nghiem="do_nhay", fold="test")
                        kq.ghi_chu = f"nguong={x}"
                        dong.append(kq.__dict__ | dau_van_cot(dau_van))
                        print(f"   {m:8} n_train={kq.n_train:>9,} "
                              f"{kq.giay:>7.1f}s")
                        ghi_dan(out, dong, [], [])
        (out / "meta.json").write_text(json.dumps(
            {"nguong": x, "models": MODELS, "modes": MODES,
             "horizons": HORIZONS, "seed": seed, "van_tay": dau_van,
             "sieu_tham_so": "lấy từ Thí nghiệm A, không dò lại"},
            ensure_ascii=False, indent=2), encoding="utf-8")


def buoc_rf(envs: list, seed: int) -> None:
    """Random Forest, một cấu hình định trước, N0, tầm 1 và 12."""
    out = ROOT / "results" / "rf"
    (out / "du_doan").mkdir(parents=True, exist_ok=True)
    feat, proc = ROOT / "data" / "features", ROOT / "data" / "processed"
    dau_van = vt.van_tay(ROOT, [proc / f"{e}.parquet" for e in envs])
    dong = []
    print(f"\n=== Random Forest, cấu hình định trước {RF_CAND} → {out}")
    for env in envs:
        for h in HORIZONS:
            d = chuan_bi(env, h, "N0", feat_dir=feat, proc_dir=proc,
                         train_end=KHUNG_8N.n_train)
            m_fit = fit_mask(d.off, h, KHUNG_8N)
            m_test = mask_khoang(d.off, h, KHUNG_8N.n_train + KHUNG_8N.n_val,
                                 KHUNG_8N.window_buckets)
            for m in ("naive", "rf"):
                kq = khop_va_cham(m, d, mask_fit=m_fit, mask_cham=m_test,
                                  cand=RF_CAND if m == "rf" else {}, seed=seed,
                                  thu_muc_du_doan=out / "du_doan",
                                  thi_nghiem="rf", fold="test")
                dong.append(kq.__dict__ | dau_van_cot(dau_van))
                print(f"{env} h={h:<2} {m:6} n_train={kq.n_train:>9,} "
                      f"{kq.giay:>8.1f}s")
                ghi_dan(out, dong, [], [])
    (out / "meta.json").write_text(json.dumps(
        {"cau_hinh": RF_CAND, "mac_dinh_trong_registry": REGISTRY["rf"].luoi,
         "envs": envs, "horizons": HORIZONS, "mode": "N0", "seed": seed,
         "van_tay": dau_van,
         "ghi_chu": "cấu hình định trước, không dò siêu tham số; chỉ để đối "
                    "chứng phạm vi hẹp"},
        ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--buoc", required=True,
                    choices=["du-lieu", "mo-hinh", "rf"])
    ap.add_argument("--nguong", default=",".join(str(x) for x in NGUONG))
    ap.add_argument("--env", default=",".join(ENVS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chosen", default="results/tn_a/chosen.csv")
    a = ap.parse_args()

    envs = [x.strip() for x in a.env.split(",")]
    nguong = [float(x) for x in a.nguong.split(",")] if a.nguong else []
    t0 = time.time()
    print(f"python: {sys.executable}")
    if a.buoc == "du-lieu":
        buoc_du_lieu(nguong, envs)
    elif a.buoc == "mo-hinh":
        buoc_mo_hinh(nguong, envs, ROOT / a.chosen, a.seed)
    else:
        buoc_rf(envs, a.seed)
    print(f"\nXong sau {(time.time() - t0) / 60:.1f} phút.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
