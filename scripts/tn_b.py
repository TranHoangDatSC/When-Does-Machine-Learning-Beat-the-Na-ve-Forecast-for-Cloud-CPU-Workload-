"""Thí nghiệm B: kết luận có giữ nguyên qua các giai đoạn thời gian không.

    python scripts/tn_b.py --env E2m08 --gioi-han-fold 1      # lượt thử đo chi phí
    python scripts/tn_b.py --env E1L,E2m07,E2m09              # lượt xác nhận
    python scripts/tn_b.py --env E1L --resume

Cửa sổ mở rộng dần. Với mỗi môi trường dài ngày:

    lịch sử tối thiểu   14 ngày = 4.032 bucket
    khoảng huấn luyện lại   7 ngày = 2.016 bucket
    fold i: khớp trên [0, o_i), chấm trên [o_i, o_i + 2016)

Mỗi fold ghi dự đoán riêng với nhãn `fold = "o<offset>"`, nên
`scripts/tong_hop.py` tách được kết quả theo từng ngày và theo thứ trong tuần mà
không phải chạy lại.

Bốn quy ước, chốt trước lượt xác nhận
-------------------------------------

1. **Siêu tham số lấy từ Thí nghiệm A**, theo môi trường tương ứng: E1L lấy của E1,
   ba tháng của Rnd lấy của E2. Không dò lại trên giai đoạn xác nhận, vì dò lại là để
   chính giai đoạn đó tham gia chọn mô hình.

2. **Quần thể máy cố định theo lịch sử tối thiểu.** Bước tiền xử lý chạy với
   `--n-train 4032`, tức quyết định giữ hay loại một máy chỉ dùng 14 ngày đầu. Mọi
   fold về sau dùng đúng quần thể đó.

3. **Thống kê chuẩn hoá N1 tính lại ở từng fold**, trên `[0, o_i)` của chính máy đó.

4. **Không chạm giai đoạn xác nhận khi chọn thiết kế.** E2m08 là dữ liệu phát triển
   và chỉ dùng để đo chi phí; E1L, E2m07 và E2m09 chỉ chạy sau khi protocol đã khoá.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.evaluation.splits import BUCKET_MOI_NGAY, fold_bounds_truot, mask_khoang  # noqa: E402
from cwp.features.io_ma_tran import doc as doc_ma_tran  # noqa: E402
from cwp.models.baselines import BASELINE_NAMES  # noqa: E402
from cwp.models.registry import SEED  # noqa: E402
from cwp.quy_trinh import van_tay as vt  # noqa: E402
from cwp.quy_trinh.chay import chuan_bi, khop_va_cham  # noqa: E402
from tn_a import KHOA, dau_van_cot, ghi_dan  # noqa: E402
from tn_c import doc_sieu_tham_so  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1L", "E2m07", "E2m09"]
ENV_PHAT_TRIEN = "E2m08"
# Môi trường nào mượn siêu tham số của môi trường nào ở Thí nghiệm A.
NGUON_THAM_SO = {"E1L": "E1", "E2m07": "E2", "E2m08": "E2", "E2m09": "E2"}
HORIZONS = [1, 12]
MODES = ["N0", "N1"]
MODELS = ["naive", "ma6", "seasonal", "seasonal_tuan", "ridge", "xgb", "xgb_mae"]
NGAY_TOI_THIEU = 14
NGAY_MOI_LAN = 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default=",".join(ENVS))
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ngay-toi-thieu", type=int, default=NGAY_TOI_THIEU)
    ap.add_argument("--ngay-moi-lan", type=int, default=NGAY_MOI_LAN)
    ap.add_argument("--gioi-han-fold", type=int, default=0,
                    help="chỉ chạy bấy nhiêu fold đầu; 0 là chạy hết")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--chosen", default="results/tn_a/chosen.csv")
    ap.add_argument("--out", default="results/tn_b")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    envs = [x.strip() for x in a.env.split(",")]
    hs = [int(x) for x in a.horizons.split(",")]
    modes = [x.strip().upper() for x in a.modes.split(",")]
    models = [x.strip() for x in a.models.split(",")]

    feat_dir, proc_dir = ROOT / a.features, ROOT / a.processed
    out = ROOT / a.out
    (out / "du_doan").mkdir(parents=True, exist_ok=True)

    tep = [proc_dir / f"{e}.parquet" for e in envs]
    tep += [feat_dir / f"{e}_h{h}.parquet" for e in envs for h in hs]
    dau_van = vt.van_tay(ROOT, tep)
    bo_qua = set()
    if a.resume and (out / "tom_tat.csv").exists():
        d = pd.read_csv(out / "tom_tat.csv")
        giu = ((d["van_tay_ma"] == dau_van["ma"])
               & (d["van_tay_cau_hinh"] == dau_van["cau_hinh"])
               & (d["van_tay_du_lieu"] == dau_van["du_lieu"]))
        bo_qua = {tuple(r) for r in d.loc[giu, KHOA].astype(str).to_numpy()}

    stp = doc_sieu_tham_so(ROOT / a.chosen)
    moc_dau = a.ngay_toi_thieu * BUCKET_MOI_NGAY
    buoc = a.ngay_moi_lan * BUCKET_MOI_NGAY

    print()
    print("THÍ NGHIỆM B — độ ổn định theo thời gian")
    print(f"python  : {sys.executable}")
    print(f"môi trường {envs} · tầm {hs} · chế độ {modes} · seed {a.seed}")
    print(f"lịch sử tối thiểu {a.ngay_toi_thieu} ngày · huấn luyện lại mỗi "
          f"{a.ngay_moi_lan} ngày")
    print(f"vân tay : {dau_van}")
    if ENV_PHAT_TRIEN in envs and len(envs) > 1:
        print(f"CẢNH BÁO: {ENV_PHAT_TRIEN} là dữ liệu phát triển, "
              "không gộp chung bảng với giai đoạn xác nhận.")
    print()

    dong = []
    t_tong = time.time()
    for env in envs:
        proc = pd.read_parquet(proc_dir / f"{env}.parquet",
                               columns=["series_id", "bucket", "y"])
        b0 = int(proc["bucket"].min())
        W = int(proc["bucket"].max()) - b0 + 1
        folds = fold_bounds_truot(moc_dau, W, buoc)
        if a.gioi_han_fold:
            folds = folds[:a.gioi_han_fold]
        print(f"{env}: {proc['series_id'].nunique()} chuỗi · cửa sổ {W} bucket "
              f"= {W / BUCKET_MOI_NGAY:.1f} ngày · {len(folds)} fold")

        for h in hs:
            X0 = doc_ma_tran(feat_dir / f"{env}_h{h}.parquet")
            for i_f, bien in enumerate(folds):
                o, het = bien["train"][1], bien["val"][1]
                for mode in modes:
                    d = chuan_bi(env, h, mode, feat_dir=feat_dir,
                                 proc_dir=proc_dir, train_end=o,
                                 X_goc=X0, processed=proc, dtype="float32")
                    m_fit = mask_khoang(d.off, h, 0, o)
                    m_cham = mask_khoang(d.off, h, o, het)
                    if not m_cham.any():
                        continue
                    for ten in models:
                        fold_ten = f"o{o}"
                        khoa = ("tn_b", env, env, mode, str(h), ten,
                                str(a.seed), fold_ten)
                        if khoa in bo_qua:
                            continue
                        cand = {}
                        if ten not in BASELINE_NAMES:
                            cand = stp.get((NGUON_THAM_SO[env], mode, h, ten))
                            if cand is None:
                                raise SystemExit(
                                    f"Thiếu siêu tham số ({NGUON_THAM_SO[env]}, "
                                    f"{mode}, h={h}, {ten}) trong {a.chosen}."
                                )
                        kq = khop_va_cham(ten, d, mask_fit=m_fit,
                                          mask_cham=m_cham, cand=cand,
                                          seed=a.seed,
                                          thu_muc_du_doan=out / "du_doan",
                                          thi_nghiem="tn_b", fold=fold_ten)
                        dong.append(kq.__dict__ | dau_van_cot(dau_van))
                        print(f"  {env} {mode} h={h:<2} fold {i_f} o={o:<5} "
                              f"{ten:14} fit={kq.n_train:>9,} "
                              f"chấm={kq.n_cham:>7,} {kq.giay:>7.1f}s")
                        ghi_dan(out, dong, [], [])
            del X0

    ghi_dan(out, dong, [], [])
    (out / "meta.json").write_text(json.dumps({
        "thi_nghiem": "tn_b",
        "ngay": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:], "python": sys.executable,
        "env": envs, "horizons": hs, "modes": modes, "models": models,
        "seed": a.seed,
        "ngay_toi_thieu": a.ngay_toi_thieu, "ngay_moi_lan": a.ngay_moi_lan,
        "sieu_tham_so": "lấy từ Thí nghiệm A theo NGUON_THAM_SO",
        "van_tay": dau_van,
        "giay": round(time.time() - t_tong, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Xong sau {(time.time() - t_tong) / 60:.1f} phút. Kết quả ở {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
