"""Thí nghiệm C: dùng lại mô hình giữa các tập vết, và thí nghiệm máy giả.

    python scripts/tn_c.py --cap E1:E2 --modes N0 --horizons 1 --models ridge
    python scripts/tn_c.py --tat-ca
    python scripts/tn_c.py --may-gia
    python scripts/tn_c.py --tat-ca --resume

Sáu chiều chuyển giao: E1→E2, E2→E1, E1→E3, E3→E1, E2→E3, E3→E2. Mô hình khớp trên
`[0, 1957)` của **nguồn**, chấm trên `[1957, 2304)` của **đích**, không huấn luyện
lại ở đích.

Ba quy ước
----------

1. **Không chạy lại chiều tại chỗ.** Mẫu số của tỉ số mất mát chuyển giao là mô hình
   học ngay tại đích, và nó chính là tổ hợp tương ứng của Thí nghiệm A: cùng dữ liệu,
   cùng mã, cùng hạt giống, cùng đường ghi dự đoán. `scripts/tong_hop.py` ghép hai
   nguồn dự đoán lại. Chạy lại chỉ tốn máy và thêm một chỗ để lệch nhau.

2. **Siêu tham số lấy theo môi trường nguồn**, đọc từ `results/tn_a/chosen.csv`.
   Không dò lại trên dữ liệu đích: dò lại là để đích tham gia vào việc chọn mô hình.

3. **Chuẩn hoá N1 dùng thống kê phần train của chính máy đích.** Câu hỏi của thí
   nghiệm vì vậy là "quy luật biến động có dùng lại được không", không phải "chuyển
   được sang nơi hoàn toàn chưa có dữ liệu".

Thí nghiệm máy giả dùng cùng đường mã, với bốn chiều E1a→E1a, E1a→E1g, E1g→E1g và
E1g→E1a; siêu tham số lấy của E1.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.evaluation.splits import KHUNG_8N, fit_mask, mask_khoang  # noqa: E402
from cwp.models.registry import SEED  # noqa: E402
from cwp.quy_trinh import van_tay as vt  # noqa: E402
from cwp.quy_trinh.chay import chuan_bi, khop_va_cham  # noqa: E402
from tn_a import KHOA, dau_van_cot, ghi_dan  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CAP = [("E1", "E2"), ("E2", "E1"), ("E1", "E3"), ("E3", "E1"),
       ("E2", "E3"), ("E3", "E2")]
CAP_MAY_GIA = [("E1a", "E1a"), ("E1a", "E1g"), ("E1g", "E1g"), ("E1g", "E1a")]
# Máy giả mượn siêu tham số của E1.
NGUON_THAM_SO = {"E1a": "E1", "E1g": "E1"}
HORIZONS = [1, 6, 12]
MODES = ["N0", "N1"]
MODELS = ["ridge", "xgb", "xgb_mae"]
# Máy giả: chiều **tại chỗ** chạy thêm mô hình cơ sở. `chuyen_giao.csv` lấy "naive tại
# đích" từ chiều tại chỗ của cùng môi trường đích, và protocol mục 9 đòi báo hai tỉ số
# cùng lúc (so với mô hình học tại đích, và so với naive tại đích). Thí nghiệm C không cần
# vì đích của nó là E1, E2, E3 nên naive tại đích đã có ở Thí nghiệm A; còn E1a và E1g
# không xuất hiện ở bất kỳ thí nghiệm nào khác, nên nếu đây không chạy thì không có naive.
# Chiều chuyển giao KHÔNG chạy mô hình cơ sở, vì dự đoán cơ sở không phụ thuộc mô hình
# nguồn nên sẽ trùng đúng chiều tại chỗ.
MODELS_CO_SO_TAI_CHO = ["naive", "ma6", "seasonal"]


def doc_sieu_tham_so(chosen: Path) -> dict:
    """`(env, mode, h, model) -> dict siêu tham số` lấy từ Thí nghiệm A."""
    if not chosen.exists():
        raise SystemExit(
            f"Thiếu {chosen}. Chạy Thí nghiệm A trước: siêu tham số của Thí nghiệm C "
            "lấy theo môi trường nguồn, không dò lại."
        )
    d = pd.read_csv(chosen)
    ra = {}
    for r in d.itertuples():
        ra[(r.env, r.mode, int(r.h), r.model)] = json.loads(r.sieu_tham_so)
    return ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", default=None,
                    help="danh sách cặp dạng E1:E2,E2:E1; bỏ trống là sáu chiều")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--chosen", default="results/tn_a/chosen.csv")
    ap.add_argument("--out", default="results/tn_c")
    ap.add_argument("--may-gia", action="store_true",
                    help="chạy bốn chiều của thí nghiệm máy giả")
    ap.add_argument("--tat-ca", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    if a.may_gia:
        cap = CAP_MAY_GIA
    elif a.cap:
        cap = [tuple(x.split(":")) for x in a.cap.split(",")]
    else:
        cap = CAP
    hs = HORIZONS if a.tat_ca else [int(x) for x in a.horizons.split(",")]
    modes = MODES if a.tat_ca else [x.strip().upper() for x in a.modes.split(",")]
    models = MODELS if a.tat_ca else [x.strip() for x in a.models.split(",")]

    feat_dir, proc_dir = ROOT / a.features, ROOT / a.processed
    out = ROOT / a.out
    (out / "du_doan").mkdir(parents=True, exist_ok=True)
    thi_nghiem = "may_gia" if a.may_gia else "tn_c"
    co_so_tai_cho = MODELS_CO_SO_TAI_CHO if a.may_gia else []

    envs = sorted({e for c in cap for e in c})
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

    print()
    print(f"THÍ NGHIỆM {'MÁY GIẢ' if a.may_gia else 'C'} — dùng lại mô hình")
    print(f"python  : {sys.executable}")
    print(f"cặp     : {cap}")
    print(f"tầm {hs} · chế độ {modes} · mô hình {models} · seed {a.seed}")
    print(f"vân tay : {dau_van}")
    print()

    dong = []
    t_tong = time.time()
    # Một mô hình dùng cho mọi đích của cùng nguồn, nên khớp một lần rồi chấm nhiều nơi.
    for nguon in sorted({c[0] for c in cap}):
        dich_list = [c[1] for c in cap if c[0] == nguon]
        for mode in modes:
            for h in hs:
                dn = chuan_bi(nguon, h, mode, feat_dir=feat_dir, proc_dir=proc_dir,
                              train_end=KHUNG_8N.n_train)
                m_fit = fit_mask(dn.off, h, KHUNG_8N)
                for dich in dich_list:
                    dd = dn if dich == nguon else chuan_bi(
                        dich, h, mode, feat_dir=feat_dir, proc_dir=proc_dir,
                        train_end=KHUNG_8N.n_train)
                    m_test = mask_khoang(dd.off, h,
                                         KHUNG_8N.n_train + KHUNG_8N.n_val,
                                         KHUNG_8N.window_buckets)
                    mo_hinh_ho = (models + co_so_tai_cho
                                  if dich == nguon else models)
                    for ten in mo_hinh_ho:
                        khoa = (thi_nghiem, nguon, dich, mode, str(h), ten,
                                str(a.seed), "test")
                        if khoa in bo_qua:
                            print(f"{nguon}→{dich} {mode} h={h} {ten}: bỏ qua")
                            continue
                        env_tham_so = NGUON_THAM_SO.get(nguon, nguon)
                        cand = stp.get((env_tham_so, mode, h, ten))
                        if cand is None:
                            raise SystemExit(
                                f"Thiếu siêu tham số của ({env_tham_so}, {mode}, "
                                f"h={h}, {ten}) trong {a.chosen}."
                            )
                        kq = khop_va_cham(ten, dn, mask_fit=m_fit,
                                          mask_cham=m_test, cand=cand,
                                          seed=a.seed,
                                          thu_muc_du_doan=out / "du_doan",
                                          thi_nghiem=thi_nghiem, fold="test",
                                          dich=dd)
                        dong.append(kq.__dict__ | dau_van_cot(dau_van))
                        print(f"{nguon}→{dich} {mode} h={h:<2} {ten:8} "
                              f"n_train={kq.n_train:>9,} chấm {kq.n_cham:>8,} "
                              f"{kq.giay:>7.1f}s")
                        ghi_dan(out, dong, [], [])

    ghi_dan(out, dong, [], [])
    (out / "meta.json").write_text(json.dumps({
        "thi_nghiem": thi_nghiem,
        "ngay": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:], "python": sys.executable,
        "cap": [list(c) for c in cap], "horizons": hs, "modes": modes,
        "models": models, "co_so_tai_cho": co_so_tai_cho, "seed": a.seed,
        "chieu_tai_cho": "lấy từ Thí nghiệm A, không chạy lại",
        "van_tay": dau_van,
        "phien_ban": {"python": platform.python_version()},
        "giay": round(time.time() - t_tong, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Xong sau {(time.time() - t_tong) / 60:.1f} phút. Kết quả ở {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
