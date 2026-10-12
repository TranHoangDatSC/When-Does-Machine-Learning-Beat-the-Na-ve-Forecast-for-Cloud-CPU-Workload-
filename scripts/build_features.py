"""Sinh ma trận đặc trưng cho ba môi trường × ba horizon.

    python scripts/build_features.py --env all
    python scripts/build_features.py --env E2 --horizons 1
    python scripts/build_features.py --env all --modes N0,N1,N2      # ba chế độ chuẩn hoá

Đọc `data/processed/{env}.parquet`, gọi `cwp.features.make_feature_matrix`, ghi ra
`data/features/{env}_h{h}.parquet` — đúng 22 cột: `series_id`, `bucket`, 19 đặc
trưng, `target`.

**Ba chế độ chuẩn hoá (notebook 05).** Với `--modes`, script biến đổi chuỗi
`y` **trước** bằng `cwp.preprocess.normalize.bien_doi`, rồi sinh lại đủ 19 đặc trưng
từ chuỗi đã biến đổi, ghi ra `{env}_{mode}_h{h}.parquet`. Bốn đặc trưng lịch tự khắc
giữ nguyên vì chúng suy từ `bucket`, không từ `y`.

Vì sao sinh lại chứ không nhân `mu`/`sd` vào cột có sẵn: với N1 hai cách tương đương
(z-score là phép biến đổi affine), nhưng với **N2 thì không**: `roll_std` của chuỗi
sai phân khác hẳn `roll_std` của chuỗi gốc. Một đường đi chung cho cả ba chế độ thì
không phải nhớ ngoại lệ.

Phép đối chiếu với catalog chỉ áp cho **N0**: N1 phải có đúng số dòng của N0 (z-score
không sinh thêm NaN khi `sd > 0`), còn N2 mất vài dòng ở E3 vì sai phân biến điểm ngay
sau lỗ hổng thành `NaN`.

Việc lọc dòng nằm trọn trong `cwp.features`, script này không lọc thêm gì. Lọc theo
**luật cửa sổ** `[t-24, t]` sạch NaN và `t+h` không NaN, **không** bằng `dropna()` trên
ma trận đặc trưng: `dropna` lỏng hơn vì 19 đặc trưng chỉ chạm 15 trong 25 điểm của
cửa sổ.

Cuối cùng, script **đối chiếu** số dòng với `valid_rows_h*` trong
`data/catalog.parquet` (tính ở bước tiền xử lý) và thoát 1 nếu lệch. Lệch dù một dòng
cũng là lỗi, không phải nhiễu.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # ưu tiên src của chính repo này

from cwp.features import MATRIX_COLS, make_feature_matrix
from cwp.features.io_ma_tran import ghi as ghi_ma_tran
from cwp.preprocess.normalize import MODES, bien_doi

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2", "E3"]
HORIZONS = [1, 6, 12]


def neo_catalog(cat: pd.DataFrame | None, env: str, h: int) -> int | None:
    """Số dòng hợp lệ mà catalog ghi cho `env` ở horizon `h`, hoặc None nếu chưa có."""
    if cat is None:
        return None
    giu = cat[(cat["env"] == env) & cat["kept"]]
    cot = f"valid_rows_h{h}"
    if giu.empty or cot not in giu.columns:
        return None
    return int(giu[cot].sum())


def bien_doi_bang(df: pd.DataFrame, mode: str, train_end: int = 1612) -> pd.DataFrame:
    """Biến đổi cột `y` theo **từng chuỗi**, giữ nguyên mọi cột khác.

    Sắp xếp rồi reshape thành `(số chuỗi, 2304)` nên ranh giới chuỗi là ranh giới
    hàng: sai phân của N2 không thể bắc cầu từ đuôi chuỗi này sang đầu chuỗi kia.
    """
    if mode in (None, "N0"):  # None: không chuẩn hoá, tên tệp {env}_h{h}
        return df

    d = df.sort_values(["series_id", "bucket"], kind="mergesort").reset_index(drop=True)
    n = d["series_id"].nunique()
    if len(d) % n:
        raise SystemExit(f"{len(d)} dòng không chia hết cho {n} chuỗi: lưới không đều")
    m = len(d) // n

    Y = d["y"].to_numpy(dtype="float64").reshape(n, m)
    d["y"] = np.vstack([bien_doi(Y[i], mode, train_end) for i in range(n)]).reshape(-1)
    return d


def sinh_mot_moi_truong(env: str, horizons: list[int], proc_dir: Path,
                        out_dir: Path, cat: pd.DataFrame | None,
                        modes: list[str], train_end: int = 1612) -> list[dict]:
    """Sinh và ghi ma trận đặc trưng của một môi trường, trả về các dòng báo cáo."""
    nguon = proc_dir / f"{env}.parquet"
    if not nguon.exists():
        raise SystemExit(
            f"Thiếu {nguon}. Chạy tiền xử lý trước: "
            "python -m cwp.preprocess.build --env all"
        )

    t0 = time.time()
    df = pd.read_parquet(nguon)
    print(f"{env}: đọc {len(df):,} điểm, {df['series_id'].nunique()} chuỗi "
          f"({time.time() - t0:.1f}s)")

    # Gọi thẳng make_feature_matrix cho từng horizon, không tự sắp xếp trước: hàm đó tự
    # sắp và tự kiểm lưới.
    bao_cao = []
    for mode in modes:
        dfm = bien_doi_bang(df, mode, train_end)
        for h in horizons:
            t1 = time.time()
            X = make_feature_matrix(dfm, h=h)

            assert list(X.columns) == MATRIX_COLS, "schema lệch khỏi MATRIX_COLS"

            ten = f"{env}_h{h}.parquet" if mode is None else f"{env}_{mode}_h{h}.parquet"
            dich = out_dir / ten
            ghi_ma_tran(X, dich)

            # Catalog chỉ ràng buộc N0. N1/N2 so với N0 ở bảng tổng kết.
            neo = neo_catalog(cat, env, h) if mode in (None, "N0") else None
            bao_cao.append({
                "env": env,
                "mode": mode,
                "h": h,
                "dong": len(X),
                "neo": neo,
                "lech": None if neo is None else len(X) - neo,
                "chuoi": X["series_id"].nunique(),
                "mb": dich.stat().st_size / 1024 / 1024,
                "giay": time.time() - t1,
            })
            print(f"  {mode} h={h:<2} → {dich.name:<20} {len(X):>10,} dòng  "
                  f"{bao_cao[-1]['mb']:>6.1f} MB  ({bao_cao[-1]['giay']:.1f}s)")

    return bao_cao


def in_bang(bao_cao: list[dict], co_neo: bool) -> int:
    """In bảng số dòng, trả về số chỗ lệch so với catalog."""
    print()
    print("=" * 74)
    print("SỐ DÒNG MA TRẬN ĐẶC TRƯNG")
    print("=" * 74)

    print(f"{'':<4} {'mode':>5} {'h':>3} {'số dòng':>12} {'catalog':>12} "
          f"{'lệch':>8} {'so N0':>8} {'MB':>7}")

    n0 = {(r["env"], r["h"]): r["dong"] for r in bao_cao if r["mode"] == "N0"}
    lech = 0
    for r in bao_cao:
        if co_neo and r["neo"] is not None:
            dau = "khớp" if r["lech"] == 0 else f"{r['lech']:+,}"
            lech += r["lech"] != 0
            neo_s = f"{r['neo']:,}"
        else:
            dau, neo_s = "-", "-"
        goc = n0.get((r["env"], r["h"]))
        so_n0 = "-" if goc is None or r["mode"] == "N0" else (
            "bằng" if r["dong"] == goc else f"{r['dong'] - goc:+,}")
        print(f"{r['env']:<4} {r['mode'] or '-':>5} {r['h']:>3} {r['dong']:>12,} "
              f"{neo_s:>12} {dau:>8} {so_n0:>8} {r['mb']:>7.1f}")

    print("=" * 74)
    return lech


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="all",
                    help="mã môi trường, hoặc all cho ba môi trường 8 ngày")
    ap.add_argument("--train-end", type=int, default=1612,
                    help="ranh giới phần train, chỉ dùng cho N1 và N2")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                    help="danh sách horizon, ngăn bằng dấu phẩy (mặc định 1,6,12)")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--catalog", default="data/catalog.parquet")
    ap.add_argument("--modes", default=None,
                    help="danh sách chế độ chuẩn hoá, ví dụ N0,N1,N2. "
                         "Bỏ trống thì không chuẩn hoá (tên tệp không có mode).")
    a = ap.parse_args()

    envs = ENVS if a.env == "all" else [x.strip() for x in a.env.split(",")]
    horizons = [int(x) for x in a.horizons.split(",")]
    if a.modes is None:
        modes = [None]                       # {env}_h{h}.parquet
    else:
        modes = [x.strip().upper() for x in a.modes.split(",")]
        la = [m for m in modes if m not in MODES]
        if la:
            raise SystemExit(f"Chế độ không hợp lệ: {la}. Chỉ có {list(MODES)}.")
    proc_dir = ROOT / a.processed
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_path = ROOT / a.catalog
    cat = pd.read_parquet(cat_path) if cat_path.exists() else None
    if cat is None:
        print(f"CẢNH BÁO: không thấy {cat_path}, bỏ qua bước đối chiếu với catalog.")

    print()
    print("SINH ĐẶC TRƯNG" + (f" · chế độ {','.join(modes)}" if a.modes else ""))
    print(f"nguồn : {proc_dir}")
    print(f"đích  : {out_dir}")
    print()

    t0 = time.time()
    bao_cao = []
    for env in envs:
        bao_cao += sinh_mot_moi_truong(env, horizons, proc_dir, out_dir, cat, modes,
                                    train_end=a.train_end)

    lech = in_bang(bao_cao, co_neo=cat is not None)
    print(f"{len(bao_cao)} tệp, {time.time() - t0:.1f}s tổng.")

    if cat is None:
        return 0
    if lech:
        print()
        print(f"CHƯA ĐẠT — {lech}/{len(bao_cao)} tệp lệch so với catalog.")
        print("NHIỀU dòng hơn: có thể đã lọc bằng dropna() thay vì luật cửa sổ, hoặc "
              "dùng min_periods=1.")
        print("ÍT dòng hơn   : có thể thiếu groupby(series_id), hoặc nhầm chỉ số t+h.")
        return 1

    print("Mọi tệp khớp tuyệt đối với catalog.parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
