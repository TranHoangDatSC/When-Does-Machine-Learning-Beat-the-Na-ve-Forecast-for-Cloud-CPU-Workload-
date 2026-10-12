"""Kiểm tra môi trường, repo, dữ liệu và cấu hình trước khi chạy bất cứ thứ gì.

    python scripts/kiem_tra_moi_truong.py
    python scripts/kiem_tra_moi_truong.py --tn-b     # kiểm thêm dữ liệu dài ngày

Thoát 0 khi mọi phép kiểm bắt buộc đạt, thoát 1 khi có phép kiểm hỏng. In đường dẫn
Python đang dùng và đường dẫn gói `cwp`, vì hai thứ đó là chỗ dễ sai nhất: nếu gói `cwp`
từng được cài ở chế độ editable từ một thư mục khác, `import cwp` sẽ lấy mã ở đó thay vì
mã trong repo này.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2", "E3"]
ENVS_B = ["E1L", "E2m07", "E2m08", "E2m09"]
HORIZONS = [1, 6, 12]
GHIM = {"pandas": "2.2.3", "numpy": "2.1.3", "pyarrow": "18.1.0",
        "scipy": "1.14.1", "sklearn": "1.5.2", "xgboost": "2.1.3"}


class Bao:
    def __init__(self):
        self.dong = []

    def them(self, ma, dat, mo_ta, chi_tiet="", bat_buoc=True):
        self.dong.append({"ma": ma, "dat": bool(dat), "mo_ta": mo_ta,
                          "chi_tiet": chi_tiet, "bat_buoc": bat_buoc})

    def in_ra(self) -> int:
        print()
        for r in self.dong:
            dau = "đạt " if r["dat"] else ("HỎNG" if r["bat_buoc"] else "thiếu")
            print(f"  [{dau}] {r['ma']:<6} {r['mo_ta']}")
            if r["chi_tiet"]:
                print(f"              {r['chi_tiet']}")
        hong = [r for r in self.dong if not r["dat"] and r["bat_buoc"]]
        thieu = [r for r in self.dong if not r["dat"] and not r["bat_buoc"]]
        print()
        print(f"{len(self.dong) - len(hong) - len(thieu)}/{len(self.dong)} đạt, "
              f"{len(thieu)} chưa có, {len(hong)} hỏng.")
        return 1 if hong else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tn-b", action="store_true")
    a = ap.parse_args()

    print("KIỂM TRA ĐẦU VÀO")
    print(f"repo   : {ROOT}")
    print(f"python : {sys.executable}")

    b = Bao()
    try:
        import cwp
        duong_dan = Path(cwp.__file__).resolve()
        print(f"cwp    : {duong_dan}")
        b.them("M01", ROOT in duong_dan.parents,
               "gói cwp lấy từ chính repo này",
               f"đang lấy từ {duong_dan}")
    except Exception as e:
        print(f"cwp    : không import được ({e})")
        b.them("M01", False, "gói cwp import được", str(e))
        return b.in_ra()

    b.them("M02", Path(sys.executable).resolve().is_relative_to(ROOT.resolve()),
           "python đang dùng là .venv của repo này",
           f"{sys.executable}", bat_buoc=False)

    for ten, ghim in GHIM.items():
        try:
            v = __import__(ten).__version__
        except Exception:
            v = None
        b.them(f"M03.{ten}", v == ghim, f"{ten} đúng phiên bản ghim {ghim}",
               f"đang có {v}")

    from cwp.features.io_ma_tran import co_nhan
    from cwp.features.spec import MATRIX_COLS

    b.them("M04", len(MATRIX_COLS) == 23, "lược đồ ma trận đặc trưng là 23 cột",
           f"{len(MATRIX_COLS)} cột")

    envs = ENVS + (ENVS_B if a.tn_b else [])
    hs = HORIZONS if not a.tn_b else HORIZONS + []
    for e in envs:
        p = ROOT / "data" / "processed" / f"{e}.parquet"
        b.them(f"D01.{e}", p.exists(), f"có data/processed/{e}.parquet",
               "" if p.exists() else "chạy bước dựng dữ liệu",
               bat_buoc=e in ENVS)
    for e in envs:
        for h in (hs if e in ENVS else [1, 12]):
            p = ROOT / "data" / "features" / f"{e}_h{h}.parquet"
            b.them(f"D02.{e}.h{h}", co_nhan(p),
                   f"ma trận {e}_h{h} có đúng nhãn lược đồ",
                   "" if p.exists() else "chưa sinh", bat_buoc=e in ENVS)
    for e in envs:
        for h in (hs if e in ENVS else [1, 12]):
            p = ROOT / "data" / "eval_rows" / f"{e}_h{h}.parquet"
            b.them(f"D03.{e}.h{h}", p.exists(),
                   f"có tập dòng đánh giá chung {e}_h{h}", "",
                   bat_buoc=e in ENVS)

    cv = ROOT / "results" / "tables" / "cv.csv"
    co_cot = False
    if cv.exists():
        import pandas as pd
        co_cot = "cv_train" in pd.read_csv(cv, nrows=1).columns
    b.them("D04", co_cot, "cv.csv có cột cv_train",
           "sinh lại bằng scripts/fig_burstiness.py")

    for t in ("datasets.yaml", "preprocess.yaml", "split.yaml", "features.yaml"):
        b.them(f"C01.{t}", (ROOT / "config" / t).exists(), f"có config/{t}")

    trong = shutil.disk_usage(ROOT).free / 1e9
    b.them("C03", trong > 20, f"còn trống {trong:.0f} GB trên ổ đĩa",
           "cần khoảng 20 GB cho dữ liệu độ nhạy và dự đoán")

    from cwp.quy_trinh import van_tay as vt
    dv = vt.van_tay(ROOT, [ROOT / "data" / "processed" / f"{e}.parquet"
                           for e in ENVS])
    print()
    print(f"vân tay hiện tại: {dv}")
    return b.in_ra()


if __name__ == "__main__":
    sys.exit(main())
