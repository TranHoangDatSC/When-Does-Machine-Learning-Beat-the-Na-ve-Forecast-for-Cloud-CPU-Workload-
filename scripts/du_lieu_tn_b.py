"""Dựng dữ liệu dài ngày cho Thí nghiệm B.

    python scripts/du_lieu_tn_b.py                 # cả bốn môi trường
    python scripts/du_lieu_tn_b.py --env E2m08     # chỉ dữ liệu phát triển

Bốn môi trường, cùng nguồn dữ liệu thô với Thí nghiệm A nhưng khác cửa sổ thời gian:

    E1L     Bitbrains fastStorage, 30 ngày từ 12/08/2013
    E2m08   Bitbrains Rnd tháng 8, **dữ liệu phát triển**, dùng để đo chi phí
    E2m07   Bitbrains Rnd tháng 7, giai đoạn xác nhận
    E2m09   Bitbrains Rnd tháng 9, giai đoạn xác nhận, 29 ngày

Ba tháng của Rnd **không nối lại với nhau**: trên mẫu 60 máy chỉ 31 máy giữ nguyên cả
số nhân, xung nhịp và dung lượng RAM qua ba tháng, và giữa hai tháng liên tiếp có một
ngày trống. Chưa đủ bằng chứng để coi cùng tên tệp là cùng một máy, nên mỗi tháng là
một giai đoạn riêng và `series_id` mang phần tháng.

Quần thể máy quyết định bằng **14 ngày đầu**, tức `--n-train 4032`, đúng bằng lịch sử
tối thiểu của fold đầu tiên. Mọi fold về sau dùng đúng quần thể đó, nên không fold nào
chọn quần thể bằng dữ liệu của chính giai đoạn nó chấm.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Môi trường và số ngày của cửa sổ.
MOI_TRUONG = {"E1L": 30, "E2m08": 30, "E2m07": 30, "E2m09": 29}
NGAY_TOI_THIEU = 14
HORIZONS = "1,12"


def chay(cmd: list) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"Lệnh thoát với mã {r.returncode}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default=",".join(MOI_TRUONG))
    ap.add_argument("--bo-qua-tien-xu-ly", action="store_true")
    a = ap.parse_args()

    envs = [x.strip() for x in a.env.split(",")]
    la = [e for e in envs if e not in MOI_TRUONG]
    if la:
        raise SystemExit(f"Không biết môi trường {la}. Có: {list(MOI_TRUONG)}")

    n_train = NGAY_TOI_THIEU * 288
    t0 = time.time()
    print(f"python: {sys.executable}")
    print(f"quần thể quyết định bằng {NGAY_TOI_THIEU} ngày đầu "
          f"= {n_train} bucket")

    for e in envs:
        ngay = MOI_TRUONG[e]
        print(f"\n=== {e}: cửa sổ {ngay} ngày")
        if not a.bo_qua_tien_xu_ly:
            chay([sys.executable, "-m", "cwp.preprocess.build", "--env", e,
                  "--window-days", str(ngay), "--n-train", str(n_train)])
        chay([sys.executable, "scripts/build_features.py", "--env", e,
              "--horizons", HORIZONS])
        chay([sys.executable, "scripts/tap_dong_chung.py", "--env", e,
              "--horizons", HORIZONS, "--tuan"])

    print(f"\nXong sau {(time.time() - t0) / 60:.1f} phút.")
    print("Nhắc: E2m08 là dữ liệu phát triển. E1L, E2m07 và E2m09 chỉ chạy mô hình "
          "sau khi protocol đã khoá.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
