"""Đóng gói dữ liệu tối thiểu để chạy GRU trên Colab.

    python scripts/xuat_goi_colab.py

Gói gồm đúng những thứ `scripts/dl.py` cần, không có ma trận đặc trưng:

    processed/{E1,E2}.parquet          bảng dài, cột series_id, bucket, y
    eval_rows/{E1,E2}_h{1,12}.parquet  tập dòng đánh giá chung
    dl.py                            chính script chạy, không sửa gì
    pham_vi_gru.json                    bốn tổ hợp và hạt giống đã chốt

Phạm vi GRU chốt trước khi xem kết quả: hai môi trường E1 và E2, hai tầm dự báo 1 và
12, chế độ N1, hạt giống 42. Bốn tổ hợp, không hơn.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENVS = ["E1", "E2"]
HORIZONS = [1, 12]
PHAM_VI = {
    "models": ["gru"],
    "envs": ENVS,
    "horizons": HORIZONS,
    "mode": "N1",
    "seeds": [42],
    "L": 288,
    "epochs": 30,
    "patience": 5,
    "batch": 1024,
    "lr": 0.001,
    "so_to_hop": len(ENVS) * len(HORIZONS),
    "ghi_chu": "Phạm vi chốt ngày 23/09/2026, trước khi xem bất kỳ kết quả nào.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/colab/goi_colab.zip")
    a = ap.parse_args()

    dich = ROOT / a.out
    dich.parent.mkdir(parents=True, exist_ok=True)

    can = []
    for e in ENVS:
        can.append((ROOT / "data" / "processed" / f"{e}.parquet",
                    f"processed/{e}.parquet"))
        for h in HORIZONS:
            can.append((ROOT / "data" / "eval_rows" / f"{e}_h{h}.parquet",
                        f"eval_rows/{e}_h{h}.parquet"))
    can.append((ROOT / "scripts" / "dl.py", "dl.py"))

    thieu = [str(p) for p, _ in can if not p.exists()]
    if thieu:
        raise SystemExit("Thiếu tệp: " + ", ".join(thieu))

    with zipfile.ZipFile(dich, "w", zipfile.ZIP_DEFLATED) as z:
        for p, ten in can:
            z.write(p, ten)
        z.writestr("pham_vi_gru.json",
                   json.dumps(PHAM_VI, ensure_ascii=False, indent=2))

    mb = dich.stat().st_size / 1e6
    print(f"Đã ghi {dich} ({mb:.1f} MB)")
    print(f"Gồm {len(can) + 1} tệp. Tải tệp này lên Colab theo "
          "notebooks/colab_gru.ipynb.")
    print(f"Phạm vi đã chốt: {PHAM_VI['so_to_hop']} tổ hợp, "
          f"seed {PHAM_VI['seeds']}, chế độ {PHAM_VI['mode']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
