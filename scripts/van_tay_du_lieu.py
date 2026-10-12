"""Vân tay nội dung của dữ liệu trung gian, để kiểm bước dựng dữ liệu mà không chạy mô hình.

    python scripts/van_tay_du_lieu.py                      # đối chiếu với bản tham chiếu
    python scripts/van_tay_du_lieu.py --ghi <tệp.json>     # ghi vân tay của máy bạn

Vì sao có script này. Chạy trọn đường ống mất khoảng 47 giờ, gần hết là huấn luyện mô hình.
Nhưng phần **dựng dữ liệu** chỉ mất khoảng 30 phút, và nếu dữ liệu bạn dựng ra trùng dữ liệu
chúng tôi dựng thì mọi thứ phía sau chạy trên cùng một đầu vào. Script này cho bạn kiểm điều
đó mà không phải bỏ ra hai ngày máy.

Vân tay là gì. Với mỗi tệp `.parquet`, script ghi số dòng, danh sách cột, và một mã băm
SHA-256 tính từ **nội dung từng cột** chứ không từ byte của tệp. Cách này bền với khác biệt
về nén, về cách chia row-group, và về phiên bản thư viện parquet — những thứ đổi byte của
tệp mà không đổi dữ liệu. Băm byte của tệp cũng được ghi kèm, nhưng chỉ để tham khảo: hai
máy cho ra byte khác nhau vẫn có thể có dữ liệu giống hệt.

Script chỉ đọc, không sửa gì. Mã thoát 0 khi mọi tệp đã dựng đều trùng nội dung.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Thư mục dữ liệu trung gian do đường ống sinh ra. `data/raw` không có ở đây vì nó là dữ
# liệu tải về, không phải thứ đường ống tạo.
THU_MUC = ["processed", "features", "eval_rows", "k0",
           "nguong_0p0", "nguong_0p5", "nguong_2p0"]
TEP_LE = ["catalog.parquet"]


def van_tay_mot_tep(p: Path) -> dict:
    """Số dòng, cột, và băm nội dung theo từng cột.

    Đọc từng cột một để không phải giữ cả bảng trong bộ nhớ: tệp đặc trưng lớn nhất khoảng
    6,2 triệu dòng × 23 cột.
    """
    lo = pq.ParquetFile(p)
    cot = list(lo.schema_arrow.names)
    h = hashlib.sha256()
    h.update(str(len(cot)).encode())
    for c in cot:
        h.update(c.encode())
        s = pq.read_table(p, columns=[c]).column(0).to_pandas()
        # hash_pandas_object băm theo giá trị, không theo cách lưu trên đĩa
        h.update(pd.util.hash_pandas_object(s, index=False).to_numpy().tobytes())
    return {"n_dong": lo.metadata.num_rows, "cot": cot, "noi_dung": h.hexdigest()}


def _sha_byte(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest()


def quet(goc: Path, kem_byte: bool = True) -> dict:
    ra = {}
    muc = [goc / t for t in TEP_LE] + [goc / d for d in THU_MUC]
    for m in muc:
        if not m.exists():
            continue
        tep = [m] if m.is_file() else sorted(m.rglob("*.parquet"))
        for p in tep:
            rel = p.relative_to(goc).as_posix()
            v = van_tay_mot_tep(p)
            if kem_byte:
                v["byte"] = _sha_byte(p)
            ra[rel] = v
            print(f"  {rel:46} {v['n_dong']:>10,} dòng")
    return ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--du-lieu", default="data")
    ap.add_argument("--tham-chieu", default="ket_qua_tham_chieu/van_tay_du_lieu.json")
    ap.add_argument("--ghi", default=None, help="ghi vân tay ra tệp thay vì đối chiếu")
    a = ap.parse_args()

    goc = ROOT / a.du_lieu
    if not goc.exists():
        print(f"Thiếu {goc}. Xem README, mục lấy dữ liệu thô và chạy khối K2.")
        return 2

    def _hien(q: Path) -> str:
        try:
            return q.relative_to(ROOT).as_posix()
        except ValueError:                      # cho phép trỏ ra ngoài kho mã
            return q.as_posix()

    print(f"Quét {_hien(goc)} ...")
    hien = quet(goc)
    print(f"{len(hien)} tệp.\n")

    if a.ghi:
        q = ROOT / a.ghi
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(hien, ensure_ascii=False, indent=1, sort_keys=True),
                     encoding="utf-8")
        print(f"Đã ghi {_hien(q)}")
        return 0

    tc = ROOT / a.tham_chieu
    if not tc.exists():
        print(f"Thiếu bản tham chiếu {tc}.")
        return 2
    goc_tc = json.loads(tc.read_text(encoding="utf-8"))

    trung, khac, thieu, byte_khac = [], [], [], 0
    for ten, v in sorted(goc_tc.items()):
        if ten not in hien:
            thieu.append(ten)
            continue
        m = hien[ten]
        if m["noi_dung"] == v["noi_dung"]:
            trung.append(ten)
            if v.get("byte") and m.get("byte") and m["byte"] != v["byte"]:
                byte_khac += 1
        else:
            khac.append(ten)
            print(f"  [KHÁC ] {ten}")
            if m["n_dong"] != v["n_dong"]:
                print(f"           số dòng: tham chiếu {v['n_dong']:,}, của bạn {m['n_dong']:,}")
            if m["cot"] != v["cot"]:
                print(f"           cột lệch")

    print(f"{len(trung)}/{len(goc_tc)} tệp trùng nội dung · {len(khac)} khác · "
          f"{len(thieu)} chưa dựng")
    if byte_khac:
        print(f"{byte_khac} tệp trùng nội dung nhưng khác byte: bình thường, do phiên bản "
              f"thư viện parquet ghi khác nhau.")
    if thieu:
        print("Tệp chưa dựng là bước bạn chưa chạy, không phải sai lệch.")
    if khac:
        print("Có tệp khác nội dung. Dữ liệu đầu vào của bạn khác chúng tôi; đừng so kết "
              "quả mô hình trước khi hiểu vì sao.")
        return 1
    if not trung:
        return 2
    print("Mọi tệp đã dựng đều trùng nội dung bản tham chiếu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
