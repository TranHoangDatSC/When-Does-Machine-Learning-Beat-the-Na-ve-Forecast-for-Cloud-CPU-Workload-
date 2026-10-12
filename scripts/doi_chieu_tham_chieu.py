"""Đối chiếu kết quả bạn vừa chạy với kết quả tham chiếu kèm trong kho mã.

    python scripts/doi_chieu_tham_chieu.py
    python scripts/doi_chieu_tham_chieu.py --dung-sai 1e-9
    python scripts/doi_chieu_tham_chieu.py --chi tong_hop

Mọi bảng trong `ket_qua_tham_chieu/` được so với bảng cùng đường dẫn trong `results/`.
Script này **chỉ đọc**, không sửa gì.

Vì sao cần. Kho mã kèm sẵn bảng kết quả để người đọc khỏi phải chạy 40 giờ mới thấy con số
đầu tiên. Nhưng bảng kèm sẵn chỉ đáng tin khi nó **sinh lại được**. Chạy lại đường chạy rồi
gọi script này là cách kiểm: nếu mọi bảng trùng trong dung sai thì con số trong bài không
phải do chỉnh tay.

Cách so
-------

* Cột số thực: `np.isclose` với `--dung-sai` (mặc định 1e-9, tương đối và tuyệt đối).
  Vài chỉ số đi qua bootstrap nên có thể lệch ở chữ số cuối giữa hai lần chạy; dung sai
  mặc định vẫn bắt được mọi khác biệt có nghĩa.
* Cột chữ và cột nguyên: phải bằng nhau tuyệt đối.
* Số dòng, tên cột, và tập khoá phải trùng.
* `meta.json` chỉ so những trường không phụ thuộc máy chạy: cấu hình, phạm vi, hạt giống.
  Không so thời gian chạy, đường dẫn Python, hay ngày tháng.
* Vài cột ghi lại **hoàn cảnh chạy** chứ không phải kết quả, và chắc chắn khác nhau giữa
  hai máy: `giay` (thời gian chạy), `tep_du_doan` (đường dẫn), `van_tay_ma`,
  `van_tay_cau_hinh`, `van_tay_du_lieu`. So chúng thì lần đối chiếu nào cũng báo lệch dù
  mọi con số đều trùng, nên chúng bị bỏ qua. Script in ra đúng những cột đã bỏ, để việc
  bỏ qua không thành chỗ giấu khác biệt. Mọi cột còn lại đều được so.

Mã thoát 0 khi mọi bảng trùng, 1 khi có khác biệt, 2 khi thiếu tệp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Trường của meta.json phụ thuộc máy chạy hoặc thời điểm, không so.
BO_QUA_META = {"ngay", "python", "giay", "argv", "thiet_bi", "torch", "phien_ban",
               "van_tay", "luc", "host", "may"}

# Cột của bảng CSV phụ thuộc máy chạy chứ không phải kết quả. So chúng thì lần đối chiếu
# nào cũng báo lệch, kể cả khi mọi con số đều trùng. Script in rõ đã bỏ qua cột nào, để
# việc bỏ qua này không thành chỗ giấu khác biệt.
BO_QUA_COT = {
    "giay": "thời gian chạy, khác nhau theo máy",
    "tep_du_doan": "đường dẫn tệp dự đoán",
    "van_tay_ma": "băm mã nguồn; đổi khi tên hoặc nội dung tệp mã đổi",
    "van_tay_cau_hinh": "băm tệp cấu hình",
    "van_tay_du_lieu": "băm theo kích thước và thời điểm sửa của dữ liệu vào",
    "ngay": "thời điểm chạy",
    "built_on": "tên máy đã dựng dữ liệu",
}


def _so_bang(a: pd.DataFrame, b: pd.DataFrame, dung_sai: float,
             da_bo: set | None = None) -> list:
    """Danh sách khác biệt giữa hai bảng; rỗng nghĩa là trùng."""
    loi = []
    bo = [c for c in a.columns if c in BO_QUA_COT]
    if bo:
        if da_bo is not None:
            da_bo.update(bo)
        a = a.drop(columns=bo)
        b = b.drop(columns=[c for c in bo if c in b.columns])
    if list(a.columns) != list(b.columns):
        thieu = [c for c in a.columns if c not in b.columns]
        thua = [c for c in b.columns if c not in a.columns]
        if thieu or thua:
            return [f"cột lệch: thiếu {thieu[:4]}, thừa {thua[:4]}"]
        loi.append("thứ tự cột khác")
        b = b[list(a.columns)]
    if len(a) != len(b):
        return [f"số dòng lệch: tham chiếu {len(a):,}, của bạn {len(b):,}"]
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
            xv = pd.to_numeric(x, errors="coerce").to_numpy("float64")
            yv = pd.to_numeric(y, errors="coerce").to_numpy("float64")
            xau = ~(np.isclose(xv, yv, rtol=dung_sai, atol=dung_sai)
                    | (np.isnan(xv) & np.isnan(yv)))
            if xau.any():
                i = int(np.argmax(xau))
                loi.append(f"cột {c}: {int(xau.sum())} ô lệch, ví dụ dòng {i} "
                           f"tham chiếu {xv[i]!r} của bạn {yv[i]!r}")
        else:
            xau = (x.astype(str) != y.astype(str)).to_numpy()
            if xau.any():
                i = int(np.argmax(xau))
                loi.append(f"cột {c}: {int(xau.sum())} ô lệch, ví dụ dòng {i} "
                           f"tham chiếu {x.iloc[i]!r} của bạn {y.iloc[i]!r}")
    return loi


def _so_meta(a: dict, b: dict) -> list:
    loi = []
    for k in sorted(set(a) | set(b)):
        if k in BO_QUA_META:
            continue
        if k not in b:
            loi.append(f"thiếu trường {k!r}")
        elif k not in a:
            continue
        elif a[k] != b[k]:
            loi.append(f"trường {k!r}: tham chiếu {a[k]!r}, của bạn {b[k]!r}")
    return loi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tham-chieu", default="ket_qua_tham_chieu")
    ap.add_argument("--cua-ban", default="results")
    ap.add_argument("--dung-sai", type=float, default=1e-9)
    ap.add_argument("--chi", default=None,
                    help="chỉ so những đường dẫn chứa chuỗi này, ví dụ tong_hop")
    a = ap.parse_args()

    tc, cb = ROOT / a.tham_chieu, ROOT / a.cua_ban
    if not tc.exists():
        print(f"Thiếu {tc}.")
        return 2

    # `van_tay_du_lieu.json` là bản tham chiếu của dữ liệu trung gian, không phải bảng kết
    # quả; nó có script kiểm riêng là `van_tay_du_lieu.py`.
    tep = [p for p in sorted(tc.rglob("*"))
           if p.suffix in (".csv", ".json") and p.name != "van_tay_du_lieu.json"]
    if a.chi:
        tep = [p for p in tep if a.chi in p.relative_to(tc).as_posix()]
    if not tep:
        print("Không có tệp nào để so.")
        return 2

    def _hien(q: Path) -> str:
        try:
            return q.relative_to(ROOT).as_posix()
        except ValueError:                      # cho phép trỏ ra ngoài kho mã
            return q.as_posix()

    print(f"Đối chiếu {len(tep)} bảng · dung sai {a.dung_sai:g}")
    print(f"  tham chiếu : {_hien(tc)}")
    print(f"  của bạn    : {_hien(cb)}\n")

    trung, khac, thieu, da_bo = [], [], [], set()
    for p in tep:
        rel = p.relative_to(tc)
        q = cb / rel
        ten = rel.as_posix()
        if not q.exists():
            thieu.append(ten)
            print(f"  [THIẾU ] {ten}")
            continue
        try:
            if p.suffix == ".json":
                loi = _so_meta(json.loads(p.read_text(encoding="utf-8")),
                               json.loads(q.read_text(encoding="utf-8")))
            else:
                loi = _so_bang(pd.read_csv(p), pd.read_csv(q), a.dung_sai, da_bo)
        except Exception as e:                       # tệp hỏng cũng là khác biệt
            loi = [f"không đọc được: {e}"]
        if loi:
            khac.append(ten)
            print(f"  [KHÁC  ] {ten}")
            for d in loi[:4]:
                print(f"            {d}")
        else:
            trung.append(ten)
            print(f"  [trùng ] {ten}")

    print(f"\n{len(trung)}/{len(tep)} bảng trùng · {len(khac)} khác · {len(thieu)} thiếu")
    if da_bo:
        print("Cột không so vì phụ thuộc máy chạy chứ không phải kết quả:")
        for c in sorted(da_bo):
            print(f"   {c:20} {BO_QUA_COT[c]}")
    if thieu:
        print("Tệp thiếu thường chỉ là bước bạn chưa chạy, không phải sai lệch.")
    if khac:
        print("Có bảng khác tham chiếu. Xem chi tiết ở trên trước khi dùng số.")
        return 1
    if not trung:
        return 2
    print("Mọi bảng đã chạy đều trùng kết quả tham chiếu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
