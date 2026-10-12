"""Nạp dự đoán theo từng họ so sánh, không nạp cả kho vào bộ nhớ (protocol v1.2).

Thí nghiệm B có khoảng 162 triệu dòng dự đoán (1,2 GB parquet nén). Nạp hết vào một
`DataFrame` cần trên 10 GB chỉ riêng các cột chuỗi, nên `pd.read_parquet(<thư mục>)`
làm cả `tong_hop.py` lẫn `kiem_ket_qua.py` hết bộ nhớ.

Mỗi tệp dự đoán do `ghi_du_doan` viết ra chỉ chứa **một** tổ hợp
`(nguon, dich, mode, lich, model, loss, h, fold, seed)`, nên tệp nào cũng nằm trọn
trong một họ so sánh. Vì vậy có thể lập chỉ mục trước rồi nạp từng họ một.

`chi_muc_du_doan` đọc khoá từ hàng đầu của mỗi tệp và số dòng từ metadata parquet, nên
không phải đọc dữ liệu. Thứ tự tệp là thứ tự tên tệp đã sắp, trùng thứ tự mà
`pd.read_parquet(<thư mục>)` trả về, nên thứ tự duyệt họ **không đổi** so với bản cũ:
điều này cần thiết vì bộ sinh số ngẫu nhiên của bootstrap dùng chung cho mọi họ.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Khoá của một tệp dự đoán: đọc từ hàng đầu tiên.
COT_KHOA = ["nguon", "dich", "mode", "lich", "model", "loss", "h", "fold", "seed"]
# Cột mà `mot_ho` của tong_hop.py thực sự dùng khi chấm.
COT_CHAM = ["series_id", "bucket_phat", "model", "seed", "y", "yhat"]
# Thêm `bucket_muc_tieu` khi cần đối chiếu với `data/processed` (R06, N04).
COT_KIEM = COT_CHAM + ["bucket_muc_tieu"]


def _khoa_cua_tep(f: Path) -> dict:
    """Khoá và số dòng của một tệp dự đoán, đọc metadata và đúng một hàng."""
    pf = pq.ParquetFile(f)
    co = [c for c in COT_KHOA if c in pf.schema_arrow.names]
    hang = pf.read_row_group(0, columns=co).slice(0, 1).to_pylist()
    if not hang:
        return {}
    return {"n_dong": pf.metadata.num_rows, **hang[0]}


def chi_muc_tep(thu_muc: Path) -> pd.DataFrame:
    """Chỉ mục của **một** thư mục `du_doan/`: một dòng cho mỗi tệp, kèm khoá và số dòng."""
    thu_muc = Path(thu_muc)
    if not thu_muc.exists():
        return pd.DataFrame()
    hang = []
    for f in sorted(thu_muc.glob("*.parquet")):
        k = _khoa_cua_tep(f)
        if k:
            hang.append({"tep": f, **k})
    return pd.DataFrame(hang)


def chi_muc_du_doan(goc: Path, tns) -> pd.DataFrame:
    """Một dòng cho mỗi tệp dự đoán của các thí nghiệm `tns`.

    Cột: `tep`, `thu_muc` (thư mục gốc), `thi_nghiem` (họ so sánh), `n_dong`, và các
    cột khoá. Dự đoán của mô hình chuỗi nằm ở `dl/` nhưng thuộc họ `tn_a` khi chấm tại
    chỗ và `tn_c` khi chuyển giao, giống hệt quy tắc cũ của `tong_hop.py`.
    """
    goc = Path(goc)
    hang = []
    for tn in tns:
        p = goc / tn / "du_doan"
        if not p.exists():
            continue
        for f in sorted(p.glob("*.parquet")):
            k = _khoa_cua_tep(f)
            if not k:
                continue
            ho = tn
            if tn == "dl":
                ho = "tn_a" if k.get("nguon") == k.get("dich") else "tn_c"
            hang.append({"tep": f, "thu_muc": tn, "thi_nghiem": ho, **k})
    if not hang:
        return pd.DataFrame()
    return pd.DataFrame(hang)


def nap_ho(teps, cot=None) -> pd.DataFrame:
    """Nạp các tệp của **một** họ, chỉ những cột cần, `series_id` ở dạng category.

    `teps` phải giữ đúng thứ tự của chỉ mục để thứ tự dòng trùng bản cũ.
    """
    cot = list(COT_CHAM if cot is None else cot)
    phan = [pd.read_parquet(t, columns=cot) for t in teps]
    d = pd.concat(phan, ignore_index=True) if len(phan) > 1 else phan[0]
    return d


__all__ = ["COT_KHOA", "COT_CHAM", "COT_KIEM", "chi_muc_tep", "chi_muc_du_doan",
           "nap_ho"]
