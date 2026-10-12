"""Lõi dùng chung cho ba thí nghiệm.

`cwp.quy_trinh.chay` gom phần lặp lại giữa Thí nghiệm A, B và C: nạp ma trận đặc trưng, đưa
về thang chuẩn hoá, dò siêu tham số bằng rolling-origin, khớp mô hình cuối, chấm và
ghi dự đoán từng dòng. Ba script `scripts/tn_a.py`, `tn_b.py`, `tn_c.py` chỉ
còn phần xếp tổ hợp và in báo cáo.
"""

from cwp.quy_trinh.chay import (
    KetQua,
    chuan_bi,
    chon_sieu_tham_so,
    khop_va_cham,
    thong_ke_n1,
)

__all__ = ["KetQua", "chuan_bi", "chon_sieu_tham_so", "khop_va_cham", "thong_ke_n1"]
