"""Kiểm thử R02 và R02b: họ nào buộc phải có naive (protocol v1.2).

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Điều quan trọng nhất: một lần kiểm hẹp (chỉ tn_c, không kèm tn_a) **không được** báo đạt
một cách âm thầm, vì khi đó không có gì để đối chiếu "naive tại đích".
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.quy_trinh.kiem import KHOA_HO_CHI_MUC, ho_thieu_naive  # noqa: E402


def _ci(hang):
    """Chỉ mục rút gọn: (thi_nghiem, nguon, dich, mode, h, fold, model)."""
    return pd.DataFrame(hang, columns=KHOA_HO_CHI_MUC + ["model"])


def _ho(tn, nguon, dich, models, mode="N0", h=1, fold="test"):
    return [(tn, nguon, dich, mode, h, fold, m) for m in models]


CO_SO = ["naive", "ma6", "seasonal"]
HOC = ["ridge", "xgb", "xgb_mae"]


class TestHoTaiCho(unittest.TestCase):
    def test_tai_cho_du_naive_thi_sach(self):
        r = ho_thieu_naive(_ci(_ho("tn_a", "E1", "E1", CO_SO + HOC)))
        self.assertEqual(r["tai_cho_thieu_naive"], [])
        self.assertEqual(r["thieu_tham_chieu"], [])
        self.assertEqual(r["ngoai_pham_vi"], [])

    def test_tai_cho_thieu_naive_la_loi(self):
        # đúng trường hợp máy giả trước khi sửa: E1a→E1a không có naive
        r = ho_thieu_naive(_ci(_ho("may_gia", "E1a", "E1a", HOC)))
        self.assertEqual(len(r["tai_cho_thieu_naive"]), 1)
        self.assertEqual(r["tai_cho_thieu_naive"][0][1:3], ("E1a", "E1a"))

    def test_chuyen_giao_thieu_naive_khong_phai_loi(self):
        # tn_c: họ chuyển giao không có naive là đúng thiết kế
        d = _ci(_ho("tn_a", "E2", "E2", CO_SO + HOC) + _ho("tn_c", "E1", "E2", HOC))
        r = ho_thieu_naive(d)
        self.assertEqual(r["tai_cho_thieu_naive"], [])
        self.assertEqual(r["thieu_tham_chieu"], [])
        self.assertEqual(r["ngoai_pham_vi"], [])


class TestThamChieuChuyenGiao(unittest.TestCase):
    def test_tai_cho_thieu_naive_lam_chuyen_giao_mat_tham_chieu(self):
        d = _ci(_ho("may_gia", "E1g", "E1g", HOC) + _ho("may_gia", "E1a", "E1g", HOC))
        r = ho_thieu_naive(d)
        self.assertEqual(len(r["tai_cho_thieu_naive"]), 1)      # E1g→E1g thiếu naive
        self.assertEqual(len(r["thieu_tham_chieu"]), 1)         # E1a→E1g mất tham chiếu
        self.assertEqual(r["thieu_tham_chieu"][0][1], ["naive"])

    def test_tai_cho_thieu_dung_mo_hinh_cua_chuyen_giao(self):
        # chiều tại chỗ không chạy xgb_mae thì tỉ số L của chiều chuyển giao sẽ NaN
        d = _ci(_ho("tn_a", "E2", "E2", CO_SO + ["ridge", "xgb"])
                + _ho("tn_c", "E1", "E2", HOC))
        r = ho_thieu_naive(d)
        self.assertEqual(r["tai_cho_thieu_naive"], [])
        self.assertEqual(len(r["thieu_tham_chieu"]), 1)
        self.assertEqual(r["thieu_tham_chieu"][0][1], ["xgb_mae"])

    def test_chi_kiem_tn_c_thi_bao_chua_kiem_duoc(self):
        # KHÔNG được báo đạt âm thầm: không có tn_a nên không đối chiếu được
        r = ho_thieu_naive(_ci(_ho("tn_c", "E1", "E2", HOC)))
        self.assertEqual(r["thieu_tham_chieu"], [])
        self.assertEqual(len(r["ngoai_pham_vi"]), 1)

    def test_tach_theo_che_do_tam_va_fold(self):
        # họ tại chỗ ở N0 không đỡ được họ chuyển giao ở N1
        d = _ci(_ho("tn_a", "E2", "E2", CO_SO + HOC, mode="N0")
                + _ho("tn_c", "E1", "E2", HOC, mode="N1"))
        r = ho_thieu_naive(d)
        self.assertEqual(len(r["ngoai_pham_vi"]), 1)

    def test_nhieu_fold_moi_fold_can_tham_chieu_rieng(self):
        d = _ci(_ho("tn_a", "E2", "E2", CO_SO + HOC, fold="o4032")
                + _ho("tn_c", "E1", "E2", HOC, fold="o4032")
                + _ho("tn_c", "E1", "E2", HOC, fold="o6048"))
        r = ho_thieu_naive(d)
        self.assertEqual(len(r["ngoai_pham_vi"]), 1)
        self.assertEqual(r["ngoai_pham_vi"][0][5], "o6048")


class TestBienGioiHan(unittest.TestCase):
    def test_chi_muc_rong(self):
        r = ho_thieu_naive(pd.DataFrame(columns=KHOA_HO_CHI_MUC + ["model"]))
        self.assertEqual(r, {"tai_cho_thieu_naive": [], "thieu_tham_chieu": [],
                             "ngoai_pham_vi": []})

    def test_none(self):
        self.assertEqual(ho_thieu_naive(None)["ngoai_pham_vi"], [])

    def test_bon_chieu_may_gia_sau_khi_sua_thi_sach(self):
        # đúng cấu hình K7.3 sau bản sửa: chiều tại chỗ có mô hình cơ sở
        d = _ci(_ho("may_gia", "E1a", "E1a", HOC + CO_SO)
                + _ho("may_gia", "E1g", "E1g", HOC + CO_SO)
                + _ho("may_gia", "E1a", "E1g", HOC)
                + _ho("may_gia", "E1g", "E1a", HOC))
        r = ho_thieu_naive(d)
        self.assertEqual(r["tai_cho_thieu_naive"], [])
        self.assertEqual(r["thieu_tham_chieu"], [])
        self.assertEqual(r["ngoai_pham_vi"], [])


if __name__ == "__main__":
    unittest.main()
