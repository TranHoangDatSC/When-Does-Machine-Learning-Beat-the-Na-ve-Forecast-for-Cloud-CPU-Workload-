"""Kiểm thử các phép kiểm của protocol v1.1. Chạy bằng .venv của repo này:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Chỉ dùng dữ liệu tổng hợp trong bộ nhớ hoặc thư mục tạm; không đọc, không ghi `data/` hay `results/`.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.quy_trinh.kiem import (KHOA_CV, doc_cv_search, khu_trung_cv,  # noqa: E402
                         kiem_huu_han_trong_tap_chung, y_khop)


class TestYKhop(unittest.TestCase):
    """R06 và N04: dung sai float32, vẫn bắt sai lệch thật."""

    def test_lech_hien_tai_toi_da_3p7e6_duoc_chap_nhan(self):
        that = np.array([0.0, 1.5, 37.25, 50.0, 99.999])
        y = that + np.array([0.0, 3.7e-6, -3.7e-6, 3.7e-6, 3.7e-6])
        dat, lech, thieu = y_khop(that, y)
        self.assertTrue(dat)
        self.assertLess(lech, 3.8e-6)
        self.assertEqual(thieu, 0)

    def test_y_ep_float32_that_duoc_chap_nhan(self):
        rng = np.random.default_rng(0)
        that = rng.uniform(0, 100, 10_000)
        y = that.astype("float32").astype("float64")
        dat, lech, _ = y_khop(that, y)
        self.assertTrue(dat, f"lệch {lech:.3e}")

    def test_nguong_cu_1e9_bat_nham_float32(self):
        # Chứng minh vì sao phải đổi: với float32, ngưỡng tuyệt đối 1e-9 báo hỏng.
        that = np.array([37.123456789])
        y = that.astype("float32").astype("float64")
        self.assertGreater(float(np.abs(that - y).max()), 1e-9)
        self.assertTrue(y_khop(that, y)[0])

    def test_lech_1e3_van_bi_bat(self):
        that = np.array([10.0, 20.0, 30.0])
        y = that + np.array([0.0, 1e-3, 0.0])
        dat, lech, _ = y_khop(that, y)
        self.assertFalse(dat)
        self.assertAlmostEqual(lech, 1e-3, places=9)

    def test_lech_o_gia_tri_lon_van_bi_bat(self):
        dat, _, _ = y_khop(np.array([90.0]), np.array([90.0 + 1e-3]))
        self.assertFalse(dat)

    def test_diem_khong_co_quan_sat_bi_bat(self):
        # dự đoán ghi y hữu hạn nhưng data/processed không có quan sát ở đó
        dat, _, thieu = y_khop(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
        self.assertFalse(dat)
        self.assertEqual(thieu, 1)

    def test_do_dai_lech_bao_loi(self):
        with self.assertRaises(ValueError):
            y_khop(np.zeros(3), np.zeros(4))


def _du_doan(hang):
    """hang: (model, seed, series_id, bucket_phat, y, yhat)"""
    return pd.DataFrame(hang, columns=["model", "seed", "series_id", "bucket_phat", "y", "yhat"])


class TestR04TapChung(unittest.TestCase):
    def setUp(self):
        # 2 mô hình × 4 dòng; dòng (s, 4) là dòng seasonal không dự đoán được
        rows = []
        for m in ("naive", "seasonal"):
            for t in (1, 2, 3, 4):
                yhat = np.nan if (m == "seasonal" and t == 4) else 1.0
                rows.append((m, 42, "s", t, 1.0, yhat))
        self.g = _du_doan(rows)

    def test_nan_ngoai_tap_chung_khong_phai_loi(self):
        r = kiem_huu_han_trong_tap_chung(self.g, {("s", 1), ("s", 2), ("s", 3)})
        self.assertEqual(r["n_hong_trong"], 0)
        self.assertEqual(r["n_hong_ngoai"], 1)
        self.assertEqual(r["n_trong_moi_nhom"], (3, 3))

    def test_nan_trong_tap_chung_la_loi(self):
        r = kiem_huu_han_trong_tap_chung(self.g, {("s", 1), ("s", 2), ("s", 3), ("s", 4)})
        self.assertEqual(r["n_hong_trong"], 1)

    def test_y_khong_huu_han_trong_tap_chung_la_loi(self):
        g = self.g.copy()
        g.loc[(g.model == "naive") & (g.bucket_phat == 2), "y"] = np.inf
        r = kiem_huu_han_trong_tap_chung(g, {("s", 1), ("s", 2), ("s", 3)})
        self.assertEqual(r["n_hong_trong"], 1)

    def test_mo_hinh_khong_cung_tap_dong_bi_bao(self):
        g = self.g[~((self.g.model == "seasonal") & (self.g.bucket_phat == 3))]
        r = kiem_huu_han_trong_tap_chung(g, {("s", 1), ("s", 2), ("s", 3)})
        self.assertEqual(r["n_trong_moi_nhom"], (2, 3))

    def test_tap_chung_rong(self):
        r = kiem_huu_han_trong_tap_chung(self.g, set())
        self.assertEqual(r["n_hong_trong"], 0)
        self.assertEqual(r["n_trong_moi_nhom"], (0, 0))


def _cv(n_lan=1, lech=0.0, **ghi_de):
    """Bảng cv_search tổng hợp: 2 ứng viên × 3 fold, lặp lại `n_lan` lần."""
    dong = []
    for lan in range(n_lan):
        for uv in ("a", "b"):
            for f in range(3):
                dong.append(dict(env="E1", h=1, mode="N0", model="xgb", ung_vien=uv, fold=f,
                                 n_train=100 + f, n_val=10, mae_p50_val=1.0 + f + (lech if lan else 0.0),
                                 giay=float(lan)))
    df = pd.DataFrame(dong)
    for k, v in ghi_de.items():
        df.loc[df.index[-1], k] = v
    return df


class TestCvSearch(unittest.TestCase):
    def test_khu_trung_ban_giong_nhau(self):
        cv = _cv(n_lan=5)
        out = khu_trung_cv(cv)
        self.assertEqual(len(cv), 30)
        self.assertEqual(len(out), 6)
        self.assertFalse(out.duplicated(KHOA_CV).any())

    def test_ban_trung_khac_4e16_van_duoc(self):
        out = khu_trung_cv(_cv(n_lan=3, lech=4.4e-16))
        self.assertEqual(len(out), 6)

    def test_ban_trung_khac_qua_dung_sai_bao_loi(self):
        with self.assertRaises(ValueError):
            khu_trung_cv(_cv(n_lan=2, lech=1e-6))

    def test_n_train_khac_nhau_bao_loi(self):
        cv = _cv(n_lan=2, n_train=999)
        with self.assertRaises(ValueError):
            khu_trung_cv(cv)

    def test_khong_trung_thi_giu_nguyen(self):
        self.assertEqual(len(khu_trung_cv(_cv(n_lan=1))), 6)

    def test_doc_tu_tep_va_thieu_cot(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cv_search.csv"
            _cv(n_lan=4).to_csv(p, index=False)
            self.assertEqual(len(doc_cv_search(p)), 6)
            _cv(n_lan=1).drop(columns=["n_val"]).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                doc_cv_search(p)


class TestGhiCvSearchKhongTrung(unittest.TestCase):
    """Bộ ghi: gọi `ghi_dan` nhiều lần với nhật ký tích luỹ không được ghi nối trùng."""

    def test_ghi_nhieu_lan_khong_nhan_ban(self):
        import tn_a as tn
        cv_log = []
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for mo_hinh in range(4):                    # như vòng lặp mô hình của tn_a
                for f in range(3):
                    cv_log.append(dict(env="E1", h=1, mode="N0", model=f"m{mo_hinh}",
                                       ung_vien="a", fold=f, n_train=100, n_val=10,
                                       mae_p50_val=1.0, giay=0.1))
                tn.ghi_dan(out, [], cv_log, [])          # ghi sau mỗi mô hình
            cv = pd.read_csv(out / "cv_search.csv")
            self.assertEqual(len(cv), 12)
            self.assertFalse(cv.duplicated(KHOA_CV).any())

    def test_resume_khong_nhan_ban_du_lieu_cu(self):
        import tn_a as tn
        moi = [dict(env="E1", h=1, mode="N0", model="m0", ung_vien="a", fold=f, n_train=100,
                    n_val=10, mae_p50_val=1.0, giay=0.1) for f in range(3)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            tn.ghi_dan(out, [], moi, [])
            tn.ghi_dan(out, [], moi, [])                 # lần chạy --resume sau ghi lại cùng khoá
            self.assertEqual(len(pd.read_csv(out / "cv_search.csv")), 3)

    def test_chosen_csv_van_khu_trung_theo_khoa_cu(self):
        import tn_a as tn
        chon = [dict(env="E1", mode="N0", h=1, model="xgb", sieu_tham_so="{}",
                     mae_p50_val=1.0, n_train=5, giay=1.0)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            tn.ghi_dan(out, [], [], chon)
            tn.ghi_dan(out, [], [], chon)
            self.assertEqual(len(pd.read_csv(out / "chosen.csv")), 1)


class TestTiSoTheoMay(unittest.TestCase):
    """Tỉ số theo máy: máy có mẫu số 0 bị loại và được đếm, không làm NaN cả trung vị."""

    def setUp(self):
        import tong_hop as th
        self.f = th.ti_so_theo_may

    def test_khong_mau_so_0_thi_nhu_cu(self):
        ti, n = self.f([2.0, 3.0, 4.0], [1.0, 2.0, 4.0])
        np.testing.assert_allclose(ti, [2.0, 1.5, 1.0])
        self.assertEqual(n, 0)

    def test_0_chia_0_bi_loai_va_trung_vi_van_huu_han(self):
        a = np.array([0.0, 2.0, 3.0, 4.0]); b = np.array([0.0, 1.0, 2.0, 4.0])
        ti, n = self.f(a, b)
        self.assertEqual(n, 1)
        self.assertTrue(np.isfinite(np.median(ti)))

    def test_duong_chia_0_la_vo_han_bi_loai(self):
        with np.errstate(divide="ignore"):
            ti, n = self.f([5.0, 2.0], [0.0, 1.0])
        np.testing.assert_allclose(ti, [2.0]); self.assertEqual(n, 1)

    def test_tu_so_0_mau_so_duong_la_hop_le(self):
        ti, n = self.f([0.0, 1.0], [1.0, 1.0])
        np.testing.assert_allclose(ti, [0.0, 1.0]); self.assertEqual(n, 0)

    def test_nan_dau_vao_bi_loai(self):
        ti, n = self.f([np.nan, 2.0], [1.0, 1.0])
        np.testing.assert_allclose(ti, [2.0]); self.assertEqual(n, 1)

    def test_tat_ca_bi_loai(self):
        ti, n = self.f([0.0, 0.0], [0.0, 0.0])
        self.assertEqual((len(ti), n), (0, 2))


if __name__ == "__main__":
    unittest.main()
