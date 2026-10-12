"""Kiểm thử bộ nạp dự đoán theo từng họ (protocol v1.2).

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Điều quan trọng nhất được kiểm ở đây: **thứ tự duyệt họ** của chỉ mục phải trùng thứ tự
của bản cũ (nạp cả kho rồi `groupby(sort=False)`). Bootstrap dùng chung một bộ sinh số
ngẫu nhiên cho mọi họ, nên đổi thứ tự duyệt là đổi khoảng tin cậy.

Chỉ dùng tệp parquet tổng hợp trong thư mục tạm; không đọc `data/` hay `results/`.
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

from cwp.quy_trinh.nap import (COT_CHAM, COT_KIEM, chi_muc_du_doan,  # noqa: E402
                        nap_ho)

COT = ["nguon", "dich", "mode", "lich", "model", "loss", "h", "fold", "seed",
       "series_id", "bucket_phat", "bucket_muc_tieu", "y", "yhat"]


def _tep(thu_muc: Path, *, nguon, dich, mode, model, h, fold, seed=42, n=5,
         loss="khong", lich="co"):
    """Một tệp dự đoán tổng hợp, đúng 14 cột như `ghi_du_doan` viết ra."""
    thu_muc.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame({
        "nguon": nguon, "dich": dich, "mode": mode, "lich": lich,
        "model": model, "loss": loss, "h": h, "fold": fold, "seed": seed,
        "series_id": [f"s{i % 2}" for i in range(n)],
        "bucket_phat": np.arange(n) + 100,
        "bucket_muc_tieu": np.arange(n) + 100 + h,
        "y": np.linspace(1.0, 2.0, n),
        "yhat": np.linspace(1.1, 2.1, n),
    })[COT]
    p = thu_muc / f"{nguon}-{dich}_{mode}_{lich}_{model}_{loss}_h{h}_{fold}.parquet"
    d.to_parquet(p, index=False)
    return p


class TestChiMuc(unittest.TestCase):
    def test_mot_dong_moi_tep_va_dung_so_dong(self):
        with tempfile.TemporaryDirectory() as t:
            goc = Path(t)
            _tep(goc / "tn_a" / "du_doan", nguon="E1", dich="E1", mode="N0",
                 model="naive", h=1, fold="test", n=7)
            _tep(goc / "tn_a" / "du_doan", nguon="E1", dich="E1", mode="N0",
                 model="xgb", h=1, fold="test", n=7)
            i = chi_muc_du_doan(goc, ["tn_a"])
            self.assertEqual(len(i), 2)
            self.assertEqual(sorted(i["model"]), ["naive", "xgb"])
            self.assertTrue((i["n_dong"] == 7).all())
            self.assertTrue((i["thi_nghiem"] == "tn_a").all())

    def test_thu_muc_thieu_tra_bang_rong(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(chi_muc_du_doan(Path(t), ["tn_a", "tn_b"]).empty)

    def test_dl_tai_cho_vao_tn_a_chuyen_giao_vao_tn_c(self):
        with tempfile.TemporaryDirectory() as t:
            goc = Path(t)
            _tep(goc / "dl" / "du_doan", nguon="E1", dich="E1", mode="N1",
                 model="dlinear", h=1, fold="test")
            _tep(goc / "dl" / "du_doan", nguon="E1", dich="E2", mode="N1",
                 model="dlinear", h=1, fold="test")
            i = chi_muc_du_doan(goc, ["dl"]).sort_values("dich")
            self.assertEqual(list(i["thi_nghiem"]), ["tn_a", "tn_c"])
            self.assertTrue((i["thu_muc"] == "dl").all())

    def test_khong_doc_du_lieu_chi_doc_khoa(self):
        # chỉ mục phải có đúng các cột khoá cần cho họ so sánh
        with tempfile.TemporaryDirectory() as t:
            goc = Path(t)
            _tep(goc / "tn_b" / "du_doan", nguon="E1L", dich="E1L", mode="N0",
                 model="naive", h=12, fold="o4032")
            i = chi_muc_du_doan(goc, ["tn_b"])
            for c in ("nguon", "dich", "mode", "h", "fold", "seed", "model",
                      "n_dong", "tep", "thu_muc", "thi_nghiem"):
                self.assertIn(c, i.columns)
            self.assertEqual(i["fold"].iloc[0], "o4032")


class TestNapHo(unittest.TestCase):
    def test_noi_dung_thu_tu_va_chi_cot_can(self):
        with tempfile.TemporaryDirectory() as t:
            goc = Path(t) / "tn_a" / "du_doan"
            p1 = _tep(goc, nguon="E1", dich="E1", mode="N0", model="ma6", h=1,
                      fold="test", n=3)
            p2 = _tep(goc, nguon="E1", dich="E1", mode="N0", model="naive", h=1,
                      fold="test", n=4)
            g = nap_ho([p1, p2])
            self.assertEqual(list(g.columns), list(COT_CHAM))
            self.assertEqual(len(g), 7)
            self.assertEqual(list(g["model"]), ["ma6"] * 3 + ["naive"] * 4)

    def test_cot_kiem_co_bucket_muc_tieu(self):
        with tempfile.TemporaryDirectory() as t:
            p = _tep(Path(t) / "tn_a" / "du_doan", nguon="E1", dich="E1",
                     mode="N0", model="naive", h=6, fold="test")
            g = nap_ho([p], cot=COT_KIEM)
            self.assertIn("bucket_muc_tieu", g.columns)
            self.assertTrue((g["bucket_muc_tieu"] - g["bucket_phat"] == 6).all())

    def test_mot_tep_khong_can_noi(self):
        with tempfile.TemporaryDirectory() as t:
            p = _tep(Path(t) / "tn_a" / "du_doan", nguon="E2", dich="E2",
                     mode="N1", model="svr", h=1, fold="test", n=2)
            self.assertEqual(len(nap_ho([p])), 2)


class TestThuTuDuyetHoKhongDoi(unittest.TestCase):
    """Chỉ mục phải cho cùng thứ tự họ như cách cũ: nạp cả kho rồi groupby(sort=False)."""

    KHOA_HO = ["thi_nghiem", "nguon", "dich", "mode", "h", "fold"]

    def _dung_kho(self, goc: Path):
        for env in ("E1", "E2"):
            for mode in ("N0", "N1"):
                for h in (1, 12):
                    for model in ("naive", "ma6", "xgb"):
                        _tep(goc / "tn_a" / "du_doan", nguon=env, dich=env,
                             mode=mode, model=model, h=h, fold="test")
        for env in ("E1", "E2"):
            _tep(goc / "dl" / "du_doan", nguon=env, dich=env, mode="N1",
                 model="dlinear", h=1, fold="test")

    def test_thu_tu_trung_ban_cu(self):
        with tempfile.TemporaryDirectory() as t:
            goc = Path(t)
            self._dung_kho(goc)
            tns = ["tn_a", "dl"]

            # cách cũ: nạp cả kho theo thứ tự thí nghiệm rồi groupby(sort=False)
            phan = []
            for tn in tns:
                x = pd.read_parquet(goc / tn / "du_doan")
                x["thi_nghiem"] = (
                    np.where(x["nguon"] == x["dich"], "tn_a", "tn_c")
                    if tn == "dl" else tn)
                phan.append(x)
            cu = [k for k, _ in pd.concat(phan, ignore_index=True)
                  .groupby(self.KHOA_HO, sort=False)]

            moi = [k for k, _ in chi_muc_du_doan(goc, tns)
                   .groupby(self.KHOA_HO, sort=False)]
            self.assertEqual(moi, cu)

    def test_so_dong_moi_ho_trung_ban_cu(self):
        with tempfile.TemporaryDirectory() as t:
            goc = Path(t)
            self._dung_kho(goc)
            i = chi_muc_du_doan(goc, ["tn_a", "dl"])
            for khoa, gi in i.groupby(self.KHOA_HO, sort=False):
                g = nap_ho(gi["tep"])
                self.assertEqual(len(g), int(gi["n_dong"].sum()))


if __name__ == "__main__":
    unittest.main()
