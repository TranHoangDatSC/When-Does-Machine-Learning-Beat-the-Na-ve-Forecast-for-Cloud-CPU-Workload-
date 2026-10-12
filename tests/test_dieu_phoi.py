"""Kiểm thử vân tay theo tác vụ và bước đối chiếu của dieu_phoi.py.

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Các phép thử ở đây không chạy tác vụ nào. Phép thử cuối vá `subprocess.run` để nổ nếu
`--doi-chieu` gọi bất kỳ lệnh con nào (K2, K3 hay tác vụ khác).
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.quy_trinh import van_tay as vt  # noqa: E402


def _du_an(d: Path):
    (d / "src" / "pkg").mkdir(parents=True)
    (d / "scripts").mkdir()
    (d / "config").mkdir()
    (d / "config" / "a.yaml").write_text("x: 1\n", encoding="utf-8")
    (d / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (d / "src" / "pkg" / "dung.py").write_text("from pkg import huu_ich\n", encoding="utf-8")
    (d / "src" / "pkg" / "huu_ich.py").write_text("V = 1\n", encoding="utf-8")
    (d / "src" / "pkg" / "khong_dung.py").write_text("V = 2\n", encoding="utf-8")
    (d / "scripts" / "dung_du_lieu.py").write_text(
        "from pkg.dung import *\nimport numpy\n", encoding="utf-8")
    (d / "scripts" / "tac_vu_khac.py").write_text("import pkg.khong_dung\n", encoding="utf-8")


class TestBaoDongImport(unittest.TestCase):
    def test_bao_dong_dung(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _du_an(d)
            tep = {p.relative_to(d).as_posix() for p in vt.bao_dong_import(d, "scripts/dung_du_lieu.py")}
            self.assertEqual(tep, {"scripts/dung_du_lieu.py", "src/pkg/__init__.py",
                                   "src/pkg/dung.py", "src/pkg/huu_ich.py"})

    def test_lenh_dash_m(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _du_an(d)
            tep = {p.relative_to(d).as_posix() for p in vt.tep_vao_cua_lenh(d, ["py", "-m", "pkg.dung", "--x"])}
            self.assertIn("src/pkg/huu_ich.py", tep)
            self.assertNotIn("src/pkg/khong_dung.py", tep)

    def test_sua_tep_khong_lien_quan_khong_doi_van_tay(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _du_an(d)
            lenh = ["py", "scripts/dung_du_lieu.py"]
            truoc = vt.van_tay_tac_vu(d, lenh)
            (d / "src" / "pkg" / "khong_dung.py").write_text("V = 3\n", encoding="utf-8")
            (d / "scripts" / "tac_vu_khac.py").write_text("print('x')\n", encoding="utf-8")
            self.assertTrue(vt.khop_tac_vu(truoc, vt.van_tay_tac_vu(d, lenh)))

    def test_sua_tep_trong_bao_dong_doi_van_tay(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _du_an(d)
            lenh = ["py", "scripts/dung_du_lieu.py"]
            truoc = vt.van_tay_tac_vu(d, lenh)
            (d / "src" / "pkg" / "huu_ich.py").write_text("V = 99\n", encoding="utf-8")
            self.assertFalse(vt.khop_tac_vu(truoc, vt.van_tay_tac_vu(d, lenh)))

    def test_doi_cau_hinh_doi_van_tay(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _du_an(d)
            lenh = ["py", "scripts/dung_du_lieu.py"]
            truoc = vt.van_tay_tac_vu(d, lenh)
            (d / "config" / "a.yaml").write_text("x: 2\n", encoding="utf-8")
            self.assertFalse(vt.khop_tac_vu(truoc, vt.van_tay_tac_vu(d, lenh)))

    def test_khop_tac_vu_tu_choi_van_tay_rong(self):
        self.assertFalse(vt.khop_tac_vu({}, {}))


class TestDoiChieuKhongChayTacVu(unittest.TestCase):
    def test_doi_chieu_chi_doc_khong_goi_subprocess(self):
        import dieu_phoi as dp

        def no(*a, **k):
            raise AssertionError("--doi-chieu không được chạy lệnh con")

        tt = dp.doc_trang_thai()
        truoc = dp.TRANG_THAI.read_bytes() if dp.TRANG_THAI.exists() else b""
        with mock.patch.object(dp.subprocess, "run", no), \
                mock.patch.object(dp.subprocess, "Popen", no), \
                mock.patch.object(dp, "chay_tac_vu", no):
            dp.doi_chieu(dp.khoi_list(), tt, ap_dung=False)
        sau = dp.TRANG_THAI.read_bytes() if dp.TRANG_THAI.exists() else b""
        self.assertEqual(truoc, sau, "chế độ chỉ đọc đã ghi vào trang_thai.json")

    def test_ban_ghi_cu_hon_dau_ra_khong_duoc_xac_minh(self):
        # Bản ghi có `luc` sớm hơn đầu ra và không có log => không có bằng chứng hoàn thành.
        import dieu_phoi as dp
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "ra.txt"
            out.write_text("x", encoding="utf-8")
            tv = dp.TacVu("KX.1", "giả", ["py", "scripts/kiem_tra_moi_truong.py"])
            tv.dau_ra = []
            tt = {"KX.1": {"trang_thai": "xong", "ma_thoat": 0, "luc": "2000-01-01T00:00:00",
                           "dau_ra": {}}}
            with mock.patch.object(dp, "ROOT", Path(t)):
                (Path(t) / "logs").mkdir()
                (Path(t) / "config").mkdir()
                kq = dp.doi_chieu_tac_vu(tv, tt, {}, "")
            self.assertFalse(kq["dat"])

    def test_bo_qua_duoc_dung_van_tay_tac_vu_khi_co(self):
        import dieu_phoi as dp
        tv = next(t for k in dp.khoi_list() for t in k.tac_vu if t.ma == "K2.4")
        ban = {"trang_thai": "xong",
               "van_tay_tac_vu": vt.van_tay_tac_vu(ROOT, tv.lenh, [ROOT / x for x in tv.du_lieu_vao])}
        with mock.patch.object(dp, "dau_ra_con_nguyen", lambda _tv: True):
            self.assertTrue(dp.bo_qua_duoc(tv, {"K2.4": ban}))
            ban["van_tay_tac_vu"] = dict(ban["van_tay_tac_vu"], ma="0" * 16)
            self.assertFalse(dp.bo_qua_duoc(tv, {"K2.4": ban}))


if __name__ == "__main__":
    unittest.main()
