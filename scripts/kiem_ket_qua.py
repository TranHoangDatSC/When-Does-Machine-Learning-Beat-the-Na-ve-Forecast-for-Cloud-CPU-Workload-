"""Kiểm tra kết quả đã đủ và nhất quán, trước khi đọc số hay viết bài.

    python scripts/kiem_ket_qua.py --thi-nghiem tn_a
    python scripts/kiem_ket_qua.py --thi-nghiem tat-ca

Chín phép kiểm, mỗi phép trả lời một câu hỏi cụ thể. Không phép nào nói về việc mô
hình thắng hay thua: ở đây chỉ kiểm tính đúng và tính đủ.

Dự đoán được nạp **theo từng họ so sánh**, không nạp cả kho: Thí nghiệm B có khoảng 162
triệu dòng nên nạp hết sẽ hết bộ nhớ (protocol v1.2, Phụ lục B.6). R01 đến R03 chỉ cần
khoá và số dòng nên đọc từ metadata parquet, không đọc dữ liệu.

    R01  số tổ hợp có đủ so với phạm vi đã chốt (đếm theo THƯ MỤC dự đoán, vì phạm vi
         là phạm vi của một lần chạy script; DLinear ở `dl/` không thuộc phạm vi `tn_a`)
    R01b mỗi môi trường có đủ số fold (chỉ thí nghiệm nhiều fold, xem FOLD_MONG)
    R02  mỗi họ CHẤM TẠI CHỖ đều có mô hình naive (họ chuyển giao không cần, xem R02b)
    R02b mỗi họ chuyển giao có họ tại chỗ tương ứng kèm naive và kèm đủ mô hình của nó,
         tức hai tỉ số bắt buộc của protocol mục 9 tính được mà không ra NaN
    R03  mọi mô hình trong một họ chấm trên đúng cùng số dòng
    R04  dự đoán không có giá trị thiếu và không có giá trị vô hạn, tính TRONG tập dòng
         chung (protocol v1.1: seasonal có NaN ở dòng thiếu y(t+h-288), nằm ngoài tập chung)
    R05  dự đoán của mô hình cơ sở trùng nhau giữa hai chế độ chuẩn hoá
    R06  giá trị thật trùng với data/processed/ ở một mẫu ngẫu nhiên, np.allclose với
         rtol=1e-6, atol=1e-5 (mô hình chuỗi lưu y ở float32); mẫu lấy theo từng họ
    R07  mọi dòng dự đoán nằm trong tập dòng đánh giá chung
    R08  trong tập dòng chung, mọi mô hình của một họ có đúng cùng số dòng
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cwp.quy_trinh.kiem import (ho_thieu_naive,  # noqa: E402
                         kiem_huu_han_trong_tap_chung, y_khop)
from cwp.quy_trinh.nap import COT_KIEM, chi_muc_du_doan, nap_ho  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PHAM_VI = {
    # (môi trường, chế độ, tầm) × mô hình, đúng phạm vi đã chốt của Thí nghiệm A.
    "tn_a": {"env": 3, "mode": 2, "h": 3,
             "model": ["naive", "ma6", "seasonal", "ridge", "xgb", "xgb_mae",
                       "svr"]},
    # Thí nghiệm B: E1L, E2m07, E2m09 (E2m08 là dữ liệu phát triển, không vào bảng
    # xác nhận), tầm 1 và 12, thêm seasonal_tuan, bỏ svr. Mỗi môi trường 3 fold, xem
    # `FOLD_MONG`. Trước v1.2 `PHAM_VI` không có tn_b nên R01 lặng lẽ bỏ qua nó.
    "tn_b": {"env": 3, "mode": 2, "h": 2,
             "model": ["naive", "ma6", "seasonal", "seasonal_tuan", "ridge",
                       "xgb", "xgb_mae"]},
}
# Số fold mong đợi của từng thí nghiệm có nhiều fold. R01 đếm tổ hợp
# `(đích, chế độ, tầm, mô hình)` nên không thấy fold; thiếu hẳn một fold sẽ lọt lưới.
FOLD_MONG = {"tn_b": 3}
BASELINE = ("naive", "ma6")


class Bao:
    def __init__(self):
        self.dong = []

    def them(self, ma, dat, mo_ta, chi_tiet=""):
        self.dong.append((ma, bool(dat), mo_ta, chi_tiet))

    def in_ra(self) -> int:
        print()
        for ma, dat, mo_ta, ct in self.dong:
            print(f"  [{'đạt ' if dat else 'HỎNG'}] {ma}  {mo_ta}")
            if ct:
                print(f"            {ct}")
        hong = sum(1 for _, d, _, _ in self.dong if not d)
        print(f"\n{len(self.dong) - hong}/{len(self.dong)} phép kiểm đạt.")
        return 1 if hong else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thi-nghiem", default="tn_a")
    ap.add_argument("--goc", default="results")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--eval-rows", default="data/eval_rows")
    ap.add_argument("--n-mau", type=int, default=2000)
    a = ap.parse_args()

    tns = (["tn_a", "tn_b", "tn_c", "may_gia", "dl"]
           if a.thi_nghiem == "tat-ca"
           else [x.strip() for x in a.thi_nghiem.split(",")])
    goc = ROOT / a.goc
    b = Bao()
    rng = np.random.default_rng(42)

    print("KIỂM TRA KẾT QUẢ")
    print(f"python: {sys.executable}")

    # Chỉ mục tệp thay cho việc nạp cả kho dự đoán. Thí nghiệm B có ~162 triệu dòng;
    # `pd.read_parquet(<thư mục>)` trên đó cần trên 10 GB và làm script chết vì hết bộ
    # nhớ. R01, R02, R03 chỉ cần khoá và số dòng, đọc được từ metadata parquet. R04 đến
    # R08 nạp từng họ so sánh rồi giải phóng.
    chi_muc = chi_muc_du_doan(goc, tns)
    for tn in tns:
        q = goc / tn / "du_doan"
        if not q.exists():
            b.them(f"R00.{tn}", False, f"có thư mục dự đoán {q.relative_to(ROOT)}")
    if chi_muc.empty:
        return b.in_ra()
    for tn in tns:
        x = chi_muc[chi_muc["thu_muc"] == tn]
        if x.empty:
            continue
        print(f"  {tn}: {int(x['n_dong'].sum()):,} dòng, {x['model'].nunique()} mô hình, "
              f"{x['dich'].nunique()} môi trường đích")

    # R01 ---------------------------------------------------------------
    for tn, pv in PHAM_VI.items():
        if tn not in tns:
            continue
        # `thu_muc`, không phải `thi_nghiem`: PHAM_VI là phạm vi của một lần chạy script.
        # Dự đoán DLinear ở thư mục `dl/` thuộc HỌ `tn_a` nhưng không thuộc PHẠM VI của
        # `tn_a.py`, nên đếm chúng vào tn_a sẽ ra 135 thay vì 126.
        x = chi_muc[chi_muc["thu_muc"] == tn]
        mong = pv["env"] * pv["mode"] * pv["h"] * len(pv["model"])
        co = x.groupby(["dich", "mode", "h", "model"]).ngroups
        b.them(f"R01.{tn}", co == mong,
               f"{tn}: đủ {mong} tổ hợp (môi trường × chế độ × tầm × mô hình)",
               f"đang có {co}")
        if tn in FOLD_MONG:
            nf = x.groupby("dich")["fold"].nunique()
            du = bool(len(nf)) and bool((nf == FOLD_MONG[tn]).all())
            b.them(f"R01b.{tn}", du,
                   f"{tn}: mỗi môi trường có đủ {FOLD_MONG[tn]} fold",
                   "" if du else f"số fold theo môi trường: {nf.to_dict()}")

    # R02, R03 -----------------------------------------------------------
    # R02 nhìn họ so sánh như tong_hop.py: `dl` đã được gộp vào `tn_a` hoặc `tn_c`
    # trong cột `thi_nghiem` của chỉ mục. R03 so số dòng THÔ theo từng thư mục, vì mô
    # hình chuỗi bị cắt thêm điều kiện `chuoi` nên có ít dòng hơn naive; việc so trong
    # tập chung là của R08.
    # R02 chỉ đòi naive ở họ **chấm tại chỗ** (`nguồn == đích`). Họ chuyển giao không có
    # naive là đúng thiết kế: dự đoán naive không phụ thuộc mô hình nguồn, nên `chuyen_giao.csv`
    # lấy "naive tại đích" từ chiều tại chỗ của cùng đích. Việc đó do R02b kiểm.
    KHOA_HO_R02 = ["thi_nghiem", "nguon", "dich", "mode", "h", "fold"]
    r02 = ho_thieu_naive(chi_muc, KHOA_HO_R02)
    thieu_naive = r02["tai_cho_thieu_naive"]
    lech_dong = []
    for khoa, gi in chi_muc.groupby(["thu_muc", "nguon", "dich", "mode", "h", "fold"]):
        n = gi.groupby(["model", "seed"])["n_dong"].sum()
        if n.nunique() > 1:
            lech_dong.append((khoa, int(n.min()), int(n.max())))
    b.them("R02", not thieu_naive,
           "mọi họ chấm tại chỗ đều có naive",
           f"{len(thieu_naive)} họ thiếu: {thieu_naive[:3]}" if thieu_naive else "")
    b.them("R03", not lech_dong,
           "mọi mô hình trong một họ chấm trên cùng số dòng",
           f"{len(lech_dong)} họ lệch: {lech_dong[:3]}" if lech_dong else "")

    # R02b: `chuyen_giao.csv` ghép mỗi họ chuyển giao với họ tại chỗ của cùng đích để ra
    # hai tỉ số bắt buộc của protocol mục 9. Thiếu naive tại đích, hoặc thiếu chính mô hình
    # đó ở chiều tại chỗ, là cột tỉ số sẽ ra NaN. Nếu chiều tại chỗ không nằm trong phạm vi
    # đang kiểm (ví dụ chỉ kiểm tn_c mà không kiểm tn_a) thì không kết luận được, và nói rõ.
    thieu_tham_chieu = r02["thieu_tham_chieu"]
    ngoai_pham_vi = r02["ngoai_pham_vi"]
    # Họ chuyển giao mà chiều tại chỗ không nằm trong phạm vi lần kiểm này thì KHÔNG kết
    # luận được. Báo "chưa kiểm được" chứ không báo đạt, để lần chạy hẹp không nguỵ trang
    # thành một lần kiểm đầy đủ.
    b.them("R02b", not thieu_tham_chieu and not ngoai_pham_vi,
           "mỗi họ chuyển giao có họ tại chỗ tương ứng, kèm naive và kèm đủ mô hình của nó",
           (f"{len(thieu_tham_chieu)} họ thiếu tham chiếu: {thieu_tham_chieu[:3]}"
            if thieu_tham_chieu else "")
           + ("; " if thieu_tham_chieu and ngoai_pham_vi else "")
           + (f"CHƯA KIỂM ĐƯỢC {len(ngoai_pham_vi)} họ chuyển giao: chiều tại chỗ của đích "
              f"không nằm trong --thi-nghiem lần này, hãy kiểm kèm thí nghiệm chấm tại chỗ "
              f"(ví dụ --thi-nghiem tat-ca)" if ngoai_pham_vi else ""))

    # R04, R06, R07, R08 -------------------------------------------------
    # Một lượt duyệt, mỗi họ nạp một lần. Họ ở đây gồm cả `fold`, đúng định nghĩa họ so
    # sánh của protocol mục 8; với Thí nghiệm A `fold` luôn là `test` nên không đổi gì.
    import tong_hop as th  # noqa: E402

    hong_trong, hong_ngoai, lech_tap, loi_tap = 0, 0, [], []
    lech_y, y_dat, n_thieu_y, n_mau = 0.0, True, 0, 0
    ngoai = 0
    KHOA_HO = ["thi_nghiem", "nguon", "dich", "mode", "h", "fold"]
    n_ho = chi_muc.groupby(KHOA_HO).ngroups
    mau_moi_ho = max(1, a.n_mau // max(1, n_ho))
    bo_nho = {}          # bộ đệm một môi trường: eval_rows và data/processed

    def _hop(dich, h):
        """Tập khoá `(series_id, bucket)` hợp lệ của ma trận đặc trưng (cột `bang`).

        Dựng một tập khoá của E1L mất hàng chục giây và mỗi (đích, tầm) có 6 họ (2 chế độ
        × 3 fold), nên phải đệm lại; chỉ giữ một (đích, tầm) để không phình bộ nhớ.
        """
        k = ("hop", dich, int(h))
        if k not in bo_nho:
            for kk in [x for x in bo_nho if x[0] == "hop"]:
                del bo_nho[kk]
            er = pd.read_parquet(ROOT / a.eval_rows / f"{dich}_h{h}.parquet")
            m = er["bang"].to_numpy(dtype=bool)
            bo_nho[k] = set(zip(er.loc[m, "series_id"], er.loc[m, "bucket"]))
        return bo_nho[k]

    def _y_that(dich):
        k = ("y", dich)
        if k not in bo_nho:
            for kk in [x for x in bo_nho if x[0] == "y"]:
                del bo_nho[kk]
            q = ROOT / a.processed / f"{dich}.parquet"
            bo_nho[k] = (None if not q.exists() else
                         pd.read_parquet(q, columns=["series_id", "bucket", "y"])
                         .set_index(["series_id", "bucket"])["y"])
        return bo_nho[k]

    for khoa, gi in chi_muc.groupby(KHOA_HO, sort=True):
        tn, nguon, dich, mode, h, fold = khoa
        try:
            khoa_chung, _ = th.dong_chung(dich, int(h), set(gi["model"]),
                                          ROOT / a.eval_rows)
        except SystemExit as e:
            loi_tap.append((khoa, str(e)))
            continue
        g = nap_ho(gi["tep"], cot=COT_KIEM)

        r = kiem_huu_han_trong_tap_chung(g, khoa_chung)
        hong_trong += r["n_hong_trong"]
        hong_ngoai += r["n_hong_ngoai"]
        if r["n_trong_moi_nhom"][0] != r["n_trong_moi_nhom"][1]:
            lech_tap.append((khoa, r["n_trong_moi_nhom"]))

        # R06: mẫu ngẫu nhiên của chính họ này, đối chiếu với data/processed
        y = _y_that(dich)
        if y is not None:
            mau_g = g.sample(min(mau_moi_ho, len(g)), random_state=42)
            k = pd.MultiIndex.from_arrays([mau_g["series_id"], mau_g["bucket_muc_tieu"]])
            that = y.reindex(k).to_numpy("float64")
            ok, lech, thieu = y_khop(that, mau_g["y"].to_numpy())
            lech_y = max(lech_y, lech)
            y_dat &= ok
            n_thieu_y += thieu
            n_mau += len(mau_g)

        # R07: mọi dòng dự đoán phải nằm trong tập dòng hợp lệ của ma trận (`bang`)
        hop = _hop(dich, h)
        lay = g.sample(min(5000, len(g)), random_state=1)
        ngoai += sum(1 for s, t in zip(lay["series_id"], lay["bucket_phat"])
                     if (s, t) not in hop)
        del g

    b.them("R04", hong_trong == 0 and not loi_tap,
           "trong tập dòng chung, dự đoán và giá trị thật đều hữu hạn",
           (f"{hong_trong:,} dòng hỏng trong tập chung" if hong_trong else "")
           + (f"; không dựng được tập chung: {loi_tap[:2]}" if loi_tap else "")
           + (f"{hong_ngoai:,} dòng không hữu hạn nằm NGOÀI tập chung (không mô hình nào "
              f"được chấm trên chúng), bình thường với seasonal"
              if not hong_trong and hong_ngoai else ""))
    b.them("R08", not lech_tap,
           "trong tập chung, mọi mô hình của một họ có cùng số dòng",
           f"{len(lech_tap)} họ lệch: {lech_tap[:3]}" if lech_tap else "")

    # R05 ----------------------------------------------------------------
    # Mô hình cơ sở không dùng đặc trưng đã chuẩn hoá, nên dự đoán ở N0 và N1 phải trùng.
    lech_mode = []
    cs = chi_muc[chi_muc["model"].isin(BASELINE)]
    for khoa, gi in cs.groupby(["thu_muc", "nguon", "dich", "h", "fold", "model"]):
        if gi["mode"].nunique() < 2:
            continue
        hai = [gi[gi["mode"] == m] for m in sorted(set(gi["mode"]))[:2]]
        w = []
        for x in hai:
            t = nap_ho(x["tep"], cot=["series_id", "bucket_phat", "yhat"])
            w.append(t.set_index(["series_id", "bucket_phat"])["yhat"])
        a0, a1 = w[0].align(w[1], join="outer")
        v = float(np.nanmax(np.abs(a0.to_numpy("float64") - a1.to_numpy("float64"))))
        if v > 1e-6:
            lech_mode.append((khoa, v))
    b.them("R05", not lech_mode,
           "mô hình cơ sở cho cùng dự đoán ở N0 và N1",
           f"{len(lech_mode)} chỗ lệch: {lech_mode[:2]}" if lech_mode else "")

    b.them("R06", y_dat,
           "giá trị thật trùng data/processed/ tại thời điểm mục tiêu "
           "(np.allclose, rtol=1e-6, atol=1e-5)",
           f"lệch tối đa {lech_y:.3e} trên {n_mau:,} dòng mẫu ({mau_moi_ho}/họ)"
           + (f"; {n_thieu_y} dòng trỏ tới điểm không có quan sát" if n_thieu_y else ""))
    b.them("R07", ngoai == 0,
           "mọi dòng dự đoán nằm trong tập dòng hợp lệ",
           f"{ngoai} dòng ngoài tập" if ngoai else "")

    return b.in_ra()


if __name__ == "__main__":
    sys.exit(main())
