"""Các phép kiểm dùng chung của bộ kiểm kết quả và bộ nhập kết quả Colab (protocol v1.1).

Mỗi hàm ở đây thuần tuý: nhận mảng hoặc bảng, không đọc tệp theo đường dẫn cố định, để
kiểm thử được mà không cần dữ liệu thật. Ngoại lệ là `doc_cv_search`, đọc một tệp CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Sai lệch cho phép khi so `y` trong dự đoán với `data/processed`. Mô hình chuỗi lưu `y`
# ở float32 nên lệch tối đa khoảng 3,7e-6 trên thang phần trăm CPU; ngưỡng cũ 1e-9 là ngưỡng
# của float64 và bắt nhầm. 1e-3 (một phần nghìn điểm phần trăm) vẫn phải bị bắt.
RTOL_Y = 1e-6
ATOL_Y = 1e-5

KHOA_CV = ["env", "mode", "h", "model", "ung_vien", "fold"]
# Cột phải giống nhau (chính xác) giữa các bản trùng của cùng một khoá.
COT_CV_NGUYEN = ["n_train", "n_val"]
# Cột số thực phải giống nhau trong dung sai. `giay` là thời gian chạy nên không so.
COT_CV_THUC = ["mae_p50_val"]
DUNG_SAI_CV = 1e-9


def y_khop(that, y, rtol: float = RTOL_Y, atol: float = ATOL_Y) -> tuple:
    """So `y` ghi trong dự đoán với giá trị thật đọc từ `data/processed`.

    Trả về `(dat, lech_toi_da, n_thieu)`:
      * `n_thieu`: số dòng mà giá trị thật đọc lại là NaN trong khi `y` hữu hạn, tức dòng dự
        đoán trỏ tới một điểm không có quan sát. Đây luôn là lỗi.
      * `lech_toi_da`: max |that - y| trên các dòng cả hai hữu hạn, chỉ để báo cáo.
      * `dat`: mọi dòng cả hai hữu hạn thoả `np.allclose(that, y, rtol, atol)` và `n_thieu == 0`.
    """
    that = np.asarray(that, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if that.shape != y.shape:
        raise ValueError(f"độ dài lệch: {that.shape} và {y.shape}")
    huu = np.isfinite(that) & np.isfinite(y)
    n_thieu = int((~np.isfinite(that) & np.isfinite(y)).sum())
    if not huu.any():
        return n_thieu == 0, 0.0, n_thieu
    lech = float(np.max(np.abs(that[huu] - y[huu])))
    dat = bool(np.allclose(that[huu], y[huu], rtol=rtol, atol=atol)) and n_thieu == 0
    return dat, lech, n_thieu


def kiem_huu_han_trong_tap_chung(g: pd.DataFrame, khoa_chung) -> dict:
    """R04 cho một họ so sánh: dự đoán phải hữu hạn **bên trong** tập dòng chung.

    `seasonal` không có dự đoán ở dòng thiếu `y(t+h-288)`; các dòng đó là NaN nhưng nằm ngoài
    tập chung nên không mô hình nào được chấm trên chúng. Chỉ NaN bên trong tập chung mới là lỗi.

    Trả về:
      n_trong_moi_nhom     (min, max) số dòng trong tập chung của mỗi (mô hình, seed);
                           hai giá trị bằng nhau nghĩa là mọi mô hình cùng một tập dòng
      n_hong_trong         số dòng trong tập chung có `yhat` hoặc `y` không hữu hạn
      n_hong_ngoai         số dòng ngoài tập chung không hữu hạn (chỉ báo cáo)
    """
    ky = pd.MultiIndex.from_arrays([g["series_id"], g["bucket_phat"]])
    trong = np.asarray(ky.isin(list(khoa_chung)))
    hong = ~(np.isfinite(g["yhat"].to_numpy("float64")) & np.isfinite(g["y"].to_numpy("float64")))
    dem = (g.assign(_t=trong).groupby(["model", "seed"])["_t"].sum()
           if len(g) else pd.Series(dtype="int64"))
    return {
        "n_trong_moi_nhom": (int(dem.min()), int(dem.max())) if len(dem) else (0, 0),
        "n_hong_trong": int((hong & trong).sum()),
        "n_hong_ngoai": int((hong & ~trong).sum()),
    }


KHOA_HO_CHI_MUC = ["thi_nghiem", "nguon", "dich", "mode", "h", "fold"]


def ho_thieu_naive(chi_muc: pd.DataFrame, khoa=None) -> dict:
    """R02 và R02b trên chỉ mục tệp dự đoán (protocol v1.2).

    Chỉ **họ chấm tại chỗ** (`nguon == dich`) mới buộc có naive. Họ chuyển giao
    (`nguon != dich`) không có naive là đúng thiết kế: dự đoán naive không phụ thuộc mô
    hình nguồn, nên `chuyen_giao.csv` lấy "naive tại đích" từ chiều tại chỗ của cùng đích.

    Protocol mục 9 đòi Thí nghiệm C báo **hai tỉ số cùng lúc**: so với mô hình học tại
    đích, và so với naive tại đích. Cả hai đều lấy từ họ tại chỗ của đích, nên họ tại chỗ
    đó phải có naive **và** có đủ những mô hình mà họ chuyển giao đã chạy; thiếu là cột tỉ
    số ra NaN.

    Trả về dict:
      `tai_cho_thieu_naive`  họ tại chỗ không có naive — luôn là lỗi
      `thieu_tham_chieu`     (họ chuyển giao, mô hình còn thiếu ở họ tại chỗ) — lỗi
      `ngoai_pham_vi`        họ chuyển giao mà họ tại chỗ của đích **không có trong chỉ
                             mục**, ví dụ khi chỉ kiểm tn_c mà không kiểm tn_a. Không kết
                             luận được, và **không được coi là đạt một cách âm thầm**.
    """
    khoa = list(KHOA_HO_CHI_MUC if khoa is None else khoa)
    thieu, tham_chieu, ngoai = [], [], []
    mh_tai_cho = {}
    if chi_muc is None or len(chi_muc) == 0:
        return {"tai_cho_thieu_naive": [], "thieu_tham_chieu": [], "ngoai_pham_vi": []}
    nhom = {k: set(g["model"]) for k, g in chi_muc.groupby(khoa)}
    for k, mh in nhom.items():
        _, nguon, dich, mode, h, fold = k
        if nguon != dich:
            continue
        mh_tai_cho[(dich, mode, h, fold)] = mh
        if "naive" not in mh:
            thieu.append(k)
    for k, mh in nhom.items():
        _, nguon, dich, mode, h, fold = k
        if nguon == dich:
            continue
        goc = (dich, mode, h, fold)
        if goc not in mh_tai_cho:
            ngoai.append(k)
            continue
        con_thieu = ({"naive"} | mh) - mh_tai_cho[goc]
        if con_thieu:
            tham_chieu.append((k, sorted(con_thieu)))
    return {"tai_cho_thieu_naive": sorted(thieu),
            "thieu_tham_chieu": sorted(tham_chieu),
            "ngoai_pham_vi": sorted(ngoai)}


def doc_cv_search(p: Path, dung_sai: float = DUNG_SAI_CV) -> pd.DataFrame:
    """Đọc `cv_search.csv`, khử trùng theo khoá `KHOA_CV`.

    Các bản trùng của cùng một khoá phải giống nhau: `n_train`, `n_val` đúng bằng nhau và
    `mae_p50_val` lệch không quá `dung_sai`. Nếu không, ném `ValueError` (không chọn một bản
    một cách âm thầm). Cột `giay` không được so vì thời gian chạy khác nhau giữa các lần.
    Giữ bản đầu tiên của mỗi khoá.
    """
    cv = pd.read_csv(p)
    thieu = [c for c in KHOA_CV + COT_CV_NGUYEN + COT_CV_THUC if c not in cv.columns]
    if thieu:
        raise ValueError(f"{p}: thiếu cột {thieu}")
    return khu_trung_cv(cv, dung_sai)


def khu_trung_cv(cv: pd.DataFrame, dung_sai: float = DUNG_SAI_CV) -> pd.DataFrame:
    """Phần lõi của `doc_cv_search`, tách ra để kiểm thử trên bảng trong bộ nhớ."""
    if not cv.duplicated(KHOA_CV).any():
        return cv.reset_index(drop=True)
    gr = cv.groupby(KHOA_CV, sort=False, dropna=False)
    xau = []
    for c in COT_CV_NGUYEN:
        n = gr[c].nunique(dropna=False)
        xau += [(k, c, "khác nhau") for k in n[n > 1].index]
    for c in COT_CV_THUC:
        d = (gr[c].max() - gr[c].min())
        xau += [(k, c, f"lệch {v:.3e}") for k, v in d[d > dung_sai].items()]
    if xau:
        raise ValueError(
            f"cv_search.csv có {len(xau)} khoá trùng mà giá trị khác nhau quá dung sai "
            f"{dung_sai:g}; ví dụ {xau[:3]}")
    return cv.drop_duplicates(KHOA_CV, keep="first").reset_index(drop=True)
