"""Điều phối các khối chạy: liệt kê, chạy từng khối, chạy tiếp sau gián đoạn.

    python scripts/dieu_phoi.py --liet-ke
    python scripts/dieu_phoi.py --trang-thai
    python scripts/dieu_phoi.py --chay K2
    python scripts/dieu_phoi.py --chay K4 --tiep      # bỏ qua tác vụ đã xong
    python scripts/dieu_phoi.py --doi-chieu           # chỉ đọc: khối K2, K3 đã xong thật chưa
    python scripts/dieu_phoi.py --doi-chieu --ap-dung # ghi kết quả đối chiếu, sao lưu trước

Mười khối, chạy tuần tự. Mỗi khối gồm vài tác vụ; mỗi tác vụ là một lệnh đầy đủ.

Trạng thái ghi ở `results/trang_thai.json`, một mục mỗi tác vụ:

    trang_thai   xong | hong | chua
    van_tay      vân tay của mã, cấu hình và dữ liệu lúc chạy xong
    dau_ra       danh sách tệp đầu ra và kích thước lúc kiểm

Điều kiện bỏ qua khi chạy tiếp: trạng thái `xong`, **và** vân tay còn khớp, **và**
mọi tệp đầu ra vẫn tồn tại với kích thước lớn hơn 0. Tệp tồn tại mà vân tay lệch thì
vẫn chạy lại, vì lệch vân tay nghĩa là mã, cấu hình hoặc dữ liệu đã đổi.

Vân tay theo tác vụ (v1.1). Bản ghi mới mang thêm `van_tay_tac_vu`: mã chỉ tính những tệp mà
tác vụ thực sự import (bao đóng import), nên sửa một script không liên quan không làm một tác vụ
dựng dữ liệu bị chạy lại. Bản ghi cũ chỉ có `van_tay` (băm toàn bộ mã) và thường lệch sau mọi
lần sửa mã; `--doi-chieu` là cách duy nhất để chuyển bản ghi cũ sang vân tay mới, và chỉ khi
đầu ra đã được xác minh. `--doi-chieu` KHÔNG chạy lại tác vụ nào.

Khối nào tốn nhiều giờ thì ghi rõ ở phần mô tả; script không tự chạy khối nào, luôn
phải gọi tên khối.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwp.quy_trinh import van_tay as vt  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PY = sys.executable

# DLinear cần PyTorch, thường cài ở một môi trường riêng để không kéo torch vào môi trường
# chính. Trỏ tới nó bằng biến môi trường `CWP_PYTHON_DL`; nếu không đặt thì dùng `.venv-dl`
# cạnh thư mục dự án, và nếu vẫn không có thì dùng chính trình thông dịch đang chạy.
def _python_dl() -> str:
    tu_bien = os.environ.get("CWP_PYTHON_DL")
    if tu_bien:
        return tu_bien
    for ung in (ROOT.parent / ".venv-dl" / "Scripts" / "python.exe",
                ROOT.parent / ".venv-dl" / "bin" / "python",
                ROOT / ".venv-dl" / "Scripts" / "python.exe",
                ROOT / ".venv-dl" / "bin" / "python"):
        if ung.exists():
            return str(ung)
    return sys.executable


PY_DL = _python_dl()
TRANG_THAI = ROOT / "results" / "trang_thai.json"


@dataclass
class TacVu:
    ma: str
    mo_ta: str
    lenh: list
    dau_ra: list = field(default_factory=list)
    du_lieu_vao: list = field(default_factory=list)
    uoc_luong: str = ""
    python: str = "chinh"      # chinh | dl


@dataclass
class Khoi:
    ma: str
    ten: str
    tac_vu: list
    ghi_chu: str = ""


def khoi_list() -> list:
    e = "E1,E2,E3"
    pr = [f"data/processed/{x}.parquet" for x in ("E1", "E2", "E3")]
    return [
        Khoi("K1", "Kiểm tra môi trường, repo, dữ liệu và cấu hình", [
            TacVu("K1.1", "kiểm tra đầu vào",
                  [PY, "scripts/kiem_tra_moi_truong.py"],
                  uoc_luong="dưới 1 phút"),
        ]),
        Khoi("K2", "Dựng dữ liệu và kiểm tra tính đúng", [
            TacVu("K2.1", "tiền xử lý ba môi trường",
                  [PY, "-m", "cwp.preprocess.build", "--env", e],
                  dau_ra=pr, uoc_luong="khoảng 5 phút, đã đo"),
            TacVu("K2.2", "ma trận đặc trưng N0",
                  [PY, "scripts/build_features.py", "--env", e],
                  dau_ra=[f"data/features/{x}_h{h}.parquet"
                          for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  du_lieu_vao=pr, uoc_luong="khoảng 1 phút, đã đo"),
            TacVu("K2.3", "ma trận đặc trưng N1 để đối chiếu phép affine",
                  [PY, "scripts/build_features.py", "--env", e, "--modes", "N1"],
                  dau_ra=[f"data/features/{x}_N1_h{h}.parquet"
                          for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  du_lieu_vao=pr, uoc_luong="khoảng 1 phút, đã đo"),
            TacVu("K2.4", "bảng CV có cột cv_train",
                  [PY, "scripts/fig_burstiness.py", "--env", "all"],
                  dau_ra=["results/tables/cv.csv"], du_lieu_vao=pr,
                  uoc_luong="khoảng 1 phút, đã đo"),
            TacVu("K2.5", "tập dòng đánh giá chung",
                  [PY, "scripts/tap_dong_chung.py", "--env", e,
                   "--horizons", "1,6,12"],
                  dau_ra=[f"data/eval_rows/{x}_h{h}.parquet"
                          for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  du_lieu_vao=pr, uoc_luong="dưới 1 phút, đã đo"),
            TacVu("K2.6", "kiểm tra tính đúng ba môi trường",
                  [PY, "scripts/check_dac_trung_va_baseline.py", "--env", "all"],
                  du_lieu_vao=pr + [f"data/features/{x}_{m}h{h}.parquet"
                                    for x in ("E1", "E2", "E3") for m in ("", "N1_")
                                    for h in (1, 6, 12)]
                  + [f"data/eval_rows/{x}_h{h}.parquet"
                     for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  uoc_luong="khoảng 5 phút, đã đo"),
        ]),
        Khoi("K3", "Đối chứng không nội suy K = 0", [
            TacVu("K3.1", "dựng dữ liệu K = 0",
                  [PY, "-m", "cwp.preprocess.build", "--env", e,
                   "--gap-max-len", "0",
                   "--processed", "data/k0/processed",
                   "--catalog", "data/k0/catalog.parquet"],
                  dau_ra=[f"data/k0/processed/{x}.parquet"
                          for x in ("E1", "E2", "E3")],
                  uoc_luong="khoảng 5 phút, đã đo"),
            TacVu("K3.2", "đặc trưng K = 0",
                  [PY, "scripts/build_features.py", "--env", e,
                   "--processed", "data/k0/processed",
                   "--out", "data/k0/features",
                   "--catalog", "data/k0/catalog.parquet"],
                  dau_ra=[f"data/k0/features/{x}_h{h}.parquet"
                          for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  du_lieu_vao=[f"data/k0/processed/{x}.parquet"
                               for x in ("E1", "E2", "E3")],
                  uoc_luong="khoảng 1 phút, đã đo"),
            TacVu("K3.3", "tập dòng đánh giá chung của K = 0",
                  [PY, "scripts/tap_dong_chung.py", "--env", e,
                   "--horizons", "1,6,12",
                   "--processed", "data/k0/processed",
                   "--features", "data/k0/features",
                   "--out", "data/k0/eval_rows"],
                  dau_ra=[f"data/k0/eval_rows/{x}_h{h}.parquet"
                          for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  du_lieu_vao=[f"data/k0/features/{x}_h{h}.parquet"
                               for x in ("E1", "E2", "E3") for h in (1, 6, 12)],
                  uoc_luong="dưới 1 phút, ước lượng"),
        ], ghi_chu="Phần huấn luyện của K = 0 nằm ở khối K7, sau khi có siêu tham số "
                   "của Thí nghiệm A."),
        Khoi("K4", "Thí nghiệm A: lưới chính", [
            TacVu("K4.1", "toàn bộ lưới đã chốt",
                  [PY, "scripts/tn_a.py", "--tat-ca", "--resume"],
                  dau_ra=["results/tn_a/tom_tat.csv",
                          "results/tn_a/chosen.csv"],
                  du_lieu_vao=pr,
                  uoc_luong="28,9 giờ, đã đo"),
            TacVu("K4.2", "DLinear ba hạt giống",
                  [PY_DL, "scripts/dl.py", "--env", e,
                   "--horizons", "1,6,12", "--models", "dlinear",
                   "--seeds", "42,43,44", "--out", "results/dl", "--resume"],
                  dau_ra=["results/dl/tom_tat.csv"], du_lieu_vao=pr,
                  uoc_luong="1,2 giờ, đã đo", python="dl"),
        ]),
        Khoi("K5", "Tổng hợp và kiểm tra đủ kết quả Thí nghiệm A", [
            TacVu("K5.1", "tổng hợp",
                  [PY, "scripts/tong_hop.py", "--thi-nghiem", "tn_a,dl"],
                  dau_ra=["results/tong_hop/chi_so_gop.csv",
                          "results/tong_hop/kiem_dinh.csv"],
                  uoc_luong="2,3 phút, đã đo"),
            TacVu("K5.2", "kiểm tra đủ tổ hợp",
                  [PY, "scripts/kiem_ket_qua.py", "--thi-nghiem", "tn_a"],
                  uoc_luong="khoảng 3 phút, đã đo"),
        ]),
        Khoi("K6", "Thí nghiệm B", [
            TacVu("K6.1", "dữ liệu dài ngày",
                  [PY, "scripts/du_lieu_tn_b.py"],
                  dau_ra=[f"data/processed/{x}.parquet"
                          for x in ("E1L", "E2m07", "E2m08", "E2m09")],
                  uoc_luong="8,1 phút, đã đo"),
            TacVu("K6.2", "lượt thử đo chi phí trên dữ liệu phát triển",
                  [PY, "scripts/tn_b.py", "--env", "E2m08",
                   "--gioi-han-fold", "1", "--out", "results/tn_b_thu"],
                  dau_ra=["results/tn_b_thu/tom_tat.csv"],
                  uoc_luong="18,4 phút, đã đo"),
            TacVu("K6.3", "lượt xác nhận",
                  [PY, "scripts/tn_b.py", "--env", "E1L,E2m07,E2m09",
                   "--resume"],
                  dau_ra=["results/tn_b/tom_tat.csv"],
                  uoc_luong="8,7 giờ, đã đo"),
        ], ghi_chu="Chỉ chạy K6.3 sau khi protocol đã khoá và K6.2 cho số đo."),
        Khoi("K7", "Thí nghiệm C, máy giả, K = 0 và độ nhạy", [
            TacVu("K7.1", "sáu chiều chuyển giao",
                  [PY, "scripts/tn_c.py", "--tat-ca", "--resume"],
                  dau_ra=["results/tn_c/tom_tat.csv"],
                  uoc_luong="3,3 giờ, đã đo"),
            TacVu("K7.2", "máy giả: dựng dữ liệu",
                  [PY, "scripts/build_may_gia.py",
                   "--gop", "config/may_gia_nhom_chot.csv"],
                  dau_ra=["data/processed/E1a.parquet",
                          "data/processed/E1g.parquet"],
                  uoc_luong="0,7 phút, đã đo"),
            TacVu("K7.2b", "máy giả: tập dòng đánh giá chung",
                  [PY, "scripts/tap_dong_chung.py", "--env", "E1a,E1g",
                   "--horizons", "1,6,12"],
                  dau_ra=[f"data/eval_rows/{x}_h{h}.parquet"
                          for x in ("E1a", "E1g") for h in (1, 6, 12)],
                  uoc_luong="0,2 phút, đã đo"),
            TacVu("K7.3", "máy giả: bốn chiều",
                  [PY, "scripts/tn_c.py", "--may-gia", "--tat-ca",
                   "--out", "results/may_gia", "--resume"],
                  dau_ra=["results/may_gia/tom_tat.csv"],
                  uoc_luong="46,5 phút, đã đo"),
            TacVu("K7.4", "K = 0: chạy mô hình",
                  [PY, "scripts/tn_a.py", "--env", e,
                   "--horizons", "1,6,12", "--modes", "N0,N1",
                   "--models", "naive,ma6,seasonal,xgb,xgb_mae",
                   "--features", "data/k0/features",
                   "--processed", "data/k0/processed",
                   "--chosen-co-dinh", "results/tn_a/chosen.csv",
                   "--out", "results/k0", "--resume"],
                  dau_ra=["results/k0/tom_tat.csv"],
                  uoc_luong="45,6 phút, đã đo (không dò lại siêu tham số)"),
            TacVu("K7.5", "độ nhạy ngưỡng: dựng dữ liệu",
                  [PY, "scripts/do_nhay.py", "--buoc", "du-lieu"],
                  uoc_luong="12,3 phút, đã đo"),
            TacVu("K7.6", "độ nhạy ngưỡng: chạy mô hình",
                  [PY, "scripts/do_nhay.py", "--buoc", "mo-hinh"],
                  dau_ra=["results/do_nhay/nguong_0p0/tom_tat.csv"],
                  uoc_luong="1,7 giờ, đã đo"),
            TacVu("K7.7", "Random Forest phạm vi hẹp",
                  [PY, "scripts/do_nhay.py", "--buoc", "rf"],
                  dau_ra=["results/rf/tom_tat.csv"],
                  uoc_luong="18,3 phút, đã đo"),
        ]),
        Khoi("K8", "Nhập kết quả GRU từ Colab", [
            TacVu("K8.1", "xuất gói dữ liệu cho Colab",
                  [PY, "scripts/xuat_goi_colab.py"],
                  dau_ra=["results/colab/goi_colab.zip"],
                  uoc_luong="dưới 1 phút"),
            TacVu("K8.2", "nhập kết quả tải về",
                  [PY, "scripts/nhap_ket_qua_colab.py",
                   "--zip", "results/colab/ket_qua_gru.zip"],
                  uoc_luong="dưới 1 phút"),
        ], ghi_chu="Giữa hai tác vụ là phần chạy trên Colab, xem "
                   "notebooks/colab_gru.ipynb."),
        Khoi("K9", "Tổng hợp toàn bộ", [
            TacVu("K9.1", "tổng hợp mọi thí nghiệm",
                  [PY, "scripts/tong_hop.py", "--thi-nghiem",
                   "tn_a,tn_b,tn_c,may_gia,dl"],
                  dau_ra=["results/tong_hop/chi_so_gop.csv"],
                  uoc_luong="13,7 phút, đã đo"),
            TacVu("K9.1b", "tổng hợp riêng Thí nghiệm A, không gộp mô hình chuỗi",
                  [PY, "scripts/tong_hop.py", "--thi-nghiem", "tn_a",
                   "--out", "results/tong_hop_khong_dl"],
                  dau_ra=["results/tong_hop_khong_dl/chi_so_gop.csv"],
                  uoc_luong="2,3 phút, đã đo"),
            TacVu("K9.2", "so bản có nội suy với K = 0 trên các thời điểm chung",
                  [PY, "scripts/so_k0.py"],
                  dau_ra=["results/so_k0/chi_so_chung.csv",
                          "results/so_k0/do_bao_phu.csv"],
                  uoc_luong="3 phút, đã đo"),
            TacVu("K9.3", "tổng hợp Random Forest phạm vi hẹp",
                  [PY, "scripts/tong_hop.py", "--thi-nghiem", "rf",
                   "--out", "results/tong_hop_rf"],
                  dau_ra=["results/tong_hop_rf/chi_so_gop.csv"],
                  uoc_luong="dưới 1 phút, ước lượng"),
            TacVu("K9.4", "tổng hợp độ nhạy ngưỡng 0%",
                  [PY, "scripts/tong_hop.py", "--goc", "results/do_nhay",
                   "--thi-nghiem", "nguong_0p0",
                   "--processed", "data/nguong_0p0/processed",
                   "--eval-rows", "data/nguong_0p0/eval_rows",
                   "--out", "results/tong_hop_nguong_0p0"],
                  dau_ra=["results/tong_hop_nguong_0p0/chi_so_gop.csv"],
                  uoc_luong="dưới 1 phút, ước lượng"),
            TacVu("K9.5", "tổng hợp độ nhạy ngưỡng 0,5%",
                  [PY, "scripts/tong_hop.py", "--goc", "results/do_nhay",
                   "--thi-nghiem", "nguong_0p5",
                   "--processed", "data/nguong_0p5/processed",
                   "--eval-rows", "data/nguong_0p5/eval_rows",
                   "--out", "results/tong_hop_nguong_0p5"],
                  dau_ra=["results/tong_hop_nguong_0p5/chi_so_gop.csv"],
                  uoc_luong="dưới 1 phút, ước lượng"),
            TacVu("K9.6", "tổng hợp độ nhạy ngưỡng 2%",
                  [PY, "scripts/tong_hop.py", "--goc", "results/do_nhay",
                   "--thi-nghiem", "nguong_2p0",
                   "--processed", "data/nguong_2p0/processed",
                   "--eval-rows", "data/nguong_2p0/eval_rows",
                   "--out", "results/tong_hop_nguong_2p0"],
                  dau_ra=["results/tong_hop_nguong_2p0/chi_so_gop.csv"],
                  uoc_luong="dưới 1 phút, ước lượng"),
        ], ghi_chu="K9.1b tồn tại vì mô hình chuỗi chỉ chạy ở chế độ N1 và kéo theo điều "
                   "kiện cửa sổ 288 điểm, nên ở E3 họ N1 có ít hơn họ N0 khoảng 0,54% số "
                   "dòng. Bảng của K9.1b bỏ mô hình chuỗi ra, nên N0 và N1 chấm trên đúng "
                   "cùng một tập dòng ở cả ba môi trường; đó là bảng dùng khi so hai chế độ "
                   "chuẩn hoá. Đã kiểm: khác biệt này không đổi kết luận nào."),
        Khoi("K10", "Kiểm tra toàn bộ kết quả trước khi viết bài", [
            TacVu("K10.1", "kiểm tra đủ và nhất quán",
                  [PY, "scripts/kiem_ket_qua.py", "--thi-nghiem", "tat-ca"],
                  uoc_luong="16,5 phút, đã đo"),
        ]),
    ]


def doc_trang_thai() -> dict:
    if TRANG_THAI.exists():
        return json.loads(TRANG_THAI.read_text(encoding="utf-8"))
    return {}


def ghi_trang_thai(tt: dict) -> None:
    TRANG_THAI.parent.mkdir(parents=True, exist_ok=True)
    TRANG_THAI.write_text(json.dumps(tt, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def dau_ra_con_nguyen(tv: TacVu) -> bool:
    for p in tv.dau_ra:
        q = ROOT / p
        if not q.exists() or q.stat().st_size == 0:
            return False
    return True


def bo_qua_duoc(tv: TacVu, tt: dict) -> bool:
    m = tt.get(tv.ma)
    if not m or m.get("trang_thai") != "xong":
        return False
    if not dau_ra_con_nguyen(tv):
        return False
    if "van_tay_tac_vu" in m:
        hien = vt.van_tay_tac_vu(ROOT, tv.lenh, [ROOT / x for x in tv.du_lieu_vao])
        return vt.khop_tac_vu(hien, m["van_tay_tac_vu"])
    hien = vt.van_tay(ROOT, [ROOT / x for x in tv.du_lieu_vao])
    return vt.khop(hien, m.get("van_tay", {}))


def chay_tac_vu(tv: TacVu, tt: dict, tiep: bool) -> bool:
    if tiep and bo_qua_duoc(tv, tt):
        print(f"  [bỏ qua] {tv.ma} {tv.mo_ta} — đã xong và còn khớp vân tay")
        return True
    print(f"\n  [chạy ] {tv.ma} {tv.mo_ta}")
    print(f"           $ {' '.join(str(x) for x in tv.lenh)}")
    t0 = time.time()
    r = subprocess.run(tv.lenh, cwd=ROOT)
    giay = round(time.time() - t0, 1)
    ok = r.returncode == 0 and dau_ra_con_nguyen(tv)
    tt[tv.ma] = {
        "trang_thai": "xong" if ok else "hong",
        "ma_thoat": r.returncode,
        "giay": giay,
        "luc": datetime.now().isoformat(timespec="seconds"),
        "lenh": [str(x) for x in tv.lenh],
        "van_tay": vt.van_tay(ROOT, [ROOT / x for x in tv.du_lieu_vao]),
        "van_tay_tac_vu": vt.van_tay_tac_vu(ROOT, tv.lenh,
                                            [ROOT / x for x in tv.du_lieu_vao]),
        "dau_ra": {p: (ROOT / p).stat().st_size
                   for p in tv.dau_ra if (ROOT / p).exists()},
    }
    ghi_trang_thai(tt)
    print(f"           {'xong' if ok else 'HỎNG'} sau {giay}s "
          f"(mã thoát {r.returncode})")
    return ok


# --- Đối chiếu trạng thái (chỉ đọc, không chạy tác vụ) ------------------------------

KHOI_DOI_CHIEU = ("K2", "K3")
MANIFEST_TRUOC = ROOT / "results" / "manifest_tham_chieu.json"
DUNG_SAI_GIAY = 2.0     # `luc` ghi tròn đến giây, tệp có phần lẻ giây


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _mtime_ns(p: Path) -> int:
    return p.stat().st_mtime_ns


def doi_chieu_tac_vu(tv: TacVu, tt: dict, manifest: dict, du_lieu_k41: str = "") -> dict:
    """Xác minh một tác vụ đã xong thật. Không chạy lệnh, không ghi tệp.

    Điều kiện (mọi điều kiện phải đạt):
      ban_ghi        bản ghi có mặt, mã thoát 0
      dau_ra         mọi đầu ra tồn tại, lớn hơn 0 byte, đúng kích thước bản ghi
      hoan_thanh     có bằng chứng hoàn thành sau khi đầu ra cuối cùng được ghi: log
                     `logs/<mã>.log` hoặc thời điểm `luc` của bản ghi
      ma_khong_doi   mọi tệp trong bao đóng import và mọi `config/*.yaml` cũ hơn đầu ra
                     đầu tiên (mã không đổi kể từ khi đầu ra được tạo)
      vao_khong_doi  mọi dữ liệu vào cũ hơn đầu ra đầu tiên
      sha_manifest   đầu ra có trong manifest trước khi sửa thì sha256 phải trùng
      kiem_tra_dat   (chỉ K2.6) log ghi `N/N phép kiểm đạt`, không có HỎNG
      du_lieu_k41    (chỉ K2.1, K2.2) vân tay dữ liệu K4.1 ghi lúc bắt đầu (processed và đặc
                     trưng N0: tên, kích thước, mtime) còn khớp dữ liệu hiện tại
    """
    import re
    kq = {"ma": tv.ma, "kiem": [], "dat": True}

    def them(ten, dat, ct=""):
        kq["kiem"].append((ten, bool(dat), ct))
        kq["dat"] &= bool(dat)

    m = tt.get(tv.ma)
    them("ban_ghi", bool(m) and m.get("ma_thoat") == 0,
         "" if m else "không có bản ghi")
    outs = [ROOT / p for p in tv.dau_ra]
    thieu = [p for p in tv.dau_ra if not (ROOT / p).exists() or (ROOT / p).stat().st_size == 0]
    # chỉ so kích thước với những tệp mà bản ghi có ghi kích thước
    khac_kt = [p for p in tv.dau_ra if p not in thieu and m and p in m.get("dau_ra", {})
               and m["dau_ra"][p] != (ROOT / p).stat().st_size]
    them("dau_ra", not thieu and not khac_kt,
         f"thiếu/rỗng {thieu[:2]}" if thieu else
         (f"{len(khac_kt)} tệp lệch kích thước so với bản ghi" if khac_kt else
          f"{len(outs)} tệp" if outs else "tác vụ không có tệp đầu ra"))

    log = ROOT / "logs" / f"{tv.ma}.log"
    moc_ra_dau = min((_mtime_ns(o) for o in outs if o.exists()), default=None)
    moc_ra_cuoi = max((_mtime_ns(o) for o in outs if o.exists()), default=None)
    bang_chung = None
    if log.exists() and (moc_ra_cuoi is None or _mtime_ns(log) >= moc_ra_cuoi):
        bang_chung = ("log", _mtime_ns(log))
    elif m and m.get("luc") and moc_ra_cuoi is not None:
        t = int(datetime.fromisoformat(m["luc"]).timestamp() * 1e9)
        if t + DUNG_SAI_GIAY * 1e9 >= moc_ra_cuoi:
            bang_chung = ("ban_ghi", t)
    them("hoan_thanh", bang_chung is not None,
         f"bằng chứng: {bang_chung[0]}" if bang_chung else
         "không có log hay bản ghi nào mới hơn đầu ra cuối cùng (bản ghi cũ hơn đầu ra)")

    moc = moc_ra_dau if moc_ra_dau is not None else (bang_chung[1] if bang_chung else None)
    code = vt.tep_vao_cua_lenh(ROOT, tv.lenh) + sorted((ROOT / "config").glob("*.yaml"))
    moi_nhat = max(code, key=_mtime_ns) if code else None
    them("ma_khong_doi", moc is not None and moi_nhat is not None
         and _mtime_ns(moi_nhat) <= moc,
         (f"{len(code)} tệp mã/cấu hình, mới nhất {moi_nhat.relative_to(ROOT).as_posix()} "
          f"{datetime.fromtimestamp(moi_nhat.stat().st_mtime):%d/%m %H:%M:%S}") if moi_nhat else "")
    vao = [ROOT / x for x in tv.du_lieu_vao]
    vao_thieu = [x for x in vao if not x.exists()]
    vao_moi = max((x for x in vao if x.exists()), key=_mtime_ns, default=None)
    them("vao_khong_doi", moc is not None and not vao_thieu
         and (vao_moi is None or _mtime_ns(vao_moi) <= moc),
         f"thiếu {vao_thieu[:2]}" if vao_thieu else
         (f"{len(vao)} tệp vào, mới nhất "
          f"{datetime.fromtimestamp(vao_moi.stat().st_mtime):%d/%m %H:%M:%S}"
          if vao_moi else "không có dữ liệu vào"))

    trong = [p for p in tv.dau_ra if p in manifest and (ROOT / p).exists()]
    lech = [p for p in trong if _sha256(ROOT / p) != manifest[p]]
    them("sha_manifest", not lech,
         f"{len(trong)}/{len(tv.dau_ra)} đầu ra có trong manifest, lệch {lech[:2]}" if lech else
         f"{len(trong)}/{len(tv.dau_ra)} đầu ra có trong manifest và trùng sha256")

    if tv.ma == "K2.6":
        txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        mm = re.findall(r"(\d+)/(\d+) phép kiểm đạt", txt)
        them("kiem_tra_dat", bool(mm) and mm[-1][0] == mm[-1][1] and "HỎNG" not in txt,
             f"{mm[-1][0]}/{mm[-1][1]} phép kiểm đạt" if mm else "log không có dòng tổng kết")
    if tv.ma in ("K2.1", "K2.2"):
        # đúng danh sách tệp mà tn_a.py băm khi chạy K4.1: processed + đặc trưng N0
        k41 = [ROOT / f"data/processed/{x}.parquet" for x in ("E1", "E2", "E3")]
        k41 += [ROOT / f"data/features/{x}_h{h}.parquet" for x in ("E1", "E2", "E3") for h in (1, 6, 12)]
        hien = vt.van_tay_du_lieu(k41)
        them("du_lieu_k41", bool(du_lieu_k41) and hien == du_lieu_k41,
             f"K4.1 ghi {du_lieu_k41 or 'không có'}, hiện tại {hien}")
    kq["van_tay_tac_vu"] = vt.van_tay_tac_vu(ROOT, tv.lenh, vao)
    return kq


def doi_chieu(khoi: list, tt: dict, ap_dung: bool) -> int:
    """Đối chiếu các khối `KHOI_DOI_CHIEU`. Với `ap_dung`, ghi kết quả vào `trang_thai.json`
    sau khi sao lưu. Bản ghi chưa xác minh được chuyển thành `chua_xac_minh`, không bao giờ
    thành `xong`. Không gọi `chay_tac_vu` và không chạy lệnh con nào."""
    manifest = json.loads(MANIFEST_TRUOC.read_text(encoding="utf-8")) if MANIFEST_TRUOC.exists() else {}
    du_k41 = ""
    tk = ROOT / "results" / "tn_a" / "tom_tat.csv"
    if tk.exists():
        import csv
        with open(tk, encoding="utf-8", newline="") as f:
            r = next(csv.DictReader(f), {})
        du_k41 = r.get("van_tay_du_lieu", "")
    print(f"manifest trước khi sửa: {len(manifest)} tệp"
          + ("" if manifest else "  (KHÔNG CÓ, điều kiện sha_manifest chỉ còn hình thức)"))
    print(f"vân tay dữ liệu K4.1 lúc bắt đầu: {du_k41 or 'không đọc được'}\n")
    moi = dict(tt)
    tat_ca = True
    for k in khoi:
        if k.ma not in KHOI_DOI_CHIEU:
            continue
        for tv in k.tac_vu:
            kq = doi_chieu_tac_vu(tv, tt, manifest, du_k41)
            tat_ca &= kq["dat"]
            print(f"  [{'XÁC MINH' if kq['dat'] else 'CHƯA    '}] {tv.ma:<5} {tv.mo_ta}")
            for ten, dat, ct in kq["kiem"]:
                print(f"      {'ok  ' if dat else 'LỖI'} {ten:<14} {ct}")
            b = dict(tt.get(tv.ma) or {})
            ghi = {"luc": datetime.now().isoformat(timespec="seconds"),
                   "kiem": [[t, d, c] for t, d, c in kq["kiem"]]}
            if kq["dat"]:
                b["trang_thai"] = "xong"
                b["van_tay_tac_vu"] = kq["van_tay_tac_vu"]
                b["dau_ra"] = {p: (ROOT / p).stat().st_size for p in tv.dau_ra}
            else:
                b["trang_thai"] = "chua_xac_minh"
            b["doi_chieu"] = ghi
            moi[tv.ma] = b
    print()
    if not ap_dung:
        print("Chỉ đọc, chưa ghi gì. Thêm --ap-dung để ghi trang_thai.json (có sao lưu).")
        return 0 if tat_ca else 1
    import shutil
    if TRANG_THAI.exists():
        bak = TRANG_THAI.with_name(f"trang_thai.json.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(TRANG_THAI, bak)
        print(f"Đã sao lưu {bak.relative_to(ROOT).as_posix()}")
    ghi_trang_thai(moi)
    print(f"Đã ghi {TRANG_THAI.relative_to(ROOT).as_posix()}")
    return 0 if tat_ca else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--liet-ke", action="store_true")
    ap.add_argument("--trang-thai", action="store_true")
    ap.add_argument("--chay", default=None, help="mã khối, ví dụ K4")
    ap.add_argument("--tac-vu", default=None, help="mã tác vụ, ví dụ K4.1")
    ap.add_argument("--doi-chieu", action="store_true",
                    help="chỉ đọc: xác minh K2, K3 đã xong thật (không chạy tác vụ nào)")
    ap.add_argument("--ap-dung", action="store_true",
                    help="với --doi-chieu: ghi kết quả vào trang_thai.json, có sao lưu")
    ap.add_argument("--tiep", action="store_true",
                    help="bỏ qua tác vụ đã xong và còn khớp vân tay")
    a = ap.parse_args()

    khoi = khoi_list()
    tt = doc_trang_thai()

    if a.doi_chieu:
        return doi_chieu(khoi, tt, a.ap_dung)

    if a.liet_ke or (not a.chay and not a.tac_vu and not a.trang_thai):
        print("\nCÁC KHỐI CHẠY\n")
        for k in khoi:
            print(f"{k.ma}. {k.ten}")
            for tv in k.tac_vu:
                m = tt.get(tv.ma, {})
                dau = {"xong": "xong", "hong": "HỎNG",
                       "chua_xac_minh": "?"}.get(
                    m.get("trang_thai"), "chưa")
                moi = " (môi trường torch)" if tv.python == "dl" else ""
                print(f"   [{dau:<4}] {tv.ma:<6} {tv.mo_ta}{moi} — {tv.uoc_luong}")
            if k.ghi_chu:
                print(f"   ghi chú: {k.ghi_chu}")
            print()
        return 0

    if a.trang_thai:
        print(json.dumps(tt, ensure_ascii=False, indent=2))
        return 0

    chon = []
    for k in khoi:
        for tv in k.tac_vu:
            if a.tac_vu and tv.ma == a.tac_vu:
                chon.append(tv)
            elif a.chay and k.ma == a.chay:
                chon.append(tv)
    if not chon:
        raise SystemExit(f"Không có tác vụ nào khớp {a.chay or a.tac_vu!r}.")

    print(f"Chạy {len(chon)} tác vụ. python: {PY}")
    for tv in chon:
        if not chay_tac_vu(tv, tt, a.tiep):
            print("\nDừng vì một tác vụ hỏng. Xem log phía trên, sửa rồi chạy lại "
                  "với --tiep để bỏ qua phần đã xong.")
            return 1
    print("\nXong toàn bộ tác vụ đã chọn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
