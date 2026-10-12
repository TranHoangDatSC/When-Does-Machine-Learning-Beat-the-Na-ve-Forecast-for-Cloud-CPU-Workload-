"""Chia train/validation/test theo bucket, và rolling-origin nhiều fold.

Đầu vào là ma trận đặc trưng `data/features/...parquet` cùng horizon `h`. Đầu ra là
**mặt nạ bool** hoặc **chỉ số**, không sao chép dữ liệu, vì ma trận E1 h=1 đã có hơn
1,6 triệu dòng.

Bốn quy ước
-----------

**1. Ranh giới tính theo bucket, không theo số dòng.** Cửa sổ là chung cho cả môi
trường nên ranh giới bucket là một lát cắt thời gian *giống hệt nhau ở mọi chuỗi*.
Nếu chia theo số dòng hợp lệ, chuỗi nhiều NaN sẽ có tập test bắt đầu muộn hơn, và
"chia theo thời gian" không còn đúng nghĩa.

**2. Một dòng thuộc tập `S` khi cả `t` VÀ `t+h` nằm trong `S`.** Dòng vắt qua ranh
giới bị **loại** (purge). Nếu chỉ gán theo `t`, những dòng cuối tập train có target
`y_{t+h}` rơi vào validation hoặc test, và huấn luyện trên chúng là cho model thấy
nhãn của tập đánh giá.

**3. Không xáo trộn.** Mọi hàm ở đây làm việc trên trục bucket; không chỗ nào dùng
tới bộ sinh số ngẫu nhiên.

**4. Rolling-origin, expanding.** Luật purge ở quy ước 2 áp **trong từng fold**, và
không fold nào chạm tới test.

Khung thời gian
---------------

`Khung` gói bốn tham số: độ dài cửa sổ, tỉ lệ train, tỉ lệ validation và số fold.
Khung mặc định `KHUNG_8N` là cửa sổ 8 ngày của Thí nghiệm A: 2.304 bucket, ranh giới
`train [0, 1612)`, `val [1612, 1957)`, `test [1957, 2304)`.

Thí nghiệm B dùng cửa sổ dài hơn và không chia ba tập cố định, nên nó gọi
`mask_khoang()` với ranh giới của từng fold thay vì `split_masks()`. Mọi nơi cần biết
"phần train kết thúc ở đâu" đều phải nhận ranh giới đó từ ngoài vào, không tự giả
định 1.612: xem `preprocess.filter.judge`, `preprocess.normalize.thong_ke_train` và
`evaluation.tang.bang_cv`.

Về `b0`
-------

Offset tính so với `b0` = bucket **nhỏ nhất của môi trường**, đọc từ
`data/processed/{env}.parquet`. **Không** lấy `min(bucket)` của ma trận đặc trưng:
24 dòng đầu mỗi chuỗi không hợp lệ nên bucket nhỏ nhất ở đó là `b0 + 24`, và lấy
nhầm sẽ đẩy cả ba ranh giới đi 24 bucket mà không có gì báo lỗi. Vì vậy `b0` là
tham số **bắt buộc**; dùng `doc_b0()` để lấy đúng.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
NHAN_LOAI = "loai"  # dòng vắt qua ranh giới, không thuộc tập nào

# Số bucket trong một ngày trên lưới 5 phút.
BUCKET_MOI_NGAY = 288


@dataclass(frozen=True)
class Khung:
    """Khung thời gian của một thí nghiệm, tính bằng bucket."""

    window_buckets: int
    train_frac: float = 0.70
    val_frac: float = 0.15
    n_folds: int = 5
    ten: str = ""

    @property
    def n_train(self) -> int:
        return int(self.window_buckets * self.train_frac)

    @property
    def n_val(self) -> int:
        return int(self.window_buckets * self.val_frac)

    def kiem(self) -> "Khung":
        if self.window_buckets <= 0:
            raise ValueError(f"window_buckets phải > 0, nhận {self.window_buckets}.")
        if self.n_val % self.n_folds:
            raise ValueError(
                f"Vùng validation rộng {self.n_val} bucket không chia hết cho "
                f"{self.n_folds} fold."
            )
        return self


# Khung mặc định: cửa sổ 8 ngày của Thí nghiệm A.
KHUNG_8N = Khung(window_buckets=8 * BUCKET_MOI_NGAY, ten="8 ngay").kiem()

# Hằng số giữ lại cho mã cũ; chúng luôn là của khung mặc định.
WINDOW_BUCKETS = KHUNG_8N.window_buckets
TRAIN_FRAC = KHUNG_8N.train_frac
VAL_FRAC = KHUNG_8N.val_frac
N_TRAIN = KHUNG_8N.n_train
N_VAL = KHUNG_8N.n_val
N_FOLDS = KHUNG_8N.n_folds

assert N_TRAIN == 1612, "khung 8 ngày: train phải là [0, 1612)"
assert N_VAL == 345, "khung 8 ngày: validation phải rộng 345 bucket"


def _khung(khung: "Khung | None") -> Khung:
    return KHUNG_8N if khung is None else khung.kiem()


def khung_theo_ngay(so_ngay: float, **kw) -> Khung:
    """Khung dài `so_ngay` ngày trên lưới 5 phút."""
    return Khung(window_buckets=int(round(so_ngay * BUCKET_MOI_NGAY)), **kw).kiem()


def bucket_bounds(khung: "Khung | None" = None) -> dict:
    """Ba ranh giới theo offset so với `b0`, nửa mở `[lo, hi)`."""
    k = _khung(khung)
    return {
        "train": (0, k.n_train),
        "val": (k.n_train, k.n_train + k.n_val),
        "test": (k.n_train + k.n_val, k.window_buckets),
    }


def fold_bounds(n_splits: "int | None" = None,
                khung: "Khung | None" = None) -> list:
    """Ranh giới của rolling-origin, **expanding**, bên trong vùng validation.

    Train của fold `i` là `[0, bien[i])`, validation là `[bien[i], bien[i+1])`. Train
    của fold sau chứa trọn train của fold trước; validation của fold nào cũng nằm
    **sau** train của chính nó.
    """
    k = _khung(khung)
    n_splits = k.n_folds if n_splits is None else n_splits
    lo, hi = bucket_bounds(k)["val"]
    rong = hi - lo
    if rong % n_splits:
        raise ValueError(
            f"Vùng validation rộng {rong} bucket không chia hết cho {n_splits} fold."
        )
    buoc = rong // n_splits
    bien = [lo + i * buoc for i in range(n_splits + 1)]
    return [{"train": (0, bien[i]), "val": (bien[i], bien[i + 1])}
            for i in range(n_splits)]


def fold_bounds_truot(moc_dau: int, moc_cuoi: int, buoc: int) -> list:
    """Rolling-origin cho một vùng bất kỳ, dùng cho Thí nghiệm B.

    Chia `[moc_dau, moc_cuoi)` thành các đoạn rộng `buoc`; fold `i` huấn luyện trên
    `[0, moc_dau + i*buoc)` và chấm trên đoạn kế tiếp. Các đoạn chấm rời nhau và phủ
    kín vùng đã cho.
    """
    if buoc <= 0:
        raise ValueError(f"buoc phải > 0, nhận {buoc}.")
    ra = []
    o = int(moc_dau)
    while o < int(moc_cuoi):
        het = min(o + buoc, int(moc_cuoi))
        ra.append({"train": (0, o), "val": (o, het)})
        o = het
    return ra


def doc_b0(env: str, processed_dir="data/processed") -> int:
    """`b0` của môi trường: bucket nhỏ nhất trên **toàn bộ** chuỗi."""
    p = Path(processed_dir) / f"{env}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Thiếu {p}; chạy bước tiền xử lý để có b0 của {env}."
        )
    return int(pd.read_parquet(p, columns=["bucket"])["bucket"].min())


def offset(bucket, b0: int) -> np.ndarray:
    """Offset của bucket so với `b0`, kiểu `int64`. Không kẹp và không lọc."""
    return np.asarray(bucket, dtype="int64") - int(b0)


def _trong_khoang(off: np.ndarray, h: int, lo: int, hi: int) -> np.ndarray:
    """Luật purge: `lo <= t` **và** `t + h < hi`."""
    return (off >= lo) & (off + h < hi)


def split_masks(off, h: int, khung: "Khung | None" = None) -> dict:
    """Ba mặt nạ bool cho `train`, `val`, `test` — đã purge dòng vắt ranh giới."""
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")
    off = np.asarray(off, dtype="int64")
    return {ten: _trong_khoang(off, h, lo, hi)
            for ten, (lo, hi) in bucket_bounds(khung).items()}


def mask_khoang(off, h: int, lo: int, hi: int) -> np.ndarray:
    """Mặt nạ cho một khoảng bucket tuỳ ý, vẫn áp luật purge."""
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")
    return _trong_khoang(np.asarray(off, dtype="int64"), h, int(lo), int(hi))


def fit_mask(off, h: int, khung: "Khung | None" = None) -> np.ndarray:
    """Vùng huấn luyện của model **cuối cùng**: train + validation."""
    k = _khung(khung)
    return mask_khoang(off, h, 0, k.n_train + k.n_val)


def fold_masks(off, h: int, n_splits: "int | None" = None,
               khung: "Khung | None" = None) -> list:
    """Mặt nạ train/val của từng fold rolling-origin."""
    if h < 1:
        raise ValueError(f"Horizon phải >= 1, nhận {h}.")
    off = np.asarray(off, dtype="int64")
    return [{ten: _trong_khoang(off, h, lo, hi) for ten, (lo, hi) in bien.items()}
            for bien in fold_bounds(n_splits, khung)]


def gan_nhan(off, h: int, khung: "Khung | None" = None) -> np.ndarray:
    """Nhãn tập của từng dòng: `train | val | test | loai`."""
    mn = split_masks(off, h, khung)
    nhan = np.full(len(np.asarray(off)), NHAN_LOAI, dtype=object)
    for ten in SPLIT_NAMES:
        nhan[mn[ten]] = ten
    return nhan


def split_indices(off, h: int, khung: "Khung | None" = None) -> dict:
    """Như `split_masks()` nhưng trả về chỉ số vị trí, cho phía cần `iloc`."""
    return {ten: np.flatnonzero(m) for ten, m in split_masks(off, h, khung).items()}


def dem_dong(off, h: int, khung: "Khung | None" = None) -> dict:
    """Số dòng mỗi tập, kèm số dòng bị purge ở khoá `loai`."""
    mn = split_masks(off, h, khung)
    dem = {ten: int(m.sum()) for ten, m in mn.items()}
    dem[NHAN_LOAI] = int(len(np.asarray(off)) - sum(dem.values()))
    return dem
