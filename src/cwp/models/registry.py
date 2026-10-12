"""Năm mô hình học máy và lưới siêu tham số.

    lr       Linear Regression   — mốc dưới của nhóm học máy, không có siêu tham số
    ridge    Ridge               — mốc dưới thứ hai, một siêu tham số
    rf       Random Forest
    xgb      XGBoost             — hàm mất mát bình phương
    xgb_mae  XGBoost             — hàm mất mát tuyệt đối, khác `xgb` đúng một chỗ
    svr      SVR nhân RBF        — chỉ học trên mẫu con

Hàm mất mát
-----------

Chỉ cặp `xgb` và `xgb_mae` tạo thành một lưới hai hàm mất mát: cùng kiến trúc, cùng
lưới siêu tham số, cùng dữ liệu, khác đúng tham số `objective`. Ridge và SVR giữ hàm
mất mát riêng của chúng, lần lượt là bình phương kèm phạt L2 và ε-insensitive, nên
**không** được đọc như hai ô của cùng lưới đó.

**Global model**: một model học trên mọi chuỗi của cùng một môi trường, không phải
mỗi chuỗi một model.

Cách chọn lưới siêu tham số
---------------------------

Lưới được chọn theo **chi phí đo được**. Đo trên tổ hợp nặng nhất, E1 `h = 1`,
1.142.276 dòng train, 12 nhân:

    ridge  alpha                               0,1 s  mỗi lần khớp
    xgb    300 cây, depth 6, tree_method=hist  8,7 s
    rf     50 cây,  depth 12, msl=5           84,5 s
    rf     100 cây, depth 8,  msl=5          123,2 s
    rf     100 cây, depth 16, msl=5          229,7 s
    svr    10.000 dòng mẫu con                ~1,5 s khớp, ~70 s dự đoán 200k dòng

Rolling-origin 5 fold nhân chi phí lên khoảng 5,4 lần (fold cuối train dài hơn fold
đầu 17%), rồi còn nhân với số ứng viên và 9 tổ hợp. Vì vậy:

- `rf` giữ **50 cây** thay vì 100, và lưới chỉ hai ứng viên `max_depth`. Với 100 cây
  và hai ứng viên, riêng phần dò siêu tham số của RF đã mất hơn 3 giờ.
- `xgb` ba ứng viên: nó rẻ nên lưới rộng hơn được.
- `svr` hai ứng viên `C` trên mẫu con 10.000 dòng. Chỗ tốn của SVR là **dự đoán**, không
  phải huấn luyện: số vector hỗ trợ tăng theo cỡ mẫu, nên với mẫu 30.000 dòng đã mất
  228 giây chỉ để dự đoán 200.000 dòng test.

**50 cây là một bất lợi của RF trong các bảng kết quả**, cũng như mẫu con là bất lợi
của SVR. Không nên đọc "RF thua XGBoost" ở Thí nghiệm 1 như một kết luận về hai họ
thuật toán.

Chuẩn hoá đặc trưng
-------------------

`lr`, `ridge`, `svr` đi qua `StandardScaler`; `rf` và `xgb` không cần. Scaler nằm
**trong** pipeline nên nó khớp trên đúng phần train của từng fold. Đặt ngoài pipeline
thì thống kê của phần validation lọt vào lúc khớp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# random_state = 42 ở mọi chỗ có yếu tố ngẫu nhiên.
SEED = 42

ML_NAMES = ("lr", "ridge", "rf", "xgb", "xgb_mae", "svr")

# Nhãn hàm mất mát, ghi vào tệp dự đoán và vào bảng kết quả.
LOSS_BINH_PHUONG = "binh_phuong"
LOSS_TUYET_DOI = "tuyet_doi"
LOSS_EPSILON = "epsilon_insensitive"


@dataclass(frozen=True)
class ModelSpec:
    """Một model: cách dựng, lưới siêu tham số, và cỡ mẫu con nếu có."""

    ten: str
    nhan: str
    dung: Callable[[dict], Any]
    luoi: list[dict] = field(default_factory=lambda: [{}])
    mau_con: int | None = None   # số dòng train tối đa; None là dùng hết
    loss: str = LOSS_BINH_PHUONG

    @property
    def co_sieu_tham_so(self) -> bool:
        return len(self.luoi) > 1


def _pipeline_chuan_hoa(bo_cuoi):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scale", StandardScaler()), ("model", bo_cuoi)])


def _lr(p: dict):
    from sklearn.linear_model import LinearRegression

    return _pipeline_chuan_hoa(LinearRegression(**p))


def _ridge(p: dict):
    from sklearn.linear_model import Ridge

    return _pipeline_chuan_hoa(Ridge(**p))


def _rf(p: dict):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=SEED, **p
    )


def _xgb(p: dict):
    from xgboost import XGBRegressor

    return XGBRegressor(
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=SEED, **p
    )


def _xgb_mae(p: dict):
    """Giống `_xgb` đúng một tham số: `objective`.

    `reg:absoluteerror` có từ xgboost 1.7; bản ghim trong requirements.txt là 2.1.3.
    """
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:absoluteerror",
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=SEED, **p
    )


def _svr(p: dict):
    from sklearn.svm import SVR

    return _pipeline_chuan_hoa(SVR(kernel="rbf", gamma="scale", **p))


REGISTRY: dict[str, ModelSpec] = {
    "lr": ModelSpec("lr", "Linear Regression", _lr, [{}]),
    "ridge": ModelSpec("ridge", "Ridge", _ridge,
                       [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0)]),
    "rf": ModelSpec("rf", "Random Forest", _rf,
                    [{"max_depth": d} for d in (8, 16)]),
    "xgb": ModelSpec("xgb", "XGBoost", _xgb,
                     [{"max_depth": 4, "n_estimators": 300},
                      {"max_depth": 8, "n_estimators": 300},
                      {"max_depth": 8, "n_estimators": 600}]),
    # Khác "xgb" đúng tham số objective, giữ nguyên lưới để so được hai hàm mất mát.
    "xgb_mae": ModelSpec("xgb_mae", "XGBoost (MAE)", _xgb_mae,
                         [{"max_depth": 4, "n_estimators": 300},
                      {"max_depth": 8, "n_estimators": 300},
                      {"max_depth": 8, "n_estimators": 600}],
                         loss=LOSS_TUYET_DOI),
    # Mẫu con lấy trên DÒNG huấn luyện, test giữ nguyên 100%.
    "svr": ModelSpec("svr", "SVR RBF", _svr,
                     [{"C": 1.0}, {"C": 10.0}], mau_con=10_000,
                     loss=LOSS_EPSILON),
}


def mau_con_phan_tang(
    tang: np.ndarray, n: int, seed: int = SEED
) -> np.ndarray:
    """Chỉ số của mẫu con **phân tầng theo ba tầng CV**, giữ nguyên tỉ lệ.

    Lấy mẫu trên **dòng**, không trên chuỗi: bỏ bớt chuỗi là đổi quần thể, và mọi kết
    luận mức chuỗi sẽ không so được với các model khác. Bỏ bớt dòng chỉ làm SVR học
    từ ít mẫu hơn, là một bất lợi *của SVR*.

    Parameters
    ----------
    tang : np.ndarray
        Nhãn tầng của từng dòng (`thap | vua | cao`), suy từ `series_id`.
    n : int
        Số dòng muốn lấy. `n >= len(tang)` thì trả về toàn bộ chỉ số.
    seed : int, optional
        `random_state`, mặc định 42.

    Returns
    -------
    np.ndarray
        Chỉ số đã sắp tăng, giữ thứ tự thời gian của bảng gốc.
    """
    tang = np.asarray(tang)
    tong = len(tang)
    if n >= tong:
        return np.arange(tong)

    rng = np.random.default_rng(seed)
    chon = []
    nhan, dem = np.unique(tang, return_counts=True)
    # Phân bổ theo tỉ lệ, phần dư dồn cho tầng lớn nhất để tổng đúng bằng `n`.
    quota = {t: int(n * c / tong) for t, c in zip(nhan, dem)}
    thieu = n - sum(quota.values())
    if thieu:
        quota[nhan[int(np.argmax(dem))]] += thieu
    for t in nhan:
        vi_tri = np.flatnonzero(tang == t)
        k = min(quota[t], len(vi_tri))
        chon.append(rng.choice(vi_tri, size=k, replace=False))
    return np.sort(np.concatenate(chon))
