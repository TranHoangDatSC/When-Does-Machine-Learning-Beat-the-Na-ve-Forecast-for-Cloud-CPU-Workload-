"""Phân tầng độ giật (burstiness) để **báo cáo** kết quả.

Ba tầng, chia bằng **tam phân vị của CV tính riêng trong từng môi trường**. Ngưỡng
không ghi cứng: chúng được tính lại từ `results/tables/cv.csv` mỗi lần chạy, và cách
chia đọc từ `config/split.yaml` (`stratify.method`, `stratify.n_strata`).

Không dùng một ngưỡng tuyệt đối chung cho ba môi trường, vì hai lý do đo được:

- Tương quan giữa CV và mức tải **đổi dấu** giữa Bitbrains (ρ = +0,375 và +0,249) và
  Alibaba (ρ = −0,692), nên một ngưỡng chung sẽ chọn ra hai nhóm máy khác loại.
- Tam phân vị của E3 chỉ rộng 0,06 và nằm gọn trong tầng thấp nhất của E1.

CV dùng để chia tầng tính trên phần train
------------------------------------------

`cv.csv` có hai cột: `cv` tính trên **toàn bộ** cửa sổ 8 ngày, và `cv_train` tính
trên **riêng phần train**. Việc chia tầng dùng `cv_train`, vì tầng chạm vào phần
huấn luyện qua mẫu con của SVR: mẫu con được lấy phân tầng theo ba tầng đó. Dùng cột
`cv` cho mục đích ấy là để thông tin của validation và test lọt vào khâu chọn dòng
huấn luyện.

Cột `cv` vẫn giữ, chỉ dùng khi mô tả dữ liệu trong bài.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cwp.evaluation.splits import bucket_bounds, offset

# Nhãn ba tầng, theo thứ tự tăng dần của CV.
TEN_TANG = ("thap", "vua", "cao")

# Cột dùng để chia tầng. Phải là cột tính trên phần train.
COT_TANG = "cv_train"


def doc_cau_hinh_tang(config_path: str | Path = "config/split.yaml") -> dict:
    """Đọc `stratify` từ `config/split.yaml`. Ngưỡng **không** nằm trong config."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    st = cfg.get("stratify", {})
    if st.get("method") != "percentile_within_env":
        raise ValueError(
            f"config/split.yaml: stratify.method = {st.get('method')!r}, "
            "chỉ hỗ trợ 'percentile_within_env'."
        )
    return st


def bang_cv(df: pd.DataFrame, env: str, b0: int, khung=None) -> pd.DataFrame:
    """Một dòng mỗi chuỗi: CV trên cả cửa sổ và CV trên riêng phần train.

    Dùng chung cho `scripts/fig_burstiness.py` và `scripts/build_may_gia.py`, để hai
    bảng `cv.csv` và `cv_may_gia.csv` không lệch định nghĩa.

    Parameters
    ----------
    df : pd.DataFrame
        Bảng dài `data/processed/{env}.parquet`, cần `series_id`, `bucket`, `y`.
    env : str
        Mã môi trường, ghi vào cột `env`.
    b0 : int
        Bucket đầu của môi trường, từ `splits.doc_b0()`.

    Returns
    -------
    pd.DataFrame
        `series_id, env, mean, std, n_diem, cv, cv_chan, ti_le_cham_chan,
        mean_train, std_train, n_diem_train, cv_train`.
    """
    thieu = [c for c in ("series_id", "bucket", "y") if c not in df.columns]
    if thieu:
        raise ValueError(f"Bảng đầu vào thiếu cột {thieu}.")

    g = df.groupby("series_id")["y"]
    out = pd.DataFrame({
        "env": env,
        "mean": g.mean(),
        "std": g.std(ddof=1),
        "n_diem": g.count(),
    }).reset_index()
    out["cv"] = out["std"] / out["mean"]

    # CPU% bị chặn trong [0, 100], nên một biến có trung bình m không thể có phương
    # sai vượt m(100 − m), cực đại đạt được bởi phân phối hai điểm ở 0 và 100. Suy
    # ra CV <= sqrt((100 − m)/m). Đây là chặn của THANG ĐO, không phải của dữ liệu.
    out["cv_chan"] = np.sqrt((100.0 - out["mean"]) / out["mean"])
    out["ti_le_cham_chan"] = out["cv"] / out["cv_chan"]

    lo, hi = bucket_bounds(khung)["train"]
    off = offset(df["bucket"], b0)
    tr = df.loc[(off >= lo) & (off < hi)].groupby("series_id")["y"]
    train = pd.DataFrame({
        "mean_train": tr.mean(),
        "std_train": tr.std(ddof=1),
        "n_diem_train": tr.count(),
    })
    train["cv_train"] = train["std_train"] / train["mean_train"]
    return out.merge(train.reset_index(), on="series_id", how="left")


def nguong(cv: np.ndarray, n_strata: int = 3) -> list[float]:
    """`n_strata − 1` phân vị chia đều, tính **trong một môi trường**.

    Với `n_strata = 3` thì đây là tam phân vị: phân vị 33,33% và 66,67%.
    """
    if n_strata < 2:
        raise ValueError(f"n_strata phải >= 2, nhận {n_strata}.")
    q = [100.0 * i / n_strata for i in range(1, n_strata)]
    return [float(x) for x in np.percentile(np.asarray(cv, dtype="float64"), q)]


def gan_tang(
    cv_bang: pd.DataFrame, env: str, n_strata: int = 3,
) -> tuple[pd.Series, list[float]]:
    """Nhãn tầng của từng chuỗi trong một môi trường, kèm ngưỡng đã dùng.

    Parameters
    ----------
    cv_bang : pd.DataFrame
        `results/tables/cv.csv`, cần cột `series_id`, `env` và `cv_train`.
    env : str
        Mã môi trường.

    Returns
    -------
    (pd.Series, list[float])
        Series index `series_id`, giá trị là một trong `TEN_TANG`; và danh sách ngưỡng.
    """
    con = cv_bang[cv_bang["env"] == env]
    if con.empty:
        raise ValueError(f"cv.csv không có dòng nào của {env}.")
    if COT_TANG not in con.columns:
        raise ValueError(
            f"cv.csv thiếu cột {COT_TANG}. Sinh lại bằng scripts/fig_burstiness.py: "
            "tầng phải chia theo CV của phần train."
        )
    v = con[COT_TANG].to_numpy("float64")
    nguong_env = nguong(v, n_strata)
    chi_so = np.digitize(v, nguong_env)
    ten = list(TEN_TANG) if n_strata == 3 else [f"t{i}" for i in range(n_strata)]
    return (
        pd.Series([ten[i] for i in chi_so], index=con["series_id"].to_numpy(),
                  name="tang"),
        nguong_env,
    )


def bang_tang(
    cv_bang: pd.DataFrame, envs=("E1", "E2", "E3"), n_strata: int = 3,
) -> pd.DataFrame:
    """Bảng dài `series_id, env, tang, cv, cv_train, ti_le_cham_chan`.

    `ti_le_cham_chan` = `CV / √((100−m)/m)` lấy thẳng từ `cv.csv`. Mọi bảng phân tầng
    báo kèm cột này để phân biệt một chuỗi ít giật **do bản chất** với một chuỗi ít
    giật **vì đã chạm trần thang đo**.
    """
    phan = []
    for env in envs:
        tang, _ = gan_tang(cv_bang, env, n_strata)
        con = cv_bang[cv_bang["env"] == env].set_index("series_id")
        phan.append(pd.DataFrame({
            "series_id": tang.index,
            "env": env,
            "tang": tang.to_numpy(),
            "cv": con.loc[tang.index, "cv"].to_numpy(),
            "cv_train": con.loc[tang.index, COT_TANG].to_numpy(),
            "ti_le_cham_chan": con.loc[tang.index, "ti_le_cham_chan"].to_numpy(),
        }))
    return pd.concat(phan, ignore_index=True)
