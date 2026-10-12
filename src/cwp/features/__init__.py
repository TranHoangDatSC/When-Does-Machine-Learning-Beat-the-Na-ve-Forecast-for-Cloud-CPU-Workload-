"""Sinh 19 đặc trưng cho mỗi thời điểm `t` của một chuỗi.

Chỉ dùng lịch sử của chính chuỗi đó. Không lấy thông tin từ chuỗi khác, không chạm
`y` tại `t+1` trở đi, và **không lấp thêm gì**: lỗ hổng ngắn đã được nội suy ở bước
tiền xử lý và đánh dấu bằng cột `is_interp`.

Cách dùng:

    import pandas as pd
    from cwp.features import make_feature_matrix

    df = pd.read_parquet("data/processed/E1.parquet")
    X = make_feature_matrix(df, h=1)      # 22 cột, đã lọc dòng hợp lệ
"""

from cwp.features.calendar import (
    add_calendar_features,
    bucket_to_seconds,
    day_of_week,
    hour_of_day,
)
from cwp.features.matrix import (
    build_feature_frame,
    make_feature_matrix,
    prepare_input,
    valid_row_mask,
)
from cwp.features.spec import (
    CALENDAR_COLS,
    DIFF_COLS,
    FEATURE_COLS,
    GRID_SECONDS,
    ID_COLS,
    LAG_COLS,
    MATRIX_COLS,
    MAX_LAG,
    ROLL_COLS,
    ROLL_DDOF,
    TARGET_COL,
)
from cwp.features.windows import (
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    add_target,
    assert_contiguous_grid,
)

__all__ = [
    # đặc tả tên cột
    "FEATURE_COLS",
    "LAG_COLS",
    "ROLL_COLS",
    "DIFF_COLS",
    "CALENDAR_COLS",
    "ID_COLS",
    "TARGET_COL",
    "MATRIX_COLS",
    "GRID_SECONDS",
    "MAX_LAG",
    "ROLL_DDOF",
    # sản phẩm chính
    "make_feature_matrix",
    "build_feature_frame",
    "valid_row_mask",
    "prepare_input",
    # từng nhóm đặc trưng
    "add_lag_features",
    "add_rolling_features",
    "add_diff_features",
    "add_calendar_features",
    "add_target",
    # tiện ích lịch và kiểm lưới
    "bucket_to_seconds",
    "hour_of_day",
    "day_of_week",
    "assert_contiguous_grid",
]
