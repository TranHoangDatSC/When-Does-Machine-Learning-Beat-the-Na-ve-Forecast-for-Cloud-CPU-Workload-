"""Tên và thứ tự của 20 đặc trưng.

`config/features.yaml` mô tả cùng bộ đặc trưng bằng lời; tên cột và thứ tự chính
thức là ở tệp này. Thêm hoặc bớt một đặc trưng sẽ làm kết quả không còn so được với
các lần chạy trước.

`lag_0` là giá trị CPU tại chính thời điểm phát dự báo `t`. Dự báo naive dùng đúng
giá trị đó, nên nếu mô hình học máy không có nó trong đầu vào thì hai bên không nhìn
thấy cùng một lượng thông tin. Trước đây `y_t` chỉ dựng lại được bằng tổng
`lag_1 + diff_1`: hồi quy tuyến tính biểu diễn được tổng đó bằng hai hệ số, còn mô
hình cây phải xấp xỉ bằng nhiều lần chia, nên bất lợi rơi vào riêng nhóm cây.
"""

# Lưới thời gian 5 phút. `bucket * GRID_SECONDS` là số giây.
GRID_SECONDS = 300

# Cửa sổ đặc trưng sâu nhất, dùng cho luật dòng hợp lệ.
MAX_LAG = 24

LAGS = (0, 1, 2, 3, 6, 12, 24)
ROLL_WINDOWS = (6, 12)
ROLL_STATS = ("mean", "std", "min", "max")

# ddof = 1 cho roll_std_*. pandas mặc định 1, numpy mặc định 0, nên phải ghi rõ.
ROLL_DDOF = 1

# Epoch 1970-01-01 là thứ Năm, nên ((t // 86400) + 4) % 7 cho 0 là Chủ nhật.
DOW_EPOCH_OFFSET = 4

LAG_COLS = [f"lag_{k}" for k in LAGS]
ROLL_COLS = [f"roll_{stat}_{w}" for w in ROLL_WINDOWS for stat in ROLL_STATS]
DIFF_COLS = ["diff_1"]
CALENDAR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

FEATURE_COLS = LAG_COLS + ROLL_COLS + DIFF_COLS + CALENDAR_COLS

ID_COLS = ["series_id", "bucket"]
TARGET_COL = "target"

# Schema tệp data/features/{env}_h{h}.parquet: đúng 23 cột, không thừa cột nào.
MATRIX_COLS = ID_COLS + FEATURE_COLS + [TARGET_COL]

# Cột bắt buộc của bảng đầu vào data/processed/{env}.parquet.
INPUT_COLS = ["series_id", "bucket", "y"]

assert len(FEATURE_COLS) == 20, "phải có đúng 20 đặc trưng"
assert len(MATRIX_COLS) == 23, "20 đặc trưng + series_id + bucket + target"
