# Thiết kế thực nghiệm

**Ngày khoá: 23/09/2026.** Tài liệu này chốt mọi lựa chọn thiết kế **trước** lượt chạy xác nhận: câu hỏi, quần thể, cách chia dữ liệu, mô hình, lưới siêu tham số, chỉ số và kiểm định.

Tệp gồm hai phần. Phần chính, từ mục 1 đến mục 11, là thiết kế kèm những ghi nhận thêm sau ngày khoá, mỗi ghi nhận đều mang ngày. **Phụ lục C giữ nguyên văn bản khoá ngày 23/09/2026**, không sửa một chữ, để người đọc đối chiếu được điều đã chốt trước lượt chạy với điều đã thực sự chạy. Không chỗ nào trong tệp này sửa đè lên điều đã khoá.

**Hai sai lệch so với thiết kế đã khoá** được phát hiện trong lúc chạy và được ghi lại đầy đủ: sai lệch D1 về lưới siêu tham số của XGBoost (phát hiện 25/09/2026, sau khi lượt chạy chính đã xong) ở Phụ lục A, và sai lệch D2 về cách lấy mẫu lại của bootstrap ở Thí nghiệm B (phát hiện 27/09/2026) ở Phụ lục B. Cả hai đều được ghi nhận chứ không sửa lại lịch sử. Những chỗ làm rõ sau ngày khoá đều mang nhãn ngày.

---

## 1. Câu hỏi và phạm vi

| Mã | Câu hỏi | Thí nghiệm |
|---|---|---|
| RQ1 | Kết luận "mô hình có vượt dự báo naive không" thay đổi bao nhiêu khi đổi chuẩn hoá, hàm mất mát và tiêu chí đánh giá, ở ba tầm dự báo | A |
| RQ2 | Kết luận của RQ1 có giữ nguyên qua các giai đoạn thời gian khác nhau không | B |
| RQ3 | Mô hình học ở một tập vết dùng lại được ở tập vết khác đến mức nào | C |

Phân tích bổ sung: máy giả, đối chứng không nội suy, độ nhạy ngưỡng lọc quần thể, Random Forest phạm vi hẹp, chi phí bất đối xứng, phân nhóm máy ở mức thăm dò.

---

## 2. Dữ liệu và quần thể

| Môi trường | Nguồn | Cửa sổ | Vai trò |
|---|---|---|---|
| E1 | Bitbrains fastStorage | 8 ngày từ 12/08/2013 13:40 | Thí nghiệm A và C |
| E2 | Bitbrains Rnd tháng 8 | 8 ngày từ 31/07/2013 22:00 | Thí nghiệm A và C |
| E3 | Alibaba 2018 | 8 ngày từ đầu trace | Thí nghiệm A và C |
| E1L | Bitbrains fastStorage | 30 ngày | Thí nghiệm B, xác nhận |
| E2m08 | Bitbrains Rnd tháng 8 | 30 ngày | Thí nghiệm B, **phát triển** |
| E2m07 | Bitbrains Rnd tháng 7 | 30 ngày | Thí nghiệm B, xác nhận |
| E2m09 | Bitbrains Rnd tháng 9 | 29 ngày | Thí nghiệm B, xác nhận |

Ba tháng của Rnd **không nối lại**. Bằng chứng: trên mẫu 60 máy chỉ 31 máy giữ nguyên cả số nhân, xung nhịp và dung lượng RAM qua ba tháng, và giữa hai tháng liên tiếp có một ngày trống. Kết luận ghi đúng mức: **chưa đủ bằng chứng để nối an toàn**; thay đổi cấu hình phần cứng không tự chứng minh đó là máy khác.

**Quần thể chính.** Một chuỗi được giữ khi phần lịch sử dùng để quyết định thoả cả bốn điều: có ít nhất một điểm, CPU trung bình từ 1% trở lên, có ít nhất 3 giá trị phân biệt, và có ít nhất 500 dòng huấn luyện hợp lệ ở tầm 12. Phần lịch sử đó là `[0, 1612)` với cửa sổ 8 ngày, và `[0, 4032)` tức 14 ngày đầu với các môi trường dài ngày. Phạm vi phát biểu vì vậy là **máy đã hoạt động trong lịch sử**.

Quần thể sau khi lọc: E1 708 chuỗi, E2 302, E3 487.

**Độ nhạy ngưỡng.** Ba ngưỡng CPU trung bình phần train chốt trước: 0%, 0,5% và 2%, trong đó 0% nghĩa là bỏ tiêu chí mức tải nhưng giữ ba tiêu chí chất lượng còn lại. Ngưỡng **không** được chọn theo kết quả.

---

## 3. Tiền xử lý

1. Cắt CPU về `[0, 100]`.
2. Đưa về lưới 5 phút bằng trung bình các mẫu trong mỗi ô.
3. Cắt cửa sổ chung của môi trường.
4. Nội suy tuyến tính các cụm thiếu dài tối đa 2 điểm; cụm dài hơn giữ nguyên là thiếu.
5. Lọc chuỗi theo mục 2.

**Luật nhân quả với giá trị nội suy.** Nội suy dùng đầu mút bên phải, nên một điểm nội suy chỉ tính được khi đầu mút đó đã tới. Hệ quả áp trong luật dòng hợp lệ: dòng tại `t` bị loại nếu `y(t)` là giá trị nội suy hoặc `y(t+h)` là giá trị nội suy. Điểm nội suy nằm hẳn trước `t` vẫn dùng được, vì khi `y(t)` là quan sát thật thì đầu mút phải của chúng không vượt quá `t`. Bản đối chứng `K = 0` không nội suy gì và dùng để đo phần còn lại.

---

## 4. Đặc trưng và chia dữ liệu

20 đặc trưng, tất cả chỉ lấy từ quá khứ của chính máy đó: `lag_0` đến `lag_24` (7 giá trị trễ, trong đó `lag_0` là `y(t)`), 8 thống kê cửa sổ 6 và 12 điểm, sai phân một bước, 4 đặc trưng lịch.

Dòng tại `t` hợp lệ khi mọi điểm trong `[t-24, t]` không thiếu, `y(t+h)` không thiếu, và luật nhân quả ở mục 3 thoả.

**Khung 8 ngày**: train `[0, 1612)`, validation `[1612, 1957)`, test `[1957, 2304)`. Một dòng thuộc một tập khi cả `t` và `t+h` nằm trong tập đó; dòng vắt qua ranh giới bị loại.

**Khung dài ngày cho Thí nghiệm B**: lịch sử tối thiểu 14 ngày, huấn luyện lại mỗi 7 ngày. Fold `i` khớp trên `[0, o_i)` và chấm trên `[o_i, o_i + 2016)`, với `o_i = 4032 + 2016·i`. Các đoạn chấm rời nhau và phủ kín phần còn lại của cửa sổ.

**Fold cuối ngắn hơn 7 ngày (làm rõ 27/09/2026).** Vì các đoạn chấm phủ kín phần còn lại của cửa sổ, đoạn cuối bị cắt tại biên cửa sổ: `[o_i, min(o_i + 2016, W))`. Thực tế ở K6.3 mỗi môi trường có 3 fold, và fold 2 chỉ chấm **2 ngày** với E1L và E2m07 (cửa sổ 30 ngày, `W = 8640`) hoặc **1 ngày** với E2m09 (29 ngày, `W = 8352`). Khi báo kết quả phải ghi số ngày của từng fold, không được phát biểu như thể mọi fold dài 7 ngày.

---

## 5. Chuẩn hoá

| Chế độ | Biến đổi | Thống kê |
|---|---|---|
| N0 | giữ nguyên CPU% | không |
| N1 | z-score theo từng máy | `mu`, `sd` của **phần train của chính máy đó**, ranh giới theo khung đang dùng |

N1 dựng bằng phép affine trên ma trận N0, đúng bằng cách sinh lại đặc trưng từ chuỗi đã z-score; hai đường được đối chiếu ở phép kiểm K13. Máy có `sd` không dương bị loại khỏi cả huấn luyện lẫn chấm ở N1, và số máy bị loại được ghi ra.

Chế độ N2, tức sai phân, nằm ở phần bổ sung, không thuộc lưới chính.

Mọi dự đoán đưa về thang CPU% gốc trước khi tính bất kỳ chỉ số nào.

---

## 6. Mô hình và lưới siêu tham số

| Mô hình | Hàm mất mát | Lưới | Ghi chú |
|---|---|---|---|
| naive | không | — | `ŷ = y(t)` |
| ma6 | không | — | trung bình `y[t-5 .. t]` |
| seasonal | không | — | `y(t+h-288)` |
| seasonal_tuan | không | — | `y(t+h-2016)`, chỉ Thí nghiệm B |
| ridge | bình phương kèm phạt L2 | `alpha` ∈ {0,01; 0,1; 1; 10; 100} | giữ hàm mất mát riêng, không thuộc lưới hai hàm mất mát |
| xgb | bình phương | **Đã khoá 23/09/2026:** `max_depth` ∈ {4, 8}, 300 cây. **Đã chạy ở K4.1:** ba ứng viên (4, 300), (8, 300), (8, 600) | sai lệch D1, xem Phụ lục A |
| xgb_mae | tuyệt đối | **Đã khoá 23/09/2026:** `max_depth` ∈ {4, 8}, 300 cây. **Đã chạy ở K4.1:** cùng ba ứng viên với `xgb` | khác `xgb` **đúng một tham số**; sai lệch D1 |
| svr | ε-insensitive | `C` ∈ {1, 10}, mẫu con 10.000 dòng | giữ hàm mất mát riêng |
| dlinear | tuyệt đối | một cấu hình cố định | cửa sổ 288, 3 hạt giống |
| gru | tuyệt đối | một cấu hình cố định | chỉ trên Colab, 4 tổ hợp |
| rf | bình phương | một cấu hình định trước | 50 cây, `max_depth` 16, chỉ để nối với bảng cũ |

Lưới hai hàm mất mát **chỉ gồm cặp `xgb` và `xgb_mae`**. Ridge và SVR giữ hàm mất mát riêng của chúng và không được đọc như hai ô tương đương của cùng lưới đó.

**Lưới XGBoost (ghi nhận 25/09/2026).** Thiết kế khoá ngày 23/09/2026 quy định lưới hai ứng viên (`max_depth` 4 và 8, 300 cây) để giảm chi phí, vì hàm mất mát tuyệt đối đắt hơn hàm bình phương khoảng 7,6 lần. **Lưới đã thực sự chạy ở K4.1 có ba ứng viên**, gồm thêm (8, 600), vì `src/cwp/models/registry.py` chưa được sửa theo lưới hai ứng viên. Đây là một sai lệch protocol, không phải một lựa chọn được chốt từ đầu; xem Phụ lục A. Khi viết bài, phần phương pháp phải mô tả **lưới đã chạy** (ba ứng viên) và nêu sai lệch này, không được mô tả là lưới hai ứng viên.

**Hạt giống.** `random_state = 42` cho mọi mô hình bảng, một hạt giống. DLinear chạy ba hạt giống 42, 43, 44 và báo kèm độ phân tán. GRU chạy một hạt giống 42.

**Chuẩn hoá của mô hình chuỗi.** DLinear và GRU dùng **N1**, tức thống kê phần train của từng máy, giống hệt mô hình bảng. Không dùng chuẩn hoá theo từng cửa sổ. Dự đoán đưa về CPU% bằng `ŷ = ẑ·sd + mu`.

---

## 7. Chọn siêu tham số

- Rolling-origin 5 fold **bên trong vùng validation** của khung 8 ngày, expanding.
- Điểm chọn: **trung vị MAE theo chuỗi, trên thang CPU%**.
- Mô hình cuối khớp trên train + validation, rồi chấm test **đúng một lần**.
- Thí nghiệm B và C **không dò lại**: lấy siêu tham số của Thí nghiệm A theo môi trường nguồn. E1L lấy của E1; E2m07, E2m08, E2m09 lấy của E2; máy giả E1a, E1g lấy của E1.
- DLinear chọn số epoch bằng dừng sớm trên validation, rồi khớp lại trên train + validation bằng đúng số epoch đó.

---

## 8. Tập dòng đánh giá chung

Mọi mô hình trong một họ so sánh chấm trên **đúng cùng một tập dòng**. Họ là các mô hình cùng `(thí nghiệm, nguồn, đích, chế độ, tầm, fold)`. Điều kiện lấy giao:

| Điều kiện | Áp khi |
|---|---|
| `bang` — dòng hợp lệ của ma trận đặc trưng | luôn luôn |
| `ngay` — có `y(t+h-288)` là quan sát thật | họ có seasonal theo ngày |
| `tuan` — có `y(t+h-2016)` là quan sát thật | họ có seasonal theo tuần |
| `chuoi` — 288 điểm `[t-287, t]` không thiếu | họ có DLinear hoặc GRU |

Số dòng bị loại theo từng điều kiện báo ở `results/tong_hop/tom_tat_dong.csv` và phải nêu trong bài.

---

## 9. Chỉ số và kiểm định

- **Chỉ số chính: MAE**, tính riêng từng máy rồi lấy trung vị trên tập máy. Không đổi chỉ số chính theo kết quả.
- Báo kèm **RMSE** và **MASE**; MASE dùng mẫu số là sai số naive một bước trên phần train của chính máy đó. **Với Thí nghiệm B (làm rõ 27/09/2026)** mẫu số lấy trên **14 ngày đầu cố định** `[0, 4032)` cho mọi fold, không lấy theo phần train `[0, o_i)` đang lớn dần của từng fold. Lý do: Thí nghiệm B hỏi kết luận có đổi theo thời gian không, nên MASE của các fold phải có cùng mẫu số mới so được với nhau; mẫu số lớn dần sẽ làm MASE đổi vì mẫu số chứ không vì kết quả dự báo. Đây là điều mã đã làm từ đầu (`MOC_TRAIN_DAI = 14 · 288` trong `scripts/tong_hop.py`); ghi lại ở đây cho khớp.
- Báo kèm **độ lớn cải thiện**: tỉ số trung vị theo máy, tỉ lệ máy thắng, và khoảng tin cậy bootstrap 95% của tỉ số trung vị.
- **Kiểm định**: Wilcoxon ghép cặp theo từng máy, mỗi mô hình so với naive.
- **Họ hiệu chỉnh Holm**: các phép so mô hình với naive **trong cùng một họ** ở mục 8. Mức ý nghĩa 0,05.
- Hướng lấy từ **trung vị hiệu ghép cặp**, không phải hiệu hai trung vị.
- Bootstrap lấy mẫu lại **theo máy**; với Thí nghiệm B lấy mẫu lại **theo khối ngày** — **điều này chưa được thực hiện, xem Phụ lục B (D2)**.
- Thí nghiệm C báo **hai tỉ số cùng lúc**: so với mô hình học tại đích, và so với naive tại đích. Chỉ kết luận "dùng lại được" khi cả hai dưới 1.
- **Máy giả cũng báo hai tỉ số (làm rõ 27/09/2026).** Phân tích máy giả chạy bốn chiều giữa E1a và E1g, trong đó hai chiều là **tại chỗ** (E1a→E1a, E1g→E1g). Vì E1a và E1g không xuất hiện ở bất kỳ thí nghiệm nào khác, "naive tại đích" của chúng chỉ có thể đến từ chính hai chiều tại chỗ đó. Vì vậy hai chiều tại chỗ chạy thêm `naive`, `ma6` và `seasonal`; hai chiều chuyển giao **không** chạy mô hình cơ sở, vì dự đoán cơ sở không phụ thuộc mô hình nguồn nên sẽ trùng đúng chiều tại chỗ. Trước ngày 27/09/2026 `scripts/tn_c.py` chỉ chạy `ridge`, `xgb`, `xgb_mae` cho cả bốn chiều, nên cột `so_voi_naive_tai_dich` của máy giả sẽ toàn NaN; điều này được phát hiện và sửa **trước khi K7.3 chạy lần đầu**, nên không có kết quả nào phải bỏ đi.

**Chi phí bất đối xứng.** `chi phí = a · max(0, y − ŷ) + max(0, ŷ − y)` với `a ∈ {1, 2, 5, 10}`, chốt trước. Đây là **phân tích theo kịch bản giả định**: không suy ra chi phí tài chính, không suy ra mức giảm vi phạm cam kết chất lượng dịch vụ, và không có mô hình nào được huấn luyện bằng hàm mất mát bất đối xứng.

**Trường hợp lỗi.** Chuỗi có MASE không xác định vì mẫu số bằng 0, hoặc R² không xác định vì target hằng, bị loại khỏi phần gộp của **chỉ số đó** và được đếm riêng; không thay bằng 0 và không thay bằng vô hạn. Họ so sánh thiếu naive thì không kiểm định, và `scripts/kiem_ket_qua.py` báo ra.

**Tỉ số theo máy (làm rõ 25/09/2026).** Tỉ số `chỉ số mô hình / chỉ số naive` chỉ tính trên máy có chỉ số naive dương. Máy có chỉ số naive bằng 0 (ở E1: 6/708 máy có MAE naive bằng 0) bị loại khỏi trung vị tỉ số và được đếm ở cột `<chỉ số>_n_ti_so_loai` của `kiem_dinh.csv`. Bản tổng hợp trước ngày 25/09/2026 lấy trung vị cả tỉ số 0/0 nên `ti_so_p50` là NaN ở 12 dòng của `ma6` và `seasonal`, và bị lệch tối đa 0,055 ở 33 dòng khác vì có tỉ số vô hạn; khoảng tin cậy bootstrap vốn đã bỏ giá trị không hữu hạn nên không đổi. Wilcoxon, hiệu trung vị và tỉ lệ máy thắng tính trên hiệu, không đổi. Chưa có bảng tổng hợp nào từ bản cũ được lưu vào `results/tong_hop`.

**Dự đoán không hữu hạn (làm rõ 25/09/2026).** `seasonal` không có giá trị dự đoán ở những dòng thiếu `y(t+h-288)`; các dòng đó có `yhat` là NaN trong tệp dự đoán (E3: 217 đến 239 dòng mỗi tệp, khoảng 0,14%). Chúng nằm **ngoài** tập dòng đánh giá chung ở mục 8, nên không mô hình nào được chấm trên chúng. Phép kiểm `R04` vì vậy chỉ yêu cầu dự đoán hữu hạn **bên trong tập dòng chung**, và báo riêng số dòng không hữu hạn nằm ngoài tập đó. Đã kiểm: cả 18 họ của Thí nghiệm A chấm trên đúng cùng một tập dòng, và không còn dòng không hữu hạn nào trong tập chung (`ket_qua_tham_chieu/chan_doan/seasonal_tap_dong_chung.csv`).

**So sánh giá trị thật với dữ liệu gốc (làm rõ 25/09/2026).** Mô hình chuỗi lưu `y` ở `float32`, nên `y` trong dự đoán DLinear và GRU lệch tối đa 3,7·10⁻⁶ so với `data/processed`. Phép kiểm `R06` và `N04` dùng `np.allclose(rtol=1e-6, atol=1e-5)` thay cho ngưỡng tuyệt đối 1e-9. Sai lệch 3,7·10⁻⁶ được chấp nhận, sai lệch 1e-3 vẫn bị bắt (có kiểm thử).

**Đọc `cv_search.csv` (làm rõ 25/09/2026).** Tệp này bị ghi nối dài cộng dồn ở K4.1 (74.180 dòng, 1.170 khoá duy nhất). Bộ đọc khử trùng theo khoá `(env, mode, h, model, ung_vien, fold)` và báo lỗi nếu các bản trùng khác nhau vượt dung sai 1e-9 ở `mae_p50_val` hoặc khác nhau ở `n_train`, `n_val`. Các bản trùng thực tế chỉ khác 4,4·10⁻¹⁶.

---

## 10. Dữ liệu phát triển và giai đoạn xác nhận

| Mức | Nghĩa | Phần dữ liệu |
|---|---|---|
| Đã chấm mô hình | có mô hình huấn luyện hoặc chấm trên đó | E1, E2, E3 cửa sổ 8 ngày, cả ba tập |
| Đã xem thống kê mô tả | có tính thống kê, chưa mô hình nào chạm | E1 từ 20/08/2013 13:40 đến 11/09/2013 |
| Chưa đụng tới | chưa tính gì trên giá trị CPU | Rnd tháng 7 và tháng 9; Rnd tháng 8 sau 08/08/2013 22:00 |

Quy ước khoảng: mọi ranh giới là **nửa mở** `[bắt đầu, kết thúc)` tính theo bucket, và một dòng chỉ thuộc một giai đoạn khi cả `t` lẫn `t+h` nằm trong giai đoạn đó.

**Thí nghiệm B tách hai vai trò**: E2m08 là dữ liệu phát triển, chỉ dùng để đo chi phí và kiểm đường chạy; E1L, E2m07 và E2m09 là giai đoạn xác nhận, chạy sau khi tài liệu này khoá. Fold đầu tiên của E1L bắt đầu chấm ở bucket 4032, tức ngày thứ 14, nằm sau tập test cũ vốn kết thúc ở bucket 2304, nên hai phần không chồng nhau.

E3 không có giai đoạn xác nhận vì trace chỉ dài 8 ngày. Điều này phải nêu trong phần hạn chế.

---

## 11. Lưu vết

Mỗi lần chấm ghi dự đoán từng dòng, 14 cột: `nguon, dich, mode, lich, model, loss, h, fold, seed, series_id, bucket_phat, bucket_muc_tieu, y, yhat`. Kèm `meta.json` ghi cấu hình, phiên bản thư viện, đường dẫn Python và vân tay của mã, cấu hình và dữ liệu.

`bucket × 300` là số giây: với E1, E2 và các môi trường Bitbrains là epoch UTC thật; với E3 là giây kể từ đầu trace.


---

## Nhật ký sửa đổi sau ngày khoá

Mỗi dòng ghi ngày, điều đã đổi và lý do. **Không lần nào chạy lại thí nghiệm**, và không lần nào đổi một lựa chọn
thiết kế theo hướng có lợi cho kết quả.

| Ngày | Thay đổi | Lý do |
|---|---|---|
| 23/09/2026 | Bản khoá đầu tiên, chốt trước lượt chạy xác nhận của Thí nghiệm B. Giữ nguyên văn ở Phụ lục C. | — |
| 25/09/2026 | (1) Ghi nhận sai lệch D1: lượt chạy chính dùng lưới XGBoost ba ứng viên thay vì hai (Phụ lục A). (2) Làm rõ ba tiêu chí kiểm tra: dự đoán không hữu hạn của `seasonal`, dung sai `float32` khi đối chiếu giá trị thật, cách đọc `cv_search.csv`. (3) Tỉ số so với naive loại máy có mẫu số 0 và đếm riêng. Không đổi kết quả nào đã có. | D1 lộ ra khi đọc `chosen.csv`; ba chỗ làm rõ đến từ việc chạy thử các phép kiểm trên kết quả thật. |
| 27/09/2026 | (1) Ghi nhận sai lệch D2: bootstrap của Thí nghiệm B lấy mẫu lại theo máy chứ không theo khối ngày (Phụ lục B). (2) Làm rõ mẫu số MASE của Thí nghiệm B là 14 ngày đầu cố định. (3) Làm rõ fold cuối của Thí nghiệm B ngắn hơn 7 ngày. (4) Bộ tổng hợp và bộ kiểm kết quả nạp theo từng họ so sánh, vì nạp cả kho dự đoán làm hết bộ nhớ. (5) Máy giả chạy thêm mô hình cơ sở ở hai chiều tại chỗ để tính được tỉ số so với naive tại đích; sửa **trước khi** phần máy giả chạy lần đầu. (6) `R01` kiểm cả Thí nghiệm B và thêm `R01b` kiểm số fold; `R02` chỉ đòi naive ở họ chấm tại chỗ, thêm `R02b` kiểm tham chiếu của họ chuyển giao. (7) `chi_so_gop.csv` thêm `<chỉ số>_n_dung` và `<chỉ số>_n_loai` để đếm riêng máy bị loại khỏi từng chỉ số, đúng điều mục 9 đã đòi từ bản khoá; chỉ thêm cột, không đổi giá trị cột nào. | D2 lộ ra khi đọc mã để sửa lỗi hết bộ nhớ. Bằng chứng các bản sửa không đổi số: bảng của Thí nghiệm A tính lại sau khi sửa **trùng từng byte** với bảng trước khi sửa. |

Thực tế của điểm (7): MASE không loại máy nào; R² loại 306 máy trên 76.296, toàn bộ ở E1, đúng 6 máy mỗi họ —
chính là 6 máy có chuỗi hằng nên MAE của naive bằng 0.

---

## Phụ lục A. Sai lệch protocol D1: lưới siêu tham số của XGBoost

Phụ lục này được thêm ngày 25/09/2026. Nó mô tả điều đã xảy ra, không sửa lại điều đã chốt ngày 23/09/2026.

### A.1 Điều đã khoá và điều đã chạy

| | Đã khoá 23/09/2026 | Đã chạy ở K4.1 |
|---|---|---|
| Ứng viên của `xgb` | (4, 300), (8, 300) | (4, 300), (8, 300), **(8, 600)** |
| Ứng viên của `xgb_mae` | (4, 300), (8, 300) | (4, 300), (8, 300), **(8, 600)** |
| Số ứng viên | 2 | 3 |

Quy ước ghi: (`max_depth`, `n_estimators`).

### A.2 Nguyên nhân và thời điểm phát hiện

Lưới hai ứng viên là phương án được chọn ngày 23/09/2026 để giảm chi phí và được ghi ở mục 6 của bản khoá. `src/cwp/models/registry.py` **chưa bao giờ được sửa** theo phương án đó và vẫn khai báo ba ứng viên. Không có phép kiểm tự động nào so lưới đã chạy với lưới ghi trong protocol. (`HUONG_DAN_CHAY_V5.md` không nêu lưới; nó chỉ nêu ước lượng 13 giờ cho K4.1.)

K4.1 chạy từ 24/09/2026 khoảng 01:07 đến 25/09/2026 06:04 (28,9 giờ). Sai lệch được phát hiện ngày 25/09/2026, **sau khi K4.1 hoàn thành**, khi kiểm tra chỉ đọc thấy `chosen.csv` chứa `{"max_depth": 8, "n_estimators": 600}`. Ước lượng 13 giờ cho K4.1 thấp hơn thực tế khoảng 2,2 lần. Theo cột `giay` của `cv_search.csv` (đã khử trùng), ứng viên (8, 600) tốn 13,9 giờ trong 24,2 giờ dò siêu tham số của hai biến thể XGBoost; bỏ ứng viên này thì K4.1 còn khoảng 15 giờ. Đây là ước tính từ thời gian dò, không phải một lần chạy lại.

### A.3 Điều đã kiểm chứng

1. **Cấu hình 600 cây được chọn ở 7/36 trường hợp**: `xgb` 2/18 và `xgb_mae` 5/18.
   - `xgb`: E1 N0 h=1, E1 N0 h=6.
   - `xgb_mae`: E1 N0 h=1, E1 N1 h=1, E2 N0 h=1, E2 N1 h=1, E2 N1 h=6.
   - Không có trường hợp nào ở E3.
2. **Hai biến thể dùng cùng lưới.** Tập ứng viên duy nhất của `xgb` và `xgb_mae` trong `cv_search.csv` bằng nhau. Phép so hai hàm mất mát vẫn khác nhau đúng một tham số (`objective`) trên cùng một lưới.
3. **Test set không tham gia chọn cấu hình.**
   - Chọn siêu tham số dùng rolling-origin 5 fold bên trong vùng validation `[1612, 1957)`. Fold cuối kết thúc đúng tại bucket 1957, là đầu của test `[1957, 2304)`.
   - Bucket mục tiêu lớn nhất trong mọi fold là 1956, nhỏ hơn 1957, ở cả 9 tổ hợp `(môi trường, tầm dự báo)`.
   - `n_train` và `n_val` của từng fold trong `cv_search.csv` khớp số dòng tính độc lập từ `data/features` ở cả 9 tổ hợp.
   - Test được chấm đúng một lần cho mỗi tổ hợp: 72 lượt chấm, khớp 72 tổ hợp `(môi trường, chế độ, tầm, mô hình học máy)`; `meta.json` ghi `so_lan_cham_test = 72`.
4. **Mức ảnh hưởng, đo chỉ trên validation** (`ket_qua_tham_chieu/chan_doan/xgb_600_cay_tren_validation.csv`). Ở 7 tổ hợp chọn 600 cây, MAE trung vị theo chuỗi trên validation (trung bình 5 fold) thấp hơn ứng viên tốt nhất trong lưới đã khoá từ 0,29% đến 4,21%, trung vị 1,43%. Ở 29 tổ hợp còn lại, ứng viên 600 cây không tốt hơn. Đây là số liệu validation, không phải test.

### A.4 Hệ quả

- Lưới đã chạy là **tập cha** của lưới đã khoá: mọi cấu hình đã khoá đều nằm trong lưới đã chạy.
- Ở 7 tổ hợp nói trên, kết quả test của XGBoost dùng một cấu hình **nằm ngoài** lưới đã khoá.
- Mọi thí nghiệm đọc `chosen.csv` (Thí nghiệm B, Thí nghiệm C, máy giả, K = 0, độ nhạy) kế thừa đúng cấu hình đã chọn ở K4.1, kể cả ứng viên 600 cây. Chi phí của các thí nghiệm đó có thể cao hơn ước lượng một chút ở những tổ hợp dùng 600 cây.
- Kết quả không đổi bất kỳ chỉ số nào ở K4.1 hay K4.2.

### A.5 Quyết định (25/09/2026)

Chọn phương án **giữ nguyên kết quả K4.1 và K4.2, không chạy lại**, ghi nhận sai lệch trong protocol. Phương án còn lại, chạy lại K4.1 với lưới hai ứng viên, không được chọn.

Không làm: không chọn lại cấu hình theo kết quả test, không loại bỏ hay thay thế kết quả của 7 tổ hợp dùng 600 cây, không sửa `chosen.csv`.

### A.6 Yêu cầu khi viết bài

- Mô tả lưới đã chạy: ba ứng viên cho cả hai biến thể XGBoost.
- Nêu rõ trong phần phương pháp hoặc hạn chế rằng lưới hai ứng viên đã được lên kế hoạch nhưng lưới ba ứng viên được chạy, kèm số liệu ở A.3.
- Không viết như thể lưới ba ứng viên đã được chốt từ đầu.

### A.7 Bằng chứng toàn vẹn

Checksum SHA-256 của mọi tệp kết quả K4.1 và K4.2 (`ket_qua_tham_chieu/tn_a`, `ket_qua_tham_chieu/dl`), cùng `data/processed`, `data/features`, `data/eval_rows`, được lập **trước** khi sửa mã và lưu ở `kho lưu trữ nội bộ của nhóm`. Đối chiếu sau khi sửa nằm ở phần báo cáo kèm theo.

---

## Phụ lục B. Sai lệch protocol D2: bootstrap của Thí nghiệm B

Phụ lục này được thêm ngày 27/09/2026. Nó mô tả điều đã xảy ra, không sửa lại điều đã chốt ngày 23/09/2026.

### B.1 Điều đã khoá và điều đã chạy

| | Đã khoá 23/09/2026 | Đã chạy |
|---|---|---|
| Bootstrap của Thí nghiệm A | lấy mẫu lại theo máy | lấy mẫu lại theo máy |
| Bootstrap của Thí nghiệm B | lấy mẫu lại **theo khối ngày** | lấy mẫu lại **theo máy** |

### B.2 Nguyên nhân và thời điểm phát hiện

`scripts/tong_hop.py` có sẵn cơ chế lấy mẫu lại theo khối: hàm `boot_ti_so(ti_so, nhom, rng)` nhận
tham số `nhom` và khi `nhom` khác `None` thì lấy mẫu lại theo khối. Hàm `mot_ho` cũng nhận tham số
`theo_ngay` và bộ gọi truyền đúng `theo_ngay = (thi_nghiem == "tn_b")`. Nhưng **thân hàm `mot_ho` không
dùng `theo_ngay`**, và `boot_ti_so` luôn được gọi với `nhom = None`. Nghĩa là nhánh theo khối ngày chưa
bao giờ chạy. Không có phép kiểm nào so cách lấy mẫu lại đã chạy với cách ghi trong protocol.

Sai lệch được phát hiện ngày 27/09/2026, **sau khi K6.3 đã hoàn thành** (8,7 giờ), khi đọc
`tong_hop.py` để sửa lỗi hết bộ nhớ.

### B.3 Phạm vi ảnh hưởng

Chỉ ảnh hưởng **khoảng tin cậy bootstrap 95% của tỉ số trung vị** (`mae_ti_so_ci_lo`, `mae_ti_so_ci_hi`)
của Thí nghiệm B. Không ảnh hưởng:

- trung vị theo máy của MAE, RMSE, MASE, và tỉ số trung vị,
- kiểm định Wilcoxon ghép cặp và hiệu chỉnh Holm,
- tỉ lệ máy thắng, chi phí bất đối xứng,
- bất kỳ số nào của Thí nghiệm A.

### B.4 Quyết định (27/09/2026)

Chọn phương án **giữ cách lấy mẫu lại theo máy, ghi nhận sai lệch**, không tính lại. Phương án thực hiện
lấy mẫu lại theo khối ngày không được chọn vì số khối quá ít để có ý nghĩa: đoạn chấm của fold 0 và fold 1
chỉ dài 7 ngày, còn fold 2 chỉ 2 ngày (E1L, E2m07) hoặc 1 ngày (E2m09) — xem mục 4. Với 1 khối thì lấy mẫu
lại theo khối cho khoảng tin cậy rộng bằng 0, tức một con số không có nghĩa.

### B.5 Yêu cầu khi viết bài

- Nói rõ khoảng tin cậy của Thí nghiệm B lấy mẫu lại **theo máy**, không theo khối ngày.
- Nêu trong phần hạn chế rằng khoảng tin cậy đó **không tính đến phụ thuộc giữa các ngày**, nên có thể
  hẹp hơn thực tế.
- Không viết như thể lấy mẫu lại theo khối ngày đã được thực hiện.

### B.6 Bản sửa kèm theo, không liên quan đến số liệu

Cùng ngày 27/09/2026, `scripts/tong_hop.py` và `scripts/kiem_ket_qua.py` được sửa để nạp dự đoán
**theo từng họ so sánh** (`src/cwp/quy_trinh/nap.py`), thay cho việc nạp cả kho vào một `DataFrame`. Lý do: Thí
nghiệm B có khoảng 162 triệu dòng dự đoán, nạp hết cần trên 10 GB nên cả hai script đều chết vì hết bộ
nhớ trên máy 15,7 GB. Bản sửa giữ nguyên thứ tự duyệt họ, vì bootstrap dùng chung một bộ sinh số ngẫu
nhiên cho mọi họ. Bằng chứng không đổi số: chạy lại Thí nghiệm A sau khi sửa cho ra cả năm bảng
**trùng từng byte** với bảng trước khi sửa.

Phép kiểm `R08` từ nay so số dòng trong tập chung theo họ **gồm cả `fold`**, đúng định nghĩa họ so sánh ở
mục 8; với Thí nghiệm A `fold` luôn là `test` nên không đổi gì. Phép kiểm `R06` lấy mẫu theo từng họ thay
vì lấy mẫu trên toàn kho, vì lấy mẫu trên toàn kho cần nạp cả kho.

---

## Phụ lục C. Bản khoá ngày 23/09/2026, nguyên văn

Phần dưới đây là bản khoá ngày 23/09/2026 chép lại **không sửa một chữ nào của nội dung**. Thay đổi duy nhất là mỗi tiêu đề bị hạ một cấp để lồng vừa vào tài liệu
này. Nó ở đây để một người đọc có thể tự kiểm: mọi khác biệt giữa phần dưới và các mục 1
đến 11 ở trên đều phải có một dòng tương ứng trong nhật ký sửa đổi hoặc trong Phụ lục A, B.

Hai chỗ khác biệt **không** thuộc về thiết kế mà chỉ là đổi tên trong kho công bố, ghi ở
đây cho đủ: đường dẫn `results/v5/tong_hop/` ở mục 8 nay là `results/tong_hop/`, và
`scripts/v5_kiem_ket_qua.py` ở mục 9 nay là `scripts/kiem_ket_qua.py`. Đoạn ở mục 11 nói về
nơi lưu kết quả của lượt chạy trước không được chép sang phần chính, vì những thư mục đó
không nằm trong kho này.

Ngày khoá: 23/09/2026. Tài liệu này khoá mọi lựa chọn thiết kế **trước** lượt chạy xác nhận. Mỗi lần sửa phải tạo phiên bản mới ở cuối tài liệu, ghi ngày, phần đã đổi và lý do; không sửa đè lên phiên bản cũ.

---

### 1. Câu hỏi và phạm vi

| Mã | Câu hỏi | Thí nghiệm |
|---|---|---|
| RQ1 | Kết luận "mô hình có vượt dự báo naive không" thay đổi bao nhiêu khi đổi chuẩn hoá, hàm mất mát và tiêu chí đánh giá, ở ba tầm dự báo | A |
| RQ2 | Kết luận của RQ1 có giữ nguyên qua các giai đoạn thời gian khác nhau không | B |
| RQ3 | Mô hình học ở một tập vết dùng lại được ở tập vết khác đến mức nào | C |

Phân tích bổ sung: máy giả, đối chứng không nội suy, độ nhạy ngưỡng lọc quần thể, Random Forest phạm vi hẹp, chi phí bất đối xứng, phân nhóm máy ở mức thăm dò.

---

### 2. Dữ liệu và quần thể

| Môi trường | Nguồn | Cửa sổ | Vai trò |
|---|---|---|---|
| E1 | Bitbrains fastStorage | 8 ngày từ 12/08/2013 13:40 | Thí nghiệm A và C |
| E2 | Bitbrains Rnd tháng 8 | 8 ngày từ 31/07/2013 22:00 | Thí nghiệm A và C |
| E3 | Alibaba 2018 | 8 ngày từ đầu trace | Thí nghiệm A và C |
| E1L | Bitbrains fastStorage | 30 ngày | Thí nghiệm B, xác nhận |
| E2m08 | Bitbrains Rnd tháng 8 | 30 ngày | Thí nghiệm B, **phát triển** |
| E2m07 | Bitbrains Rnd tháng 7 | 30 ngày | Thí nghiệm B, xác nhận |
| E2m09 | Bitbrains Rnd tháng 9 | 29 ngày | Thí nghiệm B, xác nhận |

Ba tháng của Rnd **không nối lại**. Bằng chứng: trên mẫu 60 máy chỉ 31 máy giữ nguyên cả số nhân, xung nhịp và dung lượng RAM qua ba tháng, và giữa hai tháng liên tiếp có một ngày trống. Kết luận ghi đúng mức: **chưa đủ bằng chứng để nối an toàn**; thay đổi cấu hình phần cứng không tự chứng minh đó là máy khác.

**Quần thể chính.** Một chuỗi được giữ khi phần lịch sử dùng để quyết định thoả cả bốn điều: có ít nhất một điểm, CPU trung bình từ 1% trở lên, có ít nhất 3 giá trị phân biệt, và có ít nhất 500 dòng huấn luyện hợp lệ ở tầm 12. Phần lịch sử đó là `[0, 1612)` với cửa sổ 8 ngày, và `[0, 4032)` tức 14 ngày đầu với các môi trường dài ngày. Phạm vi phát biểu vì vậy là **máy đã hoạt động trong lịch sử**.

Quần thể sau khi lọc: E1 708 chuỗi, E2 302, E3 487.

**Độ nhạy ngưỡng.** Ba ngưỡng CPU trung bình phần train chốt trước: 0%, 0,5% và 2%, trong đó 0% nghĩa là bỏ tiêu chí mức tải nhưng giữ ba tiêu chí chất lượng còn lại. Ngưỡng **không** được chọn theo kết quả.

---

### 3. Tiền xử lý

1. Cắt CPU về `[0, 100]`.
2. Đưa về lưới 5 phút bằng trung bình các mẫu trong mỗi ô.
3. Cắt cửa sổ chung của môi trường.
4. Nội suy tuyến tính các cụm thiếu dài tối đa 2 điểm; cụm dài hơn giữ nguyên là thiếu.
5. Lọc chuỗi theo mục 2.

**Luật nhân quả với giá trị nội suy.** Nội suy dùng đầu mút bên phải, nên một điểm nội suy chỉ tính được khi đầu mút đó đã tới. Hệ quả áp trong luật dòng hợp lệ: dòng tại `t` bị loại nếu `y(t)` là giá trị nội suy hoặc `y(t+h)` là giá trị nội suy. Điểm nội suy nằm hẳn trước `t` vẫn dùng được, vì khi `y(t)` là quan sát thật thì đầu mút phải của chúng không vượt quá `t`. Bản đối chứng `K = 0` không nội suy gì và dùng để đo phần còn lại.

---

### 4. Đặc trưng và chia dữ liệu

20 đặc trưng, tất cả chỉ lấy từ quá khứ của chính máy đó: `lag_0` đến `lag_24` (7 giá trị trễ, trong đó `lag_0` là `y(t)`), 8 thống kê cửa sổ 6 và 12 điểm, sai phân một bước, 4 đặc trưng lịch.

Dòng tại `t` hợp lệ khi mọi điểm trong `[t-24, t]` không thiếu, `y(t+h)` không thiếu, và luật nhân quả ở mục 3 thoả.

**Khung 8 ngày**: train `[0, 1612)`, validation `[1612, 1957)`, test `[1957, 2304)`. Một dòng thuộc một tập khi cả `t` và `t+h` nằm trong tập đó; dòng vắt qua ranh giới bị loại.

**Khung dài ngày cho Thí nghiệm B**: lịch sử tối thiểu 14 ngày, huấn luyện lại mỗi 7 ngày. Fold `i` khớp trên `[0, o_i)` và chấm trên `[o_i, o_i + 2016)`, với `o_i = 4032 + 2016·i`. Các đoạn chấm rời nhau và phủ kín phần còn lại của cửa sổ.

---

### 5. Chuẩn hoá

| Chế độ | Biến đổi | Thống kê |
|---|---|---|
| N0 | giữ nguyên CPU% | không |
| N1 | z-score theo từng máy | `mu`, `sd` của **phần train của chính máy đó**, ranh giới theo khung đang dùng |

N1 dựng bằng phép affine trên ma trận N0, đúng bằng cách sinh lại đặc trưng từ chuỗi đã z-score; hai đường được đối chiếu ở phép kiểm K13. Máy có `sd` không dương bị loại khỏi cả huấn luyện lẫn chấm ở N1, và số máy bị loại được ghi ra.

Chế độ N2, tức sai phân, nằm ở phần bổ sung, không thuộc lưới chính.

Mọi dự đoán đưa về thang CPU% gốc trước khi tính bất kỳ chỉ số nào.

---

### 6. Mô hình và lưới siêu tham số

| Mô hình | Hàm mất mát | Lưới | Ghi chú |
|---|---|---|---|
| naive | không | — | `ŷ = y(t)` |
| ma6 | không | — | trung bình `y[t-5 .. t]` |
| seasonal | không | — | `y(t+h-288)` |
| seasonal_tuan | không | — | `y(t+h-2016)`, chỉ Thí nghiệm B |
| ridge | bình phương kèm phạt L2 | `alpha` ∈ {0,01; 0,1; 1; 10; 100} | giữ hàm mất mát riêng, không thuộc lưới hai hàm mất mát |
| xgb | bình phương | `max_depth` ∈ {4, 8}, 300 cây | |
| xgb_mae | tuyệt đối | `max_depth` ∈ {4, 8}, 300 cây | khác `xgb` **đúng một tham số** |
| svr | ε-insensitive | `C` ∈ {1, 10}, mẫu con 10.000 dòng | giữ hàm mất mát riêng |
| dlinear | tuyệt đối | một cấu hình cố định | cửa sổ 288, 3 hạt giống |
| gru | tuyệt đối | một cấu hình cố định | chỉ trên Colab, 4 tổ hợp |
| rf | bình phương | một cấu hình định trước | 50 cây, `max_depth` 16, chỉ để nối với bảng cũ |

Lưới hai hàm mất mát **chỉ gồm cặp `xgb` và `xgb_mae`**. Ridge và SVR giữ hàm mất mát riêng của chúng và không được đọc như hai ô tương đương của cùng lưới đó.

Lưới XGBoost đã bị thu gọn còn hai ứng viên vì chi phí: hàm mất mát tuyệt đối đắt hơn hàm bình phương khoảng 7,6 lần. Điều này phải ghi trong bài.

**Hạt giống.** `random_state = 42` cho mọi mô hình bảng, một hạt giống. DLinear chạy ba hạt giống 42, 43, 44 và báo kèm độ phân tán. GRU chạy một hạt giống 42.

**Chuẩn hoá của mô hình chuỗi.** DLinear và GRU dùng **N1**, tức thống kê phần train của từng máy, giống hệt mô hình bảng. Không dùng chuẩn hoá theo từng cửa sổ. Dự đoán đưa về CPU% bằng `ŷ = ẑ·sd + mu`.

---

### 7. Chọn siêu tham số

- Rolling-origin 5 fold **bên trong vùng validation** của khung 8 ngày, expanding.
- Điểm chọn: **trung vị MAE theo chuỗi, trên thang CPU%**.
- Mô hình cuối khớp trên train + validation, rồi chấm test **đúng một lần**.
- Thí nghiệm B và C **không dò lại**: lấy siêu tham số của Thí nghiệm A theo môi trường nguồn. E1L lấy của E1; E2m07, E2m08, E2m09 lấy của E2; máy giả E1a, E1g lấy của E1.
- DLinear chọn số epoch bằng dừng sớm trên validation, rồi khớp lại trên train + validation bằng đúng số epoch đó.

---

### 8. Tập dòng đánh giá chung

Mọi mô hình trong một họ so sánh chấm trên **đúng cùng một tập dòng**. Họ là các mô hình cùng `(thí nghiệm, nguồn, đích, chế độ, tầm, fold)`. Điều kiện lấy giao:

| Điều kiện | Áp khi |
|---|---|
| `bang` — dòng hợp lệ của ma trận đặc trưng | luôn luôn |
| `ngay` — có `y(t+h-288)` là quan sát thật | họ có seasonal theo ngày |
| `tuan` — có `y(t+h-2016)` là quan sát thật | họ có seasonal theo tuần |
| `chuoi` — 288 điểm `[t-287, t]` không thiếu | họ có DLinear hoặc GRU |

Số dòng bị loại theo từng điều kiện báo ở `results/tong_hop/tom_tat_dong.csv` và phải nêu trong bài.

---

### 9. Chỉ số và kiểm định

- **Chỉ số chính: MAE**, tính riêng từng máy rồi lấy trung vị trên tập máy. Không đổi chỉ số chính theo kết quả.
- Báo kèm **RMSE** và **MASE**; MASE dùng mẫu số là sai số naive một bước trên phần train của chính máy đó.
- Báo kèm **độ lớn cải thiện**: tỉ số trung vị theo máy, tỉ lệ máy thắng, và khoảng tin cậy bootstrap 95% của tỉ số trung vị.
- **Kiểm định**: Wilcoxon ghép cặp theo từng máy, mỗi mô hình so với naive.
- **Họ hiệu chỉnh Holm**: các phép so mô hình với naive **trong cùng một họ** ở mục 8. Mức ý nghĩa 0,05.
- Hướng lấy từ **trung vị hiệu ghép cặp**, không phải hiệu hai trung vị.
- Bootstrap lấy mẫu lại **theo máy**; với Thí nghiệm B lấy mẫu lại **theo khối ngày**.
- Thí nghiệm C báo **hai tỉ số cùng lúc**: so với mô hình học tại đích, và so với naive tại đích. Chỉ kết luận "dùng lại được" khi cả hai dưới 1.

**Chi phí bất đối xứng.** `chi phí = a · max(0, y − ŷ) + max(0, ŷ − y)` với `a ∈ {1, 2, 5, 10}`, chốt trước. Đây là **phân tích theo kịch bản giả định**: không suy ra chi phí tài chính, không suy ra mức giảm vi phạm cam kết chất lượng dịch vụ, và không có mô hình nào được huấn luyện bằng hàm mất mát bất đối xứng.

**Trường hợp lỗi.** Chuỗi có MASE không xác định vì mẫu số bằng 0, hoặc R² không xác định vì target hằng, bị loại khỏi phần gộp của **chỉ số đó** và được đếm riêng; không thay bằng 0 và không thay bằng vô hạn. Họ so sánh thiếu naive thì không kiểm định, và `scripts/v5_kiem_ket_qua.py` báo ra.

---

### 10. Dữ liệu phát triển và giai đoạn xác nhận

| Mức | Nghĩa | Phần dữ liệu |
|---|---|---|
| Đã chấm mô hình | có mô hình huấn luyện hoặc chấm trên đó | E1, E2, E3 cửa sổ 8 ngày, cả ba tập |
| Đã xem thống kê mô tả | có tính thống kê, chưa mô hình nào chạm | E1 từ 20/08/2013 13:40 đến 11/09/2013 |
| Chưa đụng tới | chưa tính gì trên giá trị CPU | Rnd tháng 7 và tháng 9; Rnd tháng 8 sau 08/08/2013 22:00 |

Quy ước khoảng: mọi ranh giới là **nửa mở** `[bắt đầu, kết thúc)` tính theo bucket, và một dòng chỉ thuộc một giai đoạn khi cả `t` lẫn `t+h` nằm trong giai đoạn đó.

**Thí nghiệm B tách hai vai trò**: E2m08 là dữ liệu phát triển, chỉ dùng để đo chi phí và kiểm đường chạy; E1L, E2m07 và E2m09 là giai đoạn xác nhận, chạy sau khi tài liệu này khoá. Fold đầu tiên của E1L bắt đầu chấm ở bucket 4032, tức ngày thứ 14, nằm sau tập test cũ vốn kết thúc ở bucket 2304, nên hai phần không chồng nhau.

E3 không có giai đoạn xác nhận vì trace chỉ dài 8 ngày. Điều này phải nêu trong phần hạn chế.

---

### 11. Lưu vết

Mỗi lần chấm ghi dự đoán từng dòng, 14 cột: `nguon, dich, mode, lich, model, loss, h, fold, seed, series_id, bucket_phat, bucket_muc_tieu, y, yhat`. Kèm `meta.json` ghi cấu hình, phiên bản thư viện, đường dẫn Python và vân tay của mã, cấu hình và dữ liệu.

`bucket × 300` là số giây: với E1, E2 và các môi trường Bitbrains là epoch UTC thật; với E3 là giây kể từ đầu trace.


---

### Lịch sử phiên bản

| Phiên bản | Ngày | Thay đổi | Lý do |
|---|---|---|---|
| 1.0 | 23/09/2026 | Bản đầu, khoá trước lượt xác nhận của Thí nghiệm B | — |
