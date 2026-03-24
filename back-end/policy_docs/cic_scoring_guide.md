# Hướng dẫn Chấm điểm Tín dụng CIC — Trung tâm Thông tin Tín dụng Quốc gia Việt Nam

## 1. Tổng quan về CIC

Trung tâm Thông tin Tín dụng Quốc gia Việt Nam (CIC) trực thuộc Ngân hàng Nhà nước Việt Nam (NHNN), là cơ quan duy nhất thực hiện việc thu thập, lưu trữ, phân tích và cung cấp thông tin tín dụng của cá nhân và tổ chức cho các tổ chức tín dụng (TCTD).

CIC phục vụ hai mục đích chính:
- Hỗ trợ TCTD đánh giá uy tín tín dụng và quản lý rủi ro cho vay.
- Cung cấp thông tin cho khách hàng tự tra cứu điểm tín dụng cá nhân (báo cáo K40).

## 2. Thang điểm tín dụng CIC (300–850)

CIC sử dụng thang điểm từ 300 đến 850 để đánh giá mức độ uy tín tín dụng. Điểm càng cao thì độ tin cậy tín dụng càng lớn.

### 2.1 Phân loại theo thang điểm

| Khoảng điểm | Mức đánh giá | Ý nghĩa đối với TCTD |
|---|---|---|
| 750–850 | Xuất sắc | Rủi ro rất thấp. Đủ điều kiện phê duyệt tự động với lãi suất ưu đãi. Hạn mức cao nhất. |
| 700–749 | Tốt | Rủi ro thấp. Dễ dàng được duyệt vay với điều kiện tốt. TCTD nên ưu tiên cross-sell. |
| 650–699 | Khá | Rủi ro trung bình-thấp. Được duyệt vay nhưng có thể cần thẩm định bổ sung. Lãi suất thông thường. |
| 550–649 | Trung bình | Rủi ro trung bình. Cần thẩm định kỹ lưỡng. Có thể cần TSBĐ hoặc bảo lãnh. Lãi suất cao hơn mặt bằng. |
| 450–549 | Dưới trung bình | Rủi ro cao. Yêu cầu TSBĐ bắt buộc, hạn mức thấp, kỳ hạn ngắn. Cần phê duyệt cấp cao. |
| 300–449 | Rủi ro cao | Rủi ro rất cao. Thường bị từ chối trừ trường hợp đặc biệt có TSBĐ giá trị cao hoặc bảo lãnh bên thứ ba. |

### 2.2 Mối liên hệ giữa điểm CIC và PD (Probability of Default)

| Khoảng điểm CIC | PD ước tính | Nhóm nợ dự kiến |
|---|---|---|
| 750–850 | < 2% | Nhóm 1 |
| 700–749 | 2–5% | Nhóm 1 |
| 650–699 | 5–10% | Nhóm 1–2 |
| 550–649 | 10–20% | Nhóm 2 |
| 450–549 | 20–35% | Nhóm 2–3 |
| 300–449 | > 35% | Nhóm 3–5 |

## 3. Năm nhóm nợ CIC

Theo Thông tư 11/2021/TT-NHNN, các khoản vay được phân loại thành 5 nhóm nợ.

### 3.1 Chi tiết phân loại

| Nhóm nợ | Tên gọi | Số ngày quá hạn (DPD) | Tỷ lệ trích lập DPRR | Mô tả |
|---|---|---|---|---|
| Nhóm 1 | Nợ đủ tiêu chuẩn | 0–9 ngày | 0% | Khoản nợ chưa quá hạn hoặc quá hạn dưới 10 ngày. Khách hàng có khả năng thu hồi đầy đủ gốc và lãi đúng hạn. |
| Nhóm 2 | Nợ cần chú ý | 10–90 ngày | 5% | Khoản nợ quá hạn từ 10 đến 90 ngày hoặc điều chỉnh kỳ hạn trả nợ lần đầu. Cần giám sát tăng cường. |
| Nhóm 3 | Nợ dưới tiêu chuẩn | 91–180 ngày | 20% | Bắt đầu rơi vào nợ xấu. Khoản nợ quá hạn 91–180 ngày hoặc gia hạn nợ lần đầu. Khả năng tổn thất một phần. |
| Nhóm 4 | Nợ nghi ngờ | 181–360 ngày | 50% | Khả năng mất vốn cao. Khoản nợ quá hạn 181–360 ngày hoặc điều chỉnh kỳ hạn lần thứ hai. |
| Nhóm 5 | Nợ có khả năng mất vốn | > 360 ngày | 100% | Mức nợ xấu nghiêm trọng nhất. Khách hàng mất khả năng trả nợ hoặc không hợp tác. Khả năng thu hồi gần như bằng không. |

### 3.2 Quy tắc phân loại quan trọng

- **Nguyên tắc nhóm nợ cao nhất**: Toàn bộ dư nợ của một khách hàng tại một TCTD phải được phân loại vào cùng một nhóm nợ — nhóm có mức độ rủi ro cao nhất.
- **Đồng bộ qua CIC**: Nếu khách hàng bị phân loại nhóm nợ cao tại TCTD A, các TCTD khác cũng phải điều chỉnh theo danh sách do CIC cung cấp.
- **Hard override**: Khách hàng thuộc Nhóm 4–5 tại bất kỳ TCTD nào → từ chối cho vay mới bất kể ML score.

## 4. Các yếu tố ảnh hưởng đến điểm CIC

### 4.1 Trọng số các yếu tố

| Yếu tố | Trọng số | Mô tả chi tiết |
|---|---|---|
| Lịch sử thanh toán | 35% | Thanh toán đúng hạn các khoản vay hiện tại và quá khứ. Một lần trả chậm >30 ngày có thể giảm 30–60 điểm. |
| Số tiền tín dụng đang nợ | 30% | Tổng dư nợ và tỷ lệ sử dụng hạn mức tín dụng (credit utilization). Tỷ lệ >50% hạn mức → giảm điểm. |
| Thời gian lịch sử tín dụng | 15% | Lịch sử tín dụng càng dài và ổn định càng được đánh giá cao. Tối thiểu 6 tháng để có điểm đáng tin cậy. |
| Sự đa dạng loại tín dụng | 10% | Kết hợp nhiều loại hình vay (thẻ tín dụng, vay mua nhà, vay tiêu dùng) → điểm cao hơn. |
| Tài khoản tín dụng mới | 10% | Mở quá nhiều khoản vay mới trong thời gian ngắn → tín hiệu rủi ro. Mỗi hard inquiry giảm 5–10 điểm. |

### 4.2 Các yếu tố giảm điểm CIC nặng

| Sự kiện | Mức giảm ước tính | Thời gian ảnh hưởng |
|---|---|---|
| Trả chậm 30–60 ngày | −30 đến −60 điểm | 2 năm |
| Trả chậm 90+ ngày (nợ xấu nhóm 3+) | −80 đến −150 điểm | 5 năm |
| Xóa nợ / write-off | −150 đến −200 điểm | 5 năm |
| Phá sản cá nhân | −200+ điểm | 7–10 năm |
| Nhiều hard inquiry (>3 trong 6 tháng) | −15 đến −30 điểm | 1 năm |

## 5. Xử lý Thin-file (Khách hàng không có lịch sử CIC)

### 5.1 Định nghĩa thin-file

Khách hàng thin-file là khách hàng không có hoặc có rất ít lịch sử tín dụng tại CIC, bao gồm:
- Chưa từng vay tại TCTD nào
- Lịch sử tín dụng < 6 tháng
- Chỉ có 1 khoản vay nhỏ đã tất toán

### 5.2 Quy trình đánh giá thin-file

Theo quy định, TCTD không được từ chối tự động khách hàng thin-file. Thay vào đó:

1. **Alternative data scoring**: Sử dụng dữ liệu thay thế — sao kê ngân hàng, giao dịch thẻ, dữ liệu viễn thông, hóa đơn tiện ích.
2. **Enhanced verification**: Yêu cầu bổ sung hồ sơ — hợp đồng lao động, xác nhận lương, giấy phép kinh doanh (SME).
3. **Conservative limits**: Hạn mức cho vay thấp hơn, kỳ hạn ngắn hơn, yêu cầu TSBĐ hoặc bảo lãnh.
4. **Flagging**: Đánh dấu `thin_file_flag = True` trong hệ thống. Báo cáo phải nêu rõ rằng đánh giá dựa trên dữ liệu thay thế.

### 5.3 Quy định liên quan

- Thông tư 39/2016 Điều 7: TCTD phải có quy trình đánh giá cho cả khách hàng có và không có lịch sử CIC.
- Nghiên cứu Cash Flow Underwriting (Ng et al., 2025): Phương pháp đánh giá tín dụng dựa trên dòng tiền giao dịch ngân hàng — áp dụng cho khách hàng underbanked.

## 6. Lưu trữ và quyền truy cập

- Thông tin nợ xấu (nhóm 3–5) được CIC lưu trữ tối đa **5 năm** kể từ ngày kết thúc (tất toán hoặc xử lý xong).
- Khách hàng có quyền yêu cầu tra cứu điểm CIC cá nhân qua **báo cáo K40**.
- TCTD phải thực hiện phân loại nợ và trích lập dự phòng **hàng tháng** (7 ngày đầu mỗi tháng), gửi kết quả về CIC.
- CIC có **3 ngày** để tổng hợp và phản hồi danh sách nhóm nợ cho các TCTD.

## 7. Ứng dụng trong CreditLens

### 7.1 Mapping CIC vào hệ thống scoring

| CIC Data Point | CreditLens Feature | Module |
|---|---|---|
| Điểm CIC 300–850 | `cic_score_proxy` (EXT_SOURCE) | A1 |
| Nhóm nợ 1–5 | `debt_group` | A1 |
| Số khoản vay active | `num_active_loans` | A1 |
| Worst-ever nhóm nợ | `worst_ever_group` | A1 |
| Không có CIC | `thin_file_flag = True` | A1 |

### 7.2 Hard override rules liên quan CIC

| Điều kiện | Hành động | Căn cứ |
|---|---|---|
| CIC Nhóm 4–5 | REJECT bất kể ML score | QĐ493/2005, TT11/2021 |
| CIC Nhóm 3 + DTI > 50% | REJECT | TT39/2016 Điều 17 |
| Thin-file + Score < 560 | Tăng yêu cầu TSBĐ | Chính sách nội bộ |
| CIC ≥ 700 + DTI < 30% | Ưu tiên AUTO APPROVE | Chính sách nội bộ |
