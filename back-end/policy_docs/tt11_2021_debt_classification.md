# Thông tư 11/2021/TT-NHNN — Phân loại Tài sản Có, Trích lập và Sử dụng Dự phòng Rủi ro

## 1. Tổng quan

**Thông tư 11/2021/TT-NHNN** ban hành ngày 30/07/2021, có hiệu lực từ 01/10/2021, quy định chi tiết về:
- Phân loại tài sản có (nợ)
- Mức trích và phương pháp trích lập dự phòng rủi ro
- Sử dụng dự phòng để xử lý rủi ro

Thông tư này **thay thế** Thông tư 02/2013/TT-NHNN và Thông tư 09/2014/TT-NHNN, đồng thời kế thừa và phát triển từ Quyết định 493/2005/QĐ-NHNN (đã hết hiệu lực).

## 2. Phạm vi áp dụng

### 2.1 Đối tượng
- Tổ chức tín dụng (TCTD) thành lập và hoạt động theo Luật các TCTD
- Chi nhánh ngân hàng nước ngoài
- **Không áp dụng** cho TCTD đang được kiểm soát đặc biệt

### 2.2 Phạm vi tài sản có
- Các khoản cho vay
- Mua nợ
- Cho thuê tài chính
- Chiết khấu giấy tờ có giá
- Bao thanh toán
- Cam kết ngoại bảng (L/C, bảo lãnh)
- Trái phiếu doanh nghiệp
- Tiền gửi có kỳ hạn tại TCTD khác

## 3. Phân loại nợ — Phương pháp định lượng (Điều 10)

### 3.1 Năm nhóm nợ theo DPD (Days Past Due)

| Nhóm | Tên gọi | DPD | Điều kiện bổ sung | Tỷ lệ DPRR |
|---|---|---|---|---|
| **Nhóm 1** | Nợ đủ tiêu chuẩn | 0–9 ngày | Khoản nợ trong hạn hoặc quá hạn < 10 ngày | **0%** |
| **Nhóm 2** | Nợ cần chú ý | 10–90 ngày | Hoặc khoản nợ điều chỉnh kỳ hạn trả nợ lần đầu còn trong hạn | **5%** |
| **Nhóm 3** | Nợ dưới tiêu chuẩn | 91–180 ngày | Hoặc khoản nợ gia hạn nợ lần đầu còn trong hạn | **20%** |
| **Nhóm 4** | Nợ nghi ngờ | 181–360 ngày | Hoặc khoản nợ điều chỉnh kỳ hạn lần 2 còn trong hạn | **50%** |
| **Nhóm 5** | Nợ có khả năng mất vốn | > 360 ngày | Hoặc khoản nợ điều chỉnh kỳ hạn lần 3+ | **100%** |

### 3.2 Các trường hợp phân loại đặc biệt

Ngoài DPD, khoản nợ phải được phân loại nhóm cao hơn khi:
- Khoản nợ được cơ cấu lại thời hạn trả nợ
- Khách hàng được miễn, giảm lãi
- Nợ được chính TCTD mua lại sau khi bán cho VAMC
- TCTD đánh giá khách hàng suy giảm khả năng trả nợ dù chưa quá hạn

### 3.3 Nguyên tắc nhóm nợ duy nhất

**Quy tắc quan trọng nhất**: Toàn bộ dư nợ của một khách hàng tại một TCTD phải được phân loại vào cùng **một nhóm nợ** — nhóm có mức độ rủi ro cao nhất:

> "Nếu khách hàng có nhiều khoản nợ tại TCTD, tất cả các khoản nợ phải được phân loại vào nhóm nợ có mức độ rủi ro cao nhất."

## 4. Phân loại nợ — Phương pháp định tính (Điều 11)

### 4.1 Điều kiện áp dụng
TCTD được phép áp dụng phương pháp định tính khi:
1. Có hệ thống xếp hạng tín dụng nội bộ (Internal Rating System - IRS)
2. IRS được NHNN chấp thuận
3. Có chính sách dự phòng rủi ro được NHNN phê duyệt

### 4.2 Xếp hạng tín dụng nội bộ

| Xếp hạng nội bộ | Nhóm nợ tương ứng | Mô tả |
|---|---|---|
| AAA, AA | Nhóm 1 | Rủi ro rất thấp, khả năng trả nợ cao |
| A, BBB | Nhóm 2 | Rủi ro thấp-trung bình |
| BB, B | Nhóm 3 | Rủi ro trung bình-cao |
| CCC, CC | Nhóm 4 | Rủi ro cao |
| C, D | Nhóm 5 | Rủi ro rất cao, mất khả năng trả nợ |

### 4.3 Ứng dụng trong CreditLens

CreditLens sử dụng phương pháp định tính kết hợp ML scoring:

| CreditLens Risk Band | Nhóm nợ mapping | Hành động |
|---|---|---|
| AAA (720–850) | Nhóm 1 | AUTO APPROVE |
| AA (640–719) | Nhóm 1–2 | APPROVE + REVIEW |
| A (560–639) | Nhóm 2 | FULL REVIEW |
| BBB (460–559) | Nhóm 2–3 | CONDITIONAL |
| CC/C (300–459) | Nhóm 3+ | REJECT |

## 5. Trích lập Dự phòng Rủi ro (Điều 12–16)

### 5.1 Dự phòng cụ thể

Công thức tính dự phòng cụ thể cho mỗi khoản nợ:

```
R = max(0, (A − C)) × r
```

Trong đó:
- **R**: Số tiền dự phòng cụ thể phải trích
- **A**: Giá trị khoản nợ (dư nợ gốc)
- **C**: Giá trị khấu trừ của TSBĐ (giá trị tối đa sau khi áp dụng tỷ lệ khấu trừ)
- **r**: Tỷ lệ trích lập theo nhóm nợ (0%, 5%, 20%, 50%, 100%)

### 5.2 Tỷ lệ khấu trừ TSBĐ

| Loại TSBĐ | Tỷ lệ khấu trừ tối đa |
|---|---|
| Tiền gửi tại chính TCTD | 100% |
| Trái phiếu Chính phủ VN | 95% |
| Sổ tiết kiệm tại TCTD khác | 95% |
| Bất động sản | 50% (tối đa) |
| Phương tiện giao thông | 50% |
| Hàng hóa, nguyên liệu | 30% |
| Bảo lãnh bên thứ ba | Tùy hạng tín nhiệm bên bảo lãnh |

### 5.3 Dự phòng chung

```
Dự phòng chung = 0.75% × Tổng dư nợ (Nhóm 1 đến Nhóm 4)
```

Lưu ý: Không bao gồm dư nợ Nhóm 5 (đã trích 100% dự phòng cụ thể).

### 5.4 Ví dụ tính dự phòng

| Khoản vay | Dư nợ | TSBĐ | Nhóm nợ | DPRR cụ thể | Tính toán |
|---|---|---|---|---|---|
| Vay mua xe | 300M VND | Xe 450M (khấu trừ 50% = 225M) | Nhóm 2 | 5% | max(0, 300M – 225M) × 5% = 3.75M |
| Vay tiêu dùng | 100M VND | Không có TSBĐ | Nhóm 3 | 20% | max(0, 100M – 0) × 20% = 20M |
| Vay kinh doanh | 500M VND | BĐS 700M (khấu trừ 50% = 350M) | Nhóm 1 | 0% | 0 VND |

## 6. Tần suất và Quy trình (Điều 8–9)

### 6.1 Lịch trình thực hiện

| Hoạt động | Thời hạn | Mô tả |
|---|---|---|
| Phân loại nợ + trích lập DPRR | Hàng tháng, 7 ngày đầu tháng | TCTD tự phân loại toàn bộ danh mục |
| Gửi kết quả cho CIC | Ngay sau phân loại | Gửi danh sách nhóm nợ về CIC |
| CIC tổng hợp và phản hồi | 3 ngày sau nhận | CIC gửi danh sách nhóm nợ cao nhất cho TCTD |
| TCTD điều chỉnh | 3 ngày sau nhận từ CIC | Điều chỉnh nhóm nợ theo danh sách CIC |
| Sử dụng dự phòng xử lý rủi ro | Khi đủ điều kiện | Theo quy định Điều 17–19 |

### 6.2 Đồng bộ qua CIC

Quy trình đồng bộ nhóm nợ qua CIC:
1. TCTD A phân loại khách hàng X vào Nhóm 3
2. Gửi về CIC
3. CIC tổng hợp — khách hàng X ở Nhóm 3 (cao nhất)
4. CIC thông báo cho TCTD B, C (cũng có KH X)
5. TCTD B, C phải điều chỉnh KH X lên ít nhất Nhóm 3

## 7. Sử dụng Dự phòng Xử lý Rủi ro (Điều 17–19)

### 7.1 Điều kiện xử lý rủi ro

TCTD sử dụng dự phòng rủi ro đã trích lập khi:
1. Khách hàng là **tổ chức bị giải thể, phá sản** theo quy định pháp luật
2. Khách hàng là **cá nhân bị chết, mất tích** theo phán quyết tòa án
3. Khoản nợ đã thuộc **Nhóm 5** theo kết quả phân loại
4. TCTD đã áp dụng **mọi biện pháp thu hồi** nhưng không thành công

### 7.2 Xuất toán ngoại bảng

Sau tối thiểu **5 năm** từ khi xử lý rủi ro và đã thực hiện mọi biện pháp thu hồi, TCTD có thể quyết định **xuất toán** nợ đã xử lý khỏi ngoại bảng.

## 8. Ứng dụng trong CreditLens — Mapping sang Báo cáo A4

### 8.1 Sử dụng trong section 5C Assessment

| Phần báo cáo | Trích dẫn TT11 | Nội dung |
|---|---|---|
| Character Assessment | Điều 10 — Nhóm nợ CIC | "Khách hàng thuộc Nhóm 1 CIC — nợ đủ tiêu chuẩn" |
| Capacity Assessment | Điều 12 — DPRR | "DTI 48% → cần trích lập dự phòng cao hơn nếu chuyển nhóm" |
| Capital Assessment | Điều 14 — Tỷ lệ khấu trừ TSBĐ | "LTV 67% → TSBĐ đủ giá trị khấu trừ" |
| Conditions Assessment | Điều 11 — Xếp hạng nội bộ | "Risk Band AA → tương ứng Nhóm 1–2" |
| Caveats | Điều 8–9 — Tần suất | "Phân loại nợ hàng tháng theo TT11/2021" |

### 8.2 Hard override rules liên quan TT11

| Điều kiện | Hành động | Căn cứ TT11 |
|---|---|---|
| KH có dư nợ Nhóm 4–5 tại CIC | REJECT | Điều 10 khoản 3 |
| KH đã được cơ cấu nợ 2 lần | Phân loại tối thiểu Nhóm 4 | Điều 10 khoản 1.d |
| KH bị miễn/giảm lãi | Nâng nhóm nợ lên ít nhất Nhóm 2 | Điều 10 khoản 1.c |
