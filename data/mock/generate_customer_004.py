"""
Generate mock data for Customer 004: Lê Minh Cường — High-risk New Graduate.

Profile:
- Sinh viên mới ra trường, 23 tuổi, đi làm 4 tháng
- DTI rất cao (71%)
- Overdraft 5 lần trong 6 tháng
- CIC có 1 khoản vay sinh viên trễ hạn
- Vay mua laptop + điện thoại 60M (tiêu dùng)
- Không có tài sản thế chấp

Expected: Score ~400-450, Band CC/C, REJECT (high DTI, overdraft, late payments)

Usage:
    python data/mock/generate_customer_004.py
"""
import json, os, csv, random
from datetime import date, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMER_DIR = os.path.join(OUTPUT_DIR, "customer_004")
os.makedirs(CUSTOMER_DIR, exist_ok=True)

# ── Font setup ────────────────────────────────────────────────
_font_registered = False
for font_path in [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]:
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("VNFont", font_path))
        _font_registered = True
        break
FONT = "VNFont" if _font_registered else "Helvetica"


# ── Customer Profile ─────────────────────────────────────────
CUSTOMER = {
    "id": "CUST_400004",
    "cccd": "001003008765",
    "ho_ten": "Lê Minh Cường",
    "ngay_sinh": "2003-01-18",
    "gioi_tinh": "Nam",
    "quoc_tich": "Việt Nam",
    "dan_toc": "Kinh",
    "ton_giao": "Không",
    "que_quan": "Xã Đại Mỗ, Huyện Nam Từ Liêm, Hà Nội",
    "thuong_tru": "Phòng 205, KTX Đại học Bách Khoa, Hai Bà Trưng, Hà Nội",
    "ngay_cap_cccd": "2021-01-20",
    "noi_cap": "Cục Cảnh sát ĐKQL cư trú và DLQG về dân cư",
    "tinh_trang_hon_nhan": "Độc thân",
    "so_con": 0,
    "so_thanh_vien_gia_dinh": 1,

    # Employment — 4 months (mới đi làm)
    "ten_cong_ty": "Công ty CP Giải pháp Phần mềm FPT Software",
    "loai_doanh_nghiep": "Business Entity Type 3",
    "dia_chi_cty": "Tòa nhà FPT Cầu Giấy, Số 10 Phạm Văn Bạch, Cầu Giấy, Hà Nội",
    "chuc_vu": "Junior Developer / Lập trình viên tập sự",
    "phong_ban": "Phòng Phát triển Web",
    "ngay_bat_dau": "2025-11-15",  # Only 4 months ago!
    "loai_hop_dong": "Xác định thời hạn 1 năm (thử việc 2 tháng)",
    "luong_thang": 12000000,  # 12M VND/month (junior)
    "luong_nam": 144000000,
    "phu_cap": 1000000,  # 1M lunch allowance
    "bao_hiem": "BHXH, BHYT, BHTN (đang thử việc, 85% lương)",
    "sdt_cong_ty": "024 7300 7300",
    "email_cty": "cuong.leminh@fpt.com",
    "ma_so_thue_cty": "0101248141",
    "loai_thu_nhap": "Working",

    # Housing — Thuê phòng trọ nhỏ
    "loai_nha": "Rented apartment",
    "dia_chi_nha": "Phòng 3, Tầng 2, Nhà trọ 15 Ngõ 89 Láng Hạ, Đống Đa, Hà Nội",
    "dien_tich": 18.0,  # 18m2 — very small
    "nam_xay": 1995,
    "so_tang_toa_nha": 4,
    "tang_can_ho": 2,
    "co_thang_may": False,
    "vat_lieu_tuong": "Block",
    "tinh_trang": "Bình thường",
    "gia_tri_bds": 0,

    # Loan — Tiêu dùng (mua laptop + iPhone)
    "loai_hop_dong_vay": "Cash loans",
    "so_tien_vay": 60000000,     # 60M VND
    "ky_han": 24,                 # 24 months
    "lai_suat": 0.18,             # 18%/year (high risk unsecured)
    "muc_dich_vay": "Mua laptop gaming và điện thoại iPhone 16 Pro Max",
    "tra_hang_thang": 3050000,    # ~3.05M/month
    "gia_tri_hang_hoa": 65000000,
    "co_xe_oto": False,
    "tuoi_xe": None,
    "co_bat_dong_san": False,
    "nguoi_dong_hanh": "Unaccompanied",
    "trinh_do_hoc_van": "Higher education",
}


# ── PDF Helpers ───────────────────────────────────────────────
def draw_header(c, title, subtitle=""):
    c.setFont(FONT, 14)
    c.drawCentredString(A4[0]/2, A4[1] - 2*cm, title)
    if subtitle:
        c.setFont(FONT, 10)
        c.drawCentredString(A4[0]/2, A4[1] - 2.8*cm, subtitle)
    c.line(2*cm, A4[1] - 3.2*cm, A4[0] - 2*cm, A4[1] - 3.2*cm)
    return A4[1] - 4*cm

def draw_field(c, y, label, value, x=2.5*cm, label_width=6*cm):
    c.setFont(FONT, 10)
    c.drawString(x, y, f"{label}:")
    c.drawString(x + label_width, y, str(value))
    return y - 0.6*cm

def draw_section(c, y, title):
    y -= 0.3*cm
    c.setFont(FONT, 11)
    c.setFillColor(colors.HexColor("#1a5276"))
    c.drawString(2.2*cm, y, title)
    c.setFillColor(colors.black)
    c.line(2.2*cm, y - 0.15*cm, A4[0] - 2*cm, y - 0.15*cm)
    return y - 0.7*cm


# ============================================================
# 1. CCCD
# ============================================================
def generate_cccd_pdf():
    path = os.path.join(CUSTOMER_DIR, "01_cccd.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc")
    y -= 0.5*cm
    c.setFont(FONT, 13)
    c.drawCentredString(A4[0]/2, y, "CĂN CƯỚC CÔNG DÂN")
    y -= 1*cm

    y = draw_section(c, y, "THÔNG TIN CÁ NHÂN")
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Quốc tịch", CUSTOMER["quoc_tich"])
    y = draw_field(c, y, "Dân tộc", CUSTOMER["dan_toc"])
    y = draw_field(c, y, "Tôn giáo", CUSTOMER["ton_giao"])
    y = draw_field(c, y, "Quê quán", CUSTOMER["que_quan"])
    y = draw_field(c, y, "Nơi thường trú", CUSTOMER["thuong_tru"])

    y = draw_section(c, y, "THÔNG TIN CẤP VÀ HIỆU LỰC")
    y = draw_field(c, y, "Ngày cấp", CUSTOMER["ngay_cap_cccd"])
    y = draw_field(c, y, "Nơi cấp", CUSTOMER["noi_cap"])
    y = draw_field(c, y, "Có giá trị đến", "2031-01-20")

    y = draw_section(c, y, "ĐẶC ĐIỂM NHẬN DẠNG")
    y = draw_field(c, y, "Chiều cao", "175 cm")
    y = draw_field(c, y, "Màu mắt", "Đen")

    c.showPage()
    y = draw_header(c, "CĂN CƯỚC CÔNG DÂN - MẶT SAU")
    y = draw_section(c, y, "XÁC NHẬN ĐĂNG KÝ CƯ TRÚ")
    y = draw_field(c, y, "Địa chỉ đăng ký", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2021-02-01")
    y = draw_field(c, y, "Địa chỉ hiện tại", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Địa chỉ làm việc", CUSTOMER["dia_chi_cty"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 2. HĐLĐ (thử việc 1 năm)
# ============================================================
def generate_labor_contract_pdf():
    path = os.path.join(CUSTOMER_DIR, "02_hop_dong_lao_dong.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "HỢP ĐỒNG LAO ĐỘNG", "So: HDLD-2025-FPT8765")
    y = draw_section(c, y, "BÊN SỬ DỤNG LAO ĐỘNG (BÊN A)")
    y = draw_field(c, y, "Tên doanh nghiệp", CUSTOMER["ten_cong_ty"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_cty"])
    y = draw_field(c, y, "Mã số thuế", CUSTOMER["ma_so_thue_cty"])
    y = draw_field(c, y, "Điện thoại", CUSTOMER["sdt_cong_ty"])
    y = draw_field(c, y, "Đại diện", "Ông Nguyễn Hoàng Minh - Trưởng phòng HR")

    y = draw_section(c, y, "NGƯỜI LAO ĐỘNG (BÊN B)")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Điện thoại", "0365 432 109")
    y = draw_field(c, y, "Email", CUSTOMER["email_cty"])
    y = draw_field(c, y, "Trình độ", "Cử nhân CNTT - Đại học Bách Khoa Hà Nội (2025)")

    y = draw_section(c, y, "ĐIỀU 1: CÔNG VIỆC")
    y = draw_field(c, y, "Chức danh", CUSTOMER["chuc_vu"])
    y = draw_field(c, y, "Phòng ban", CUSTOMER["phong_ban"])
    y = draw_field(c, y, "Ngày bắt đầu", CUSTOMER["ngay_bat_dau"])
    y = draw_field(c, y, "Loại hợp đồng", CUSTOMER["loai_hop_dong"])
    y = draw_field(c, y, "Thử việc", "2 tháng (15/11/2025 - 15/01/2026)")

    # Page 2
    c.showPage()
    y = draw_header(c, "HỢP ĐỒNG LAO ĐỘNG (tiếp theo)")
    y = draw_section(c, y, "ĐIỀU 2: LƯƠNG VÀ CHẾ ĐỘ")
    y = draw_field(c, y, "Lương cơ bản", f"{CUSTOMER['luong_thang']:,.0f} VND/thang")
    y = draw_field(c, y, "Lương thử việc (85%)", f"{int(CUSTOMER['luong_thang'] * 0.85):,.0f} VND/thang")
    y = draw_field(c, y, "Phụ cấp ăn trưa", f"{CUSTOMER['phu_cap']:,.0f} VND/thang")
    y = draw_field(c, y, "Tổng thu nhập", f"{CUSTOMER['luong_thang'] + CUSTOMER['phu_cap']:,.0f} VND/thang")
    y = draw_field(c, y, "Hình thức trả", "Chuyển khoản qua ngân hàng")
    y = draw_field(c, y, "Ngân hàng", "MB Bank - Chi nhánh Hà Nội")
    y = draw_field(c, y, "Số tài khoản", "0987654321098")
    y = draw_field(c, y, "Ngày trả lương", "Ngày 25 hàng tháng")

    y = draw_section(c, y, "ĐIỀU 3: BẢO HIỂM")
    y = draw_field(c, y, "BHXH", CUSTOMER["bao_hiem"])
    y = draw_field(c, y, "Ghi chú", "Lương thử việc = 85% lương chính thức")

    y = draw_section(c, y, "ĐIỀU 4: ĐIỀU KHOẢN KHÁC")
    c.setFont(FONT, 9)
    for txt in [
        "- Sau thử việc sẽ ký HĐ chính thức 1 năm.",
        "- NLĐ cam kết làm việc tối thiểu 1 năm sau thử việc.",
        "- Vi phạm cam kết sẽ bồi thường 1 tháng lương.",
    ]:
        y = draw_field(c, y, "", txt)

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "BÊN A: Nguyễn Hoàng Minh")
    c.drawString(12*cm, y, f"BÊN B: {CUSTOMER['ho_ten']}")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 3. Sổ hộ khẩu (ở KTX / nhà bố mẹ)
# ============================================================
def generate_household_pdf():
    path = os.path.join(CUSTOMER_DIR, "03_so_ho_khau.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "SỔ HỘ KHẨU", "So: HK-2003-008765")
    y = draw_section(c, y, "THÔNG TIN CHỦ HỘ")
    y = draw_field(c, y, "Họ và tên chủ hộ", "Lê Văn Thành")
    y = draw_field(c, y, "Giới tính", "Nam")
    y = draw_field(c, y, "Ngày sinh", "1975-06-20")
    y = draw_field(c, y, "Số CCCD", "001075001234")
    y = draw_field(c, y, "Địa chỉ thường trú", "Số 10, Ngõ 12, Xã Đại Mỗ, Nam Từ Liêm, Hà Nội")
    y = draw_field(c, y, "Ngày đăng ký", "2003-02-01")

    y = draw_section(c, y, "THÀNH VIÊN TRONG HỘ")
    y = draw_field(c, y, "1. Họ tên", "Phạm Thị Nga")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Vợ")
    y = draw_field(c, y, "   Ngày sinh", "1978-03-10")
    y = draw_field(c, y, "   Nghề nghiệp", "Giáo viên tiểu học")

    y -= 0.3*cm
    y = draw_field(c, y, "2. Họ tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Con trai")
    y = draw_field(c, y, "   Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "   Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "   Nghề nghiệp", "Lập trình viên (mới ra trường)")
    y = draw_field(c, y, "   Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])

    y -= 0.3*cm
    y = draw_field(c, y, "3. Họ tên", "Lê Thị Ngọc")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Con gái")
    y = draw_field(c, y, "   Ngày sinh", "2006-09-05")
    y = draw_field(c, y, "   Nghề nghiệp", "Học sinh lớp 11")

    c.showPage()
    y = draw_header(c, "SỔ HỘ KHẨU (tiếp theo)")
    y = draw_section(c, y, "TỔNG HỢP")
    y = draw_field(c, y, "Tổng số nhân khẩu", "4 người")
    y = draw_field(c, y, "Tình trạng hôn nhân người đề nghị", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Số con", "0")
    y = draw_field(c, y, "Ghi chú", "Người đề nghị vay đang ở trọ, hộ khẩu tại nhà bố mẹ")

    y = draw_section(c, y, "XÁC NHẬN")
    y = draw_field(c, y, "Ngày xác nhận", "2025-06-20")
    y = draw_field(c, y, "Cơ quan", "Công an Xã Đại Mỗ, Nam Từ Liêm, Hà Nội")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 4. Thẩm định nhà ở (thuê trọ nhỏ)
# ============================================================
def generate_housing_pdf():
    path = os.path.join(CUSTOMER_DIR, "04_tham_dinh_nha_o.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "PHIẾU THẨM ĐỊNH TÀI SẢN / NHÀ Ở", "Ma ho so: TD-2026-HR004")
    y = draw_section(c, y, "1. THÔNG TIN NGƯỜI ĐỀ NGHỊ")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])

    y = draw_section(c, y, "2. THÔNG TIN NHÀ Ở HIỆN TẠI")
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Loại hình", CUSTOMER["loai_nha"])
    y = draw_field(c, y, "Hình thức", "Thuê phòng trọ")
    y = draw_field(c, y, "Tiền thuê", "3,000,000 VND/tháng")
    y = draw_field(c, y, "Diện tích", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Năm xây dựng", str(CUSTOMER["nam_xay"]))
    y = draw_field(c, y, "Số tầng", str(CUSTOMER["so_tang_toa_nha"]))
    y = draw_field(c, y, "Có thang máy", "Không")
    y = draw_field(c, y, "Vật liệu tường", CUSTOMER["vat_lieu_tuong"])

    y = draw_section(c, y, "3. THÔNG SỐ KỸ THUẬT")
    y = draw_field(c, y, "Chất lượng (1-10)", "3.5 / 10")
    y = draw_field(c, y, "Diện tích sinh hoạt", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Ghi chú", "Phòng trọ nhỏ, cơ bản, khu dân cư cũ")

    y = draw_section(c, y, "4. THÔNG TIN KHU VỰC")
    y = draw_field(c, y, "Quận/Huyện", "Đống Đa, Hà Nội")
    y = draw_field(c, y, "Mật độ dân số (tương đối)", "0.045 (khu nội thành)")
    y = draw_field(c, y, "Xếp hạng khu vực", "2 (Tốt)")
    y = draw_field(c, y, "Xếp hạng khu vực (TP)", "2 (Tốt)")
    y = draw_field(c, y, "Đăng ký cùng vùng sống", "Không (hộ khẩu Nam Từ Liêm)")
    y = draw_field(c, y, "Đăng ký cùng TP làm việc", "Có")
    y = draw_field(c, y, "Sống cùng vùng làm việc", "Không (sống Đống Đa, làm việc Cầu Giấy)")

    y = draw_section(c, y, "5. TÀI SẢN BẢO ĐẢM")
    y = draw_field(c, y, "Loại TSBĐ", "KHÔNG CÓ")
    y = draw_field(c, y, "Ghi chú", "Người vay không có tài sản thế chấp")

    y = draw_section(c, y, "6. KẾT LUẬN")
    y = draw_field(c, y, "Kết luận", "Không có tài sản thế chấp, phòng trọ không đủ điều kiện")
    y = draw_field(c, y, "Khuyến nghị", "Cân nhắc kỹ — khoản vay tín chấp rủi ro cao")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 5. Đơn vay
# ============================================================
def generate_loan_application_pdf():
    path = os.path.join(CUSTOMER_DIR, "05_don_vay.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "ĐƠN ĐỀ NGHỊ VAY VỐN", "Ma don: DV-2026-HR004")
    y = draw_section(c, y, "I. THÔNG TIN NGƯỜI VAY")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Trình độ học vấn", CUSTOMER["trinh_do_hoc_van"])
    y = draw_field(c, y, "Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "SDT", "0365 432 109")
    y = draw_field(c, y, "Email", "cuong.leminh2003@gmail.com")
    y = draw_field(c, y, "Có xe ô tô", "Không")
    y = draw_field(c, y, "Có bất động sản", "Không")
    y = draw_field(c, y, "Người đồng hành", CUSTOMER["nguoi_dong_hanh"])
    y = draw_field(c, y, "Số di động liên lạc được", "Có")
    y = draw_field(c, y, "Email", "Có")
    y = draw_field(c, y, "Số điện thoại bàn", "Không")
    y = draw_field(c, y, "Ngày đổi SĐT gần nhất", "2025-10-01 (khoảng 170 ngày trước)")

    y = draw_section(c, y, "II. THÔNG TIN KHOẢN VAY")
    y = draw_field(c, y, "Loại hợp đồng", CUSTOMER["loai_hop_dong_vay"])
    y = draw_field(c, y, "Số tiền vay", f"{CUSTOMER['so_tien_vay']:,.0f} VND")
    y = draw_field(c, y, "Kỳ hạn", f"{CUSTOMER['ky_han']} thang")
    y = draw_field(c, y, "Trả hàng tháng (dự kiến)", f"{CUSTOMER['tra_hang_thang']:,.0f} VND")
    y = draw_field(c, y, "Giá trị hàng hóa", f"{CUSTOMER['gia_tri_hang_hoa']:,.0f} VND")

    # Page 2
    c.showPage()
    y = draw_header(c, "ĐƠN ĐỀ NGHỊ VAY VỐN (tiếp theo)")
    y = draw_section(c, y, "III. MỤC ĐÍCH VAY")
    y = draw_field(c, y, "Mục đích", CUSTOMER["muc_dich_vay"])
    y = draw_field(c, y, "Chi tiết", "Laptop Asus ROG Strix G16 (38M VND)")
    y = draw_field(c, y, "", "iPhone 16 Pro Max 256GB (27M VND)")

    y = draw_section(c, y, "IV. TÀI SẢN THẾ CHẤP")
    y = draw_field(c, y, "Loại TSĐB", "Không có")
    y = draw_field(c, y, "Ghi chú", "Khoản vay hoàn toàn tín chấp")

    y = draw_section(c, y, "V. NGUỒN TRẢ NỢ")
    total_income = CUSTOMER["luong_thang"] + CUSTOMER["phu_cap"]
    y = draw_field(c, y, "Thu nhập hàng tháng", f"{total_income:,.0f} VND")
    y = draw_field(c, y, "Nguồn thu nhập", "Lương tại FPT Software")
    y = draw_field(c, y, "Chi phí sinh hoạt", "8,000,000 VND/thang")
    y = draw_field(c, y, "Tiền thuê phòng", "3,000,000 VND/thang")
    y = draw_field(c, y, "Thu nhập khả dụng", f"{total_income - 8000000 - 3000000:,.0f} VND/thang")
    dti = (CUSTOMER["tra_hang_thang"] + 2500000) / total_income * 100  # existing student loan + new
    y = draw_field(c, y, "Tỷ lệ nợ/thu nhập (DTI)", f"{dti:.1f}% (bao gồm khoản vay sinh viên còn lại)")

    y = draw_section(c, y, "VI. CAM KẾT")
    c.setFont(FONT, 9)
    for txt in [
        "- Tôi cam kết các thông tin trên là đúng sự thật.",
        "- Tôi đồng ý để ngân hàng xác minh thông tin.",
        "- Tôi cam kết trả nợ đúng hạn.",
    ]:
        y = draw_field(c, y, "", txt)

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "Hà Nội, ngay 20 thang 03 nam 2026")
    y -= 1.5*cm
    c.drawString(3*cm, y, CUSTOMER["ho_ten"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 6. Bank Statement — Overdrafts, high spending, low balance
# ============================================================
def generate_bank_statement():
    path = os.path.join(CUSTOMER_DIR, "06_sao_ke_ngan_hang.csv")
    transactions = []
    base_date = date(2025, 9, 1)
    balance = 3500000  # Very low starting balance
    random.seed(404)

    for month in range(6):
        month_date = base_date + timedelta(days=30 * month)

        if month < 2:
            # Pre-employment: only student loan repayment + family support
            # Family support
            support_date = month_date + timedelta(days=random.randint(1, 5))
            balance += 5000000
            transactions.append({
                "date": support_date.isoformat(),
                "description": "CK TU LE VAN THANH - HO TRO SINH HOAT",
                "credit": 5000000, "debit": 0, "balance": balance
            })
        else:
            # Post-employment: salary on 25th
            salary_date = month_date.replace(day=25)
            salary = CUSTOMER["luong_thang"] + CUSTOMER["phu_cap"]
            if month == 2:
                salary = int(salary * 0.85)  # Probation — 85%
            balance += salary
            transactions.append({
                "date": salary_date.isoformat(),
                "description": "LUONG T" + str(month_date.month) + "/2025 - FPT SOFTWARE",
                "credit": salary, "debit": 0, "balance": balance
            })

        # Student loan repayment — 10th (sometimes late)
        loan_date = month_date.replace(day=10 if month != 3 else 18)  # Month 3: late!
        balance -= 2500000
        transactions.append({
            "date": loan_date.isoformat(),
            "description": "TRA NO VAY SINH VIEN - NGAN HANG CHINH SACH",
            "credit": 0, "debit": 2500000, "balance": balance
        })

        # Rent — 1st
        rent_date = month_date.replace(day=1)
        balance -= 3000000
        transactions.append({
            "date": rent_date.isoformat(),
            "description": "TIEN THUE PHONG TRO T" + str(month_date.month),
            "credit": 0, "debit": 3000000, "balance": balance
        })

        # Heavy spending (young, spending more than earning)
        spending_items = [
            ("SHOPEE - MUA HANG ONLINE", 1200000),
            ("LAZADA - PHAN CUNG PC", 850000),
            ("GRAB FOOD - AN NGOAI", 580000),
            ("GAME TOPUP - GARENA", 500000),
            ("HIGHLANDS COFFEE", 250000),
            ("THE THAO - YOGA/GYM", 400000),
            ("GD ONLINE - TIKI", 650000),
            ("GRAB - DI CHUYEN", 380000),
            ("BIA CRAFT - GIAI TRI", 450000),
            ("SHOPEE - MUA HANG", 780000),
        ]

        for i, (desc, amt) in enumerate(spending_items):
            sp_date = month_date + timedelta(days=random.randint(2, 28))
            balance -= amt
            transactions.append({
                "date": sp_date.isoformat(),
                "description": desc,
                "credit": 0, "debit": amt, "balance": balance
            })

        # Bills
        bill_date = month_date + timedelta(days=15)
        for d, a in [("TIEN DIEN", 350000), ("TIEN NUOC", 80000), ("CUOC 4G VIETTEL", 200000)]:
            balance -= a
            transactions.append({
                "date": bill_date.isoformat(),
                "description": d, "credit": 0, "debit": a, "balance": balance
            })

        # ⚠️ Overdraft situations — balance goes negative sometimes
        if balance < 0:
            # Emergency family transfer to cover
            emergency_date = month_date + timedelta(days=28)
            emergency_amt = abs(balance) + 500000
            balance += emergency_amt
            transactions.append({
                "date": emergency_date.isoformat(),
                "description": "CK TU LE VAN THANH - BO HO TRO KHAN CAP",
                "credit": emergency_amt, "debit": 0, "balance": balance
            })

    transactions.sort(key=lambda x: x["date"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "description", "credit", "debit", "balance"])
        writer.writeheader()
        writer.writerows(transactions)
    print(f"  Created: {path} ({len(transactions)} transactions)")


# ============================================================
# 7. CIC — Student loan with late payments
# ============================================================
def generate_cic_mock():
    path = os.path.join(CUSTOMER_DIR, "07_cic_api_response.json")

    def make_monthly_status(n_months):
        """Student loan: mostly late (DPD 1-30)."""
        statuses = []
        for m in range(n_months):
            if m < 2:
                s = "0"   # Recent: on time
            elif m < 4:
                s = "1"   # DPD 1-30 days
            elif m < 6:
                s = "0"   # On time
            elif m < 8:
                s = "1"   # Late again
            elif m < 10:
                s = "2"   # DPD 31-60 once!
            else:
                s = "0"   # Older months OK
            statuses.append({"MONTHS_BALANCE": -m, "STATUS": s})
        return statuses

    cic_data = {
        "api_version": "CIC-VN-v2.1",
        "query_timestamp": "2026-03-20T16:00:00+07:00",
        "customer_id": CUSTOMER["cccd"],
        "customer_name": CUSTOMER["ho_ten"],

        "ext_source_scores": {
            "EXT_SOURCE_1": 0.210,
            "EXT_SOURCE_2": 0.285,
            "EXT_SOURCE_3": 0.195,
            "_note": "Very low scores — young borrower, late payments, thin history"
        },

        "credit_inquiry_counts": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
            "AMT_REQ_CREDIT_BUREAU_DAY": 1,
            "AMT_REQ_CREDIT_BUREAU_WEEK": 2,
            "AMT_REQ_CREDIT_BUREAU_MON": 3,
            "AMT_REQ_CREDIT_BUREAU_QRT": 5,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 8
        },

        "social_circle": {
            "OBS_30_CNT_SOCIAL_CIRCLE": 3,
            "DEF_30_CNT_SOCIAL_CIRCLE": 1,
            "OBS_60_CNT_SOCIAL_CIRCLE": 2,
            "DEF_60_CNT_SOCIAL_CIRCLE": 1
        },

        "bureau_records": [
            {
                "SK_ID_BUREAU": 5004001,
                "CREDIT_ACTIVE": "Active",
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -400,
                "CREDIT_DAY_OVERDUE": 15,
                "DAYS_CREDIT_ENDDATE": 300,
                "DAYS_ENDDATE_FACT": None,
                "AMT_CREDIT_MAX_OVERDUE": 2500000,
                "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": 50000000,
                "AMT_CREDIT_SUM_DEBT": 25000000,
                "AMT_CREDIT_SUM_LIMIT": 0,
                "AMT_CREDIT_SUM_OVERDUE": 2500000,
                "CREDIT_TYPE": "Consumer credit",
                "DAYS_CREDIT_UPDATE": -5,
                "AMT_ANNUITY": 2500000,
                "monthly_status": make_monthly_status(14)
            }
        ],

        "thin_file_flag": False,
        "cic_score_equivalent": 420,
        "debt_group": 2
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cic_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")


# ============================================================
# 8. Internal DB — 1 previous app (rejected for same product)
# ============================================================
def generate_internal_db_mock():
    sk_id_curr = 400004

    prev_apps = [{
        "SK_ID_PREV": 4001001,
        "SK_ID_CURR": sk_id_curr,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "AMT_ANNUITY": 2500000,
        "AMT_APPLICATION": 40000000,
        "AMT_CREDIT": 40000000,
        "AMT_DOWN_PAYMENT": 0,
        "AMT_GOODS_PRICE": 45000000,
        "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
        "HOUR_APPR_PROCESS_START": 16,
        "FLAG_LAST_APPL_PER_CONTRACT": "Y",
        "NFLAG_LAST_APPL_IN_DAY": 1,
        "RATE_DOWN_PAYMENT": 0.0,
        "RATE_INTEREST_PRIMARY": 0.18,
        "RATE_INTEREST_PRIVILEGED": 0.16,
        "NAME_CASH_LOAN_PURPOSE": "XAP",
        "NAME_CONTRACT_STATUS": "Refused",
        "DAYS_DECISION": -90,
        "NAME_PAYMENT_TYPE": "Cash through the bank",
        "CODE_REJECT_REASON": "HC",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_CLIENT_TYPE": "New",
        "NAME_GOODS_CATEGORY": "Consumer Electronics",
        "NAME_PORTFOLIO": "POS",
        "NAME_PRODUCT_TYPE": "walk-in",
        "CHANNEL_TYPE": "Country-wide",
        "SELLERPLACE_AREA": 100,
        "NAME_SELLER_INDUSTRY": "Consumer electronics",
        "CNT_PAYMENT": 12,
        "NAME_YIELD_GROUP": "high",
        "PRODUCT_COMBINATION": "POS industry with interest",
        "DAYS_FIRST_DRAWING": None,
        "DAYS_FIRST_DUE": None,
        "DAYS_LAST_DUE_1ST_VERSION": None,
        "DAYS_LAST_DUE": None,
        "DAYS_TERMINATION": None,
        "NFLAG_INSURED_ON_APPROVAL": 0
    }]

    # No installments (rejected application)
    internal_data = {
        "SK_ID_CURR": sk_id_curr,
        "previous_applications": prev_apps,
        "pos_cash_balance": [],
        "installments_payments": [],
        "credit_card_balance": []
    }

    path = os.path.join(CUSTOMER_DIR, "08_internal_db.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(internal_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")
    print(f"    - 1 previous application (REFUSED — reason: HC)")
    print(f"    - 0 installment records (rejected, never disbursed)")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Generating mock data for: {CUSTOMER['ho_ten']}")
    print(f"  Profile: High-risk New Graduate")
    print(f"  Output: {CUSTOMER_DIR}")
    print(f"{'='*60}\n")

    generate_cccd_pdf()
    generate_labor_contract_pdf()
    generate_household_pdf()
    generate_housing_pdf()
    generate_loan_application_pdf()
    generate_bank_statement()
    generate_cic_mock()
    generate_internal_db_mock()

    print(f"\n{'='*60}")
    print(f"  All mock data for {CUSTOMER['ho_ten']} generated!")
    print(f"{'='*60}")
