"""
Generate mock data for Customer 002: Phạm Thị Lan — Thin-file Freelancer.

Profile:
- Freelance graphic designer, 28 tuổi
- Không có CIC record (thin-file)
- Thu nhập không đều (freelance)
- Có sao kê ngân hàng 6 tháng
- Không có lịch sử vay trước
- Vay tiêu dùng 80M (laptop + thiết bị)

Expected: Score ~600-650, Band A, REVIEW (thin-file alternative path)

Usage:
    python data/mock/generate_customer_002.py
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
CUSTOMER_DIR = os.path.join(OUTPUT_DIR, "customer_002")
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
    "id": "CUST_200002",
    "cccd": "036098007654",
    "ho_ten": "Phạm Thị Lan",
    "ngay_sinh": "1998-03-22",
    "gioi_tinh": "Nữ",
    "quoc_tich": "Việt Nam",
    "dan_toc": "Kinh",
    "ton_giao": "Không",
    "que_quan": "Phường Hàng Bài, Quận Hoàn Kiếm, Hà Nội",
    "thuong_tru": "Số 12, Ngõ 45 Đặng Tiến Đông, Phường Trung Liệt, Quận Đống Đa, Hà Nội",
    "ngay_cap_cccd": "2022-07-10",
    "noi_cap": "Cục Cảnh sát ĐKQL cư trú và DLQG về dân cư",
    "tinh_trang_hon_nhan": "Độc thân",
    "so_con": 0,
    "so_thanh_vien_gia_dinh": 1,

    # Employment — Freelancer (no formal employer)
    "ten_cong_ty": "Freelance / Tự do",
    "loai_doanh_nghiep": "Cá nhân tự do",
    "dia_chi_cty": "Làm việc tại nhà - Số 12, Ngõ 45 Đặng Tiến Đông, Đống Đa, Hà Nội",
    "chuc_vu": "Graphic Designer / Thiết kế đồ họa",
    "phong_ban": "Không (freelance)",
    "ngay_bat_dau": "2022-01-15",
    "loai_hop_dong": "Không có hợp đồng lao động cố định",
    "luong_thang": 18000000,  # Average 18M (irregular)
    "luong_nam": 216000000,
    "phu_cap": 0,
    "bao_hiem": "Tự đóng BHYT cá nhân",
    "sdt_cong_ty": "Không có",
    "email_cty": "phamlan.design@gmail.com",
    "ma_so_thue_cty": "Chưa đăng ký MST cá nhân",
    "loai_thu_nhap": "Commercial associate",

    # Housing — Thuê trọ
    "loai_nha": "Rented apartment",
    "dia_chi_nha": "Phòng 302, Tầng 3, Nhà trọ 12 Ngõ 45 Đặng Tiến Đông, Đống Đa, Hà Nội",
    "dien_tich": 35.0,
    "nam_xay": 2005,
    "so_tang_toa_nha": 5,
    "tang_can_ho": 3,
    "co_thang_may": False,
    "vat_lieu_tuong": "Stone, brick",
    "tinh_trang": "Bình thường",
    "gia_tri_bds": 0,  # Thuê, không sở hữu

    # Loan request — mua thiết bị
    "loai_hop_dong_vay": "Cash loans",
    "so_tien_vay": 80000000,     # 80M VND
    "ky_han": 24,                 # 24 months
    "lai_suat": 0.15,             # 15%/year (higher for unsecured)
    "muc_dich_vay": "Mua laptop và thiết bị thiết kế đồ họa phục vụ công việc freelance",
    "tra_hang_thang": 3867000,    # ~3.87M/month
    "gia_tri_hang_hoa": 85000000,
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
    y = draw_field(c, y, "Có giá trị đến", "2032-07-10")

    y = draw_section(c, y, "ĐẶC ĐIỂM NHẬN DẠNG")
    y = draw_field(c, y, "Chiều cao", "162 cm")
    y = draw_field(c, y, "Màu mắt", "Nâu")

    # Page 2
    c.showPage()
    y = draw_header(c, "CĂN CƯỚC CÔNG DÂN - MẶT SAU")
    y = draw_section(c, y, "XÁC NHẬN ĐĂNG KÝ CƯ TRÚ")
    y = draw_field(c, y, "Địa chỉ đăng ký", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2022-08-01")
    y = draw_field(c, y, "Địa chỉ hiện tại", CUSTOMER["dia_chi_nha"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 2. Hợp đồng cộng tác viên (thay vì HĐLĐ chính thức)
# ============================================================
def generate_labor_contract_pdf():
    path = os.path.join(CUSTOMER_DIR, "02_hop_dong_lao_dong.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "HỢP ĐỒNG CỘNG TÁC VIÊN", "So: CTV-2024-FL002")
    y = draw_section(c, y, "BÊN THUÊ DỊCH VỤ")
    y = draw_field(c, y, "Tên doanh nghiệp", "Công ty TNHH Truyền thông StarMedia")
    y = draw_field(c, y, "Địa chỉ", "Tầng 8, 36 Hoàng Cầu, Đống Đa, Hà Nội")
    y = draw_field(c, y, "Mã số thuế", "0109876543")
    y = draw_field(c, y, "Điện thoại", "024 9876 5432")
    y = draw_field(c, y, "Đại diện", "Bà Trần Thị Mai Hương - Giám đốc")

    y = draw_section(c, y, "BÊN CUNG CẤP DỊCH VỤ (FREELANCER)")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Điện thoại", "0987 654 321")
    y = draw_field(c, y, "Email", CUSTOMER["email_cty"])
    y = draw_field(c, y, "Lĩnh vực", "Thiết kế đồ họa, UI/UX")

    y = draw_section(c, y, "ĐIỀU 1: NỘI DUNG CÔNG VIỆC")
    y = draw_field(c, y, "Công việc", CUSTOMER["chuc_vu"])
    y = draw_field(c, y, "Hình thức", "Cộng tác viên / Freelance")
    y = draw_field(c, y, "Ngày bắt đầu", CUSTOMER["ngay_bat_dau"])
    y = draw_field(c, y, "Loại hợp đồng", CUSTOMER["loai_hop_dong"])
    y = draw_field(c, y, "Địa điểm", "Làm việc từ xa (remote)")

    # Page 2
    c.showPage()
    y = draw_header(c, "HỢP ĐỒNG CỘNG TÁC VIÊN (tiếp theo)")
    y = draw_section(c, y, "ĐIỀU 2: THANH TOÁN")
    y = draw_field(c, y, "Phí dịch vụ", "Theo dự án, trung bình 15,000,000 - 25,000,000 VND/tháng")
    y = draw_field(c, y, "Thu nhập trung bình", f"{CUSTOMER['luong_thang']:,.0f} VND/thang (ước tính)")
    y = draw_field(c, y, "Hình thức thanh toán", "Chuyển khoản sau khi nghiệm thu")
    y = draw_field(c, y, "Ngân hàng nhận", "Techcombank - Hà Nội")
    y = draw_field(c, y, "Số tài khoản", "19038765432100")
    y = draw_field(c, y, "Phụ cấp", "Không")
    y = draw_field(c, y, "Bảo hiểm", CUSTOMER["bao_hiem"])

    y = draw_section(c, y, "ĐIỀU 3: THỜI GIAN")
    y = draw_field(c, y, "Thời gian làm việc", "Linh hoạt, tự quản lý")
    y = draw_field(c, y, "Deadline dự án", "Theo thỏa thuận từng dự án")

    y = draw_section(c, y, "ĐIỀU 4: DANH MỤC DỰ ÁN GẦN ĐÂY")
    projects = [
        "Thiết kế bộ nhận diện thương hiệu - CafeMo (15M VND, T9/2025)",
        "UI/UX App di động - HealthTrack (22M VND, T10-T11/2025)",
        "Thiết kế catalogue - NhaTot.vn (12M VND, T12/2025)",
        "Landing page - TechConf 2026 (18M VND, T1/2026)",
        "Branding package - GreenLeaf Organic (20M VND, T2-T3/2026)",
    ]
    c.setFont(FONT, 9)
    for p in projects:
        y = draw_field(c, y, "", f"- {p}")

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "BÊN THUÊ")
    c.drawString(12*cm, y, "BÊN CUNG CẤP")
    y -= 1.5*cm
    c.drawString(3*cm, y, "Trần Thị Mai Hương")
    c.drawString(12*cm, y, CUSTOMER["ho_ten"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 3. Sổ hộ khẩu (ở cùng bố mẹ)
# ============================================================
def generate_household_pdf():
    path = os.path.join(CUSTOMER_DIR, "03_so_ho_khau.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "SỔ HỘ KHẨU", "So: HK-2010-007654")
    y = draw_section(c, y, "THÔNG TIN CHỦ HỘ")
    y = draw_field(c, y, "Họ và tên chủ hộ", "Phạm Văn Hải")
    y = draw_field(c, y, "Giới tính", "Nam")
    y = draw_field(c, y, "Ngày sinh", "1968-11-05")
    y = draw_field(c, y, "Số CCCD", "036068009876")
    y = draw_field(c, y, "Địa chỉ thường trú", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2010-03-20")

    y = draw_section(c, y, "THÀNH VIÊN TRONG HỘ")
    # Member 1: Mother
    y = draw_field(c, y, "1. Họ tên", "Nguyễn Thị Thanh")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Vợ")
    y = draw_field(c, y, "   Ngày sinh", "1970-04-18")
    y = draw_field(c, y, "   Nghề nghiệp", "Buôn bán nhỏ - Chợ Đồng Xuân")

    # Member 2: The applicant
    y -= 0.3*cm
    y = draw_field(c, y, "2. Họ tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Con gái")
    y = draw_field(c, y, "   Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "   Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "   Nghề nghiệp", "Thiết kế đồ họa tự do")
    y = draw_field(c, y, "   Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])

    c.showPage()
    y = draw_header(c, "SỔ HỘ KHẨU (tiếp theo)")
    y = draw_section(c, y, "TỔNG HỢP")
    y = draw_field(c, y, "Tổng số nhân khẩu", "3 người")
    y = draw_field(c, y, "Chủ hộ", "Phạm Văn Hải (bố)")
    y = draw_field(c, y, "Tình trạng hôn nhân người đề nghị", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Số con của người đề nghị", "0")

    y = draw_section(c, y, "XÁC NHẬN")
    y = draw_field(c, y, "Ngày xác nhận", "2024-06-15")
    y = draw_field(c, y, "Cơ quan xác nhận", "Công an Phường Trung Liệt, Quận Đống Đa, Hà Nội")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 4. Thẩm định nhà ở (thuê trọ)
# ============================================================
def generate_housing_pdf():
    path = os.path.join(CUSTOMER_DIR, "04_tham_dinh_nha_o.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "PHIẾU THẨM ĐỊNH TÀI SẢN / NHÀ Ở", "Ma ho so: TD-2026-F002")
    y = draw_section(c, y, "1. THÔNG TIN NGƯỜI ĐỀ NGHỊ")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ hiện tại", CUSTOMER["dia_chi_nha"])

    y = draw_section(c, y, "2. THÔNG TIN NHÀ Ở")
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Loại hình nhà ở", CUSTOMER["loai_nha"])
    y = draw_field(c, y, "Hình thức sở hữu", "Thuê (không sở hữu)")
    y = draw_field(c, y, "Tiền thuê", "4,500,000 VND/tháng")
    y = draw_field(c, y, "Diện tích sử dụng", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Năm xây dựng", str(CUSTOMER["nam_xay"]))
    y = draw_field(c, y, "Số tầng tòa nhà", str(CUSTOMER["so_tang_toa_nha"]))
    y = draw_field(c, y, "Tầng ở", str(CUSTOMER["tang_can_ho"]))
    y = draw_field(c, y, "Có thang máy", "Không")
    y = draw_field(c, y, "Vật liệu tường", CUSTOMER["vat_lieu_tuong"])
    y = draw_field(c, y, "Tình trạng khẩn cấp", CUSTOMER["tinh_trang"])

    y = draw_section(c, y, "3. THÔNG SỐ KỸ THUẬT")
    y = draw_field(c, y, "Chất lượng căn hộ (1-10)", "5.0 / 10")
    y = draw_field(c, y, "Diện tích sinh hoạt", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Số lối vào", "1")

    c.showPage()
    y = draw_header(c, "PHIẾU THẨM ĐỊNH (tiếp theo)")
    y = draw_section(c, y, "4. THÔNG TIN KHU VỰC")
    y = draw_field(c, y, "Quận/Huyện", "Đống Đa, Hà Nội")
    y = draw_field(c, y, "Mật độ dân số (tương đối)", "0.042 (khu nội thành)")
    y = draw_field(c, y, "Xếp hạng khu vực", "2 (Tốt)")
    y = draw_field(c, y, "Xếp hạng khu vực (TP)", "2 (Tốt)")
    y = draw_field(c, y, "Đăng ký cùng vùng sống", "Có")
    y = draw_field(c, y, "Đăng ký cùng TP làm việc", "Có")
    y = draw_field(c, y, "Sống cùng vùng làm việc", "Có")

    y = draw_section(c, y, "5. TÀI SẢN BẢO ĐẢM")
    y = draw_field(c, y, "Loại TSBĐ", "Không có tài sản thế chấp")
    y = draw_field(c, y, "Ghi chú", "Khoản vay tín chấp (unsecured)")

    y = draw_section(c, y, "6. KẾT LUẬN")
    y = draw_field(c, y, "Kết luận", "Người đề nghị đang thuê trọ, không có BĐS sở hữu")
    y = draw_field(c, y, "Khuyến nghị", "Cho vay tín chấp với hạn mức thấp")

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "Nhân viên thẩm định:")
    y -= 1.5*cm
    c.drawString(3*cm, y, "Hoàng Minh Tuấn")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 5. Đơn vay
# ============================================================
def generate_loan_application_pdf():
    path = os.path.join(CUSTOMER_DIR, "05_don_vay.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "ĐƠN ĐỀ NGHỊ VAY VỐN", "Ma don: DV-2026-F002")
    y = draw_section(c, y, "I. THÔNG TIN NGƯỜI VAY")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Trình độ học vấn", CUSTOMER["trinh_do_hoc_van"])
    y = draw_field(c, y, "Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "SDT", "0987 654 321")
    y = draw_field(c, y, "Email", "phamlan.design@gmail.com")
    y = draw_field(c, y, "Có xe ô tô", "Không")
    y = draw_field(c, y, "Có bất động sản", "Không")
    y = draw_field(c, y, "Người đồng hành", CUSTOMER["nguoi_dong_hanh"])
    y = draw_field(c, y, "Số di động liên lạc được", "Có")
    y = draw_field(c, y, "Email", "Có")
    y = draw_field(c, y, "Số điện thoại bàn", "Không")
    y = draw_field(c, y, "Ngày đổi SĐT gần nhất", "2025-09-01 (khoảng 200 ngày trước)")

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
    y = draw_field(c, y, "Chi tiết", "MacBook Pro M3 Max (65M VND) + Màn hình Studio Display (20M VND)")
    y = draw_field(c, y, "", "Thiết bị phục vụ thiết kế đồ họa chuyên nghiệp cho freelance")

    y = draw_section(c, y, "IV. TÀI SẢN THẾ CHẤP")
    y = draw_field(c, y, "Loại TSĐB", "Không có (khoản vay tín chấp)")
    y = draw_field(c, y, "Ghi chú", "Đề nghị vay tín chấp dựa trên thu nhập freelance")

    y = draw_section(c, y, "V. NGUỒN TRẢ NỢ")
    y = draw_field(c, y, "Thu nhập hàng tháng (TB)", f"{CUSTOMER['luong_thang']:,.0f} VND")
    y = draw_field(c, y, "Nguồn thu nhập", "Thu nhập từ các dự án thiết kế freelance")
    y = draw_field(c, y, "Chi phí sinh hoạt", "10,000,000 VND/thang")
    y = draw_field(c, y, "Tiền thuê nhà", "4,500,000 VND/thang")
    y = draw_field(c, y, "Thu nhập khả dụng", "3,500,000 VND/thang")
    dti = CUSTOMER["tra_hang_thang"] / CUSTOMER["luong_thang"] * 100
    y = draw_field(c, y, "Tỷ lệ nợ/thu nhập (DTI)", f"{dti:.1f}%")

    y = draw_section(c, y, "VI. CAM KẾT")
    c.setFont(FONT, 9)
    for txt in [
        "- Tôi cam kết các thông tin trên là đúng sự thật.",
        "- Tôi đồng ý để ngân hàng xác minh thông tin và tra cứu CIC.",
        "- Tôi cam kết sử dụng vốn vay đúng mục đích.",
    ]:
        y = draw_field(c, y, "", txt)

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, f"Hà Nội, ngay 18 thang 03 nam 2026")
    y -= 1.5*cm
    c.drawString(3*cm, y, "Người vay ký tên:")
    y -= 1.5*cm
    c.drawString(3*cm, y, CUSTOMER["ho_ten"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 6. Bank Statement — Thu nhập không đều (freelance)
# ============================================================
def generate_bank_statement():
    path = os.path.join(CUSTOMER_DIR, "06_sao_ke_ngan_hang.csv")
    transactions = []
    base_date = date(2025, 9, 1)
    balance = 12000000  # Starting 12M (low savings)

    random.seed(42)

    # Monthly income varies significantly (freelance)
    monthly_incomes = [
        ("CK TU STARMEDIA - DU AN CAFEMO", 15000000),
        ("CK TU HEALTHTRACK APP - THIET KE UIUX", 22000000),
        ("CK TU NHATOT.VN - CATALOGUE DESIGN", 12000000),
        ("CK TU TECHCONF 2026 - LANDING PAGE", 18000000),
        ("CK TU GREENLEAF ORGANIC - BRANDING", 20000000),
        ("CK TU KHACH LE - LOGO DESIGN", 8000000),
    ]

    for month in range(6):
        month_date = base_date + timedelta(days=30 * month)

        # Income — irregular dates, varying amounts
        income_date = month_date + timedelta(days=random.randint(8, 22))
        desc, amt = monthly_incomes[month]
        balance += amt
        transactions.append({
            "date": income_date.isoformat(),
            "description": desc,
            "credit": amt, "debit": 0, "balance": balance
        })

        # Some months have a second smaller payment
        if month in [1, 4]:
            extra_date = month_date + timedelta(days=random.randint(3, 7))
            extra_amt = random.choice([5000000, 7000000, 6000000])
            balance += extra_amt
            transactions.append({
                "date": extra_date.isoformat(),
                "description": "CK TU KHACH LE - THIET KE NHO",
                "credit": extra_amt, "debit": 0, "balance": balance
            })

        # Rent — around 1st-3rd
        rent_date = month_date + timedelta(days=random.randint(1, 3))
        balance -= 4500000
        transactions.append({
            "date": rent_date.isoformat(),
            "description": "TIEN THUE PHONG T" + str(month_date.month),
            "credit": 0, "debit": 4500000, "balance": balance
        })

        # Bills
        bill_date = month_date + timedelta(days=random.randint(14, 18))
        for desc_b, amt_b in [("TIEN DIEN EVN HA NOI", 450000), ("TIEN NUOC", 120000), ("CUOC INTERNET VNPT", 200000)]:
            balance -= amt_b
            transactions.append({
                "date": bill_date.isoformat(),
                "description": desc_b,
                "credit": 0, "debit": amt_b, "balance": balance
            })

        # Daily spending (minimal — young freelancer)
        for day_offset in [5, 10, 15, 20, 25]:
            spend_date = month_date + timedelta(days=day_offset)
            items = [
                ("GRAB - DI CHUYEN", 120000),
                ("GD ONLINE - SHOPEE", 280000),
                ("CAFE WORKING SPACE", 150000),
                ("VINMART - SIEU THI", 350000),
                ("GRAB FOOD", 95000),
            ]
            d, a = items[day_offset % len(items)]
            balance -= a
            transactions.append({
                "date": spend_date.isoformat(),
                "description": d, "credit": 0, "debit": a, "balance": balance
            })

    transactions.sort(key=lambda x: x["date"])

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "description", "credit", "debit", "balance"])
        writer.writeheader()
        writer.writerows(transactions)
    print(f"  Created: {path} ({len(transactions)} transactions)")


# ============================================================
# 7. CIC — Thin-file (no records)
# ============================================================
def generate_cic_mock():
    path = os.path.join(CUSTOMER_DIR, "07_cic_api_response.json")

    cic_data = {
        "api_version": "CIC-VN-v2.1",
        "query_timestamp": "2026-03-18T09:15:00+07:00",
        "customer_id": CUSTOMER["cccd"],
        "customer_name": CUSTOMER["ho_ten"],

        "ext_source_scores": {
            "EXT_SOURCE_1": None,
            "EXT_SOURCE_2": None,
            "EXT_SOURCE_3": None,
            "_note": "Thin-file: Không có điểm tín dụng ngoại do chưa có lịch sử vay"
        },

        "credit_inquiry_counts": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
            "AMT_REQ_CREDIT_BUREAU_DAY": 0,
            "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
            "AMT_REQ_CREDIT_BUREAU_MON": 0,
            "AMT_REQ_CREDIT_BUREAU_QRT": 0,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 0
        },

        "social_circle": {
            "OBS_30_CNT_SOCIAL_CIRCLE": 0,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "OBS_60_CNT_SOCIAL_CIRCLE": 0,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0
        },

        "bureau_records": [],

        "thin_file_flag": True,
        "cic_score_equivalent": None,
        "debt_group": None
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cic_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")


# ============================================================
# 8. Internal DB — Empty (no previous loans)
# ============================================================
def generate_internal_db_mock():
    sk_id_curr = 200002

    internal_data = {
        "SK_ID_CURR": sk_id_curr,
        "previous_applications": [],
        "pos_cash_balance": [],
        "installments_payments": [],
        "credit_card_balance": []
    }

    path = os.path.join(CUSTOMER_DIR, "08_internal_db.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(internal_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")
    print(f"    - 0 previous applications (thin-file)")
    print(f"    - 0 installment records")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Generating mock data for: {CUSTOMER['ho_ten']}")
    print(f"  Profile: Thin-file Freelancer")
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
