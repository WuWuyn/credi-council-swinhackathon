"""
Generate mock data for Customer 003: Cửa hàng Hoa Lan (Trần Văn Đức) — Micro SME.

Profile:
- Chủ cửa hàng hoa tươi, 42 tuổi, kinh doanh 3 năm
- Có CIC nhưng ít: 1 khoản vay cũ đã đóng
- Thu nhập từ doanh thu cửa hàng (có xu hướng giảm nhẹ)
- Vay vốn lưu động 200M
- Có nhà riêng thế chấp

Expected: Score ~550-600, Band A/B+, CONDITIONAL (revenue declining)

Usage:
    python data/mock/generate_customer_003.py
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
CUSTOMER_DIR = os.path.join(OUTPUT_DIR, "customer_003")
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
    "id": "CUST_300003",
    "cccd": "038084005432",
    "ho_ten": "Trần Văn Đức",
    "ngay_sinh": "1984-08-12",
    "gioi_tinh": "Nam",
    "quoc_tich": "Việt Nam",
    "dan_toc": "Kinh",
    "ton_giao": "Phật giáo",
    "que_quan": "Xã Bình Minh, Huyện Thanh Oai, Hà Nội",
    "thuong_tru": "Số 78, Đường Lê Duẩn, Phường 1, Quận Gò Vấp, TP Hồ Chí Minh",
    "ngay_cap_cccd": "2020-11-05",
    "noi_cap": "Cục Cảnh sát ĐKQL cư trú và DLQG về dân cư",
    "tinh_trang_hon_nhan": "Đã kết hôn",
    "so_con": 2,
    "so_thanh_vien_gia_dinh": 4,

    # Business info — Hộ kinh doanh cá thể
    "ten_cong_ty": "Cửa hàng Hoa Lan — Hộ kinh doanh cá thể",
    "loai_doanh_nghiep": "Hộ kinh doanh",
    "dia_chi_cty": "Số 78, Đường Lê Duẩn, Phường 1, Quận Gò Vấp, TP Hồ Chí Minh",
    "chuc_vu": "Chủ hộ kinh doanh",
    "phong_ban": "Không",
    "ngay_bat_dau": "2023-04-01",
    "loai_hop_dong": "Giấy phép kinh doanh hộ cá thể",
    "luong_thang": 25000000,  # Average monthly revenue-based income
    "luong_nam": 300000000,
    "phu_cap": 0,
    "bao_hiem": "Tự đóng BHXH tự nguyện, BHYT",
    "sdt_cong_ty": "028 5432 1098",
    "email_cty": "hoalan.flower@gmail.com",
    "ma_so_thue_cty": "0317654321",
    "loai_thu_nhap": "Commercial associate",

    # Housing — Nhà riêng mặt tiền (vừa ở vừa bán hàng)
    "loai_nha": "House / apartment",
    "dia_chi_nha": "Số 78, Đường Lê Duẩn, Phường 1, Quận Gò Vấp, TP Hồ Chí Minh",
    "dien_tich": 55.0,
    "nam_xay": 2000,
    "so_tang_toa_nha": 3,
    "tang_can_ho": 1,
    "co_thang_may": False,
    "vat_lieu_tuong": "Stone, brick",
    "tinh_trang": "Bình thường",
    "gia_tri_bds": 2800000000,  # 2.8 tỉ (nhà mặt tiền quận Gò Vấp)

    # Loan request — Vốn lưu động
    "loai_hop_dong_vay": "Revolving loans",
    "so_tien_vay": 200000000,    # 200M VND
    "ky_han": 12,                 # 12 months (revolving)
    "lai_suat": 0.14,             # 14%/year
    "muc_dich_vay": "Bổ sung vốn lưu động mua hoa nguyên liệu nhập khẩu và mở rộng dịch vụ sự kiện cưới hỏi",
    "tra_hang_thang": 18000000,   # ~18M/month
    "gia_tri_hang_hoa": 200000000,
    "co_xe_oto": False,
    "tuoi_xe": None,
    "co_bat_dong_san": True,
    "nguoi_dong_hanh": "Spouse, partner",
    "trinh_do_hoc_van": "Secondary / secondary special",
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
    y = draw_field(c, y, "Có giá trị đến", "2030-11-05")

    y = draw_section(c, y, "ĐẶC ĐIỂM NHẬN DẠNG")
    y = draw_field(c, y, "Chiều cao", "168 cm")
    y = draw_field(c, y, "Màu mắt", "Đen")

    c.showPage()
    y = draw_header(c, "CĂN CƯỚC CÔNG DÂN - MẶT SAU")
    y = draw_section(c, y, "XÁC NHẬN ĐĂNG KÝ CƯ TRÚ")
    y = draw_field(c, y, "Địa chỉ đăng ký", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2010-05-20")
    y = draw_field(c, y, "Địa chỉ hiện tại", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Địa chỉ kinh doanh", CUSTOMER["dia_chi_cty"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 2. Giấy phép kinh doanh (thay vì HĐLĐ)
# ============================================================
def generate_labor_contract_pdf():
    path = os.path.join(CUSTOMER_DIR, "02_hop_dong_lao_dong.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "GIẤY CHỨNG NHẬN ĐĂNG KÝ HỘ KINH DOANH", "So: 41H8-012345")
    y = draw_section(c, y, "THÔNG TIN HỘ KINH DOANH")
    y = draw_field(c, y, "Tên hộ kinh doanh", "Cửa hàng Hoa Lan")
    y = draw_field(c, y, "Mã số đăng ký", CUSTOMER["ma_so_thue_cty"])
    y = draw_field(c, y, "Ngày đăng ký", CUSTOMER["ngay_bat_dau"])
    y = draw_field(c, y, "Ngành nghề", "Bán buôn hoa tươi, cây cảnh, dịch vụ trang trí sự kiện")
    y = draw_field(c, y, "Mã ngành", "4719 - Bán lẻ khác trong cửa hàng kinh doanh tổng hợp")
    y = draw_field(c, y, "Vốn kinh doanh", "500,000,000 VND")
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_cty"])
    y = draw_field(c, y, "Điện thoại", CUSTOMER["sdt_cong_ty"])

    y = draw_section(c, y, "CHỦ HỘ KINH DOANH")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Địa chỉ thường trú", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Điện thoại", "0903 456 789")
    y = draw_field(c, y, "Email", CUSTOMER["email_cty"])
    y = draw_field(c, y, "Chức danh", CUSTOMER["chuc_vu"])

    y = draw_section(c, y, "THÔNG TIN KINH DOANH BỔ SUNG")
    y = draw_field(c, y, "Số nhân viên", "3 người (2 nhân viên + 1 chủ)")
    y = draw_field(c, y, "Diện tích kinh doanh", "55 m2 (tầng 1 nhà riêng)")
    y = draw_field(c, y, "Thời gian hoạt động", "Từ 04/2023 đến nay (~3 năm)")

    # Page 2 — Financial info
    c.showPage()
    y = draw_header(c, "GIẤY CHỨNG NHẬN ĐĂNG KÝ HỘ KINH DOANH (tiếp)")

    y = draw_section(c, y, "BÁO CÁO TÀI CHÍNH ĐƠN GIẢN (2025)")
    y = draw_field(c, y, "Doanh thu năm 2025", "480,000,000 VND")
    y = draw_field(c, y, "Doanh thu năm 2024", "540,000,000 VND")
    y = draw_field(c, y, "Xu hướng doanh thu", "Giảm 11% so với năm trước")
    y = draw_field(c, y, "Chi phí hàng hóa (COGS)", "320,000,000 VND/năm")
    y = draw_field(c, y, "Chi phí nhân viên", "72,000,000 VND/năm (2 NV x 3M/tháng)")
    y = draw_field(c, y, "Chi phí thuê/tiện ích", "36,000,000 VND/năm")
    y = draw_field(c, y, "Lợi nhuận ước tính", "52,000,000 VND/năm")
    y = draw_field(c, y, "Thu nhập chủ HKD/tháng", f"{CUSTOMER['luong_thang']:,.0f} VND (bao gồm lợi nhuận + lương)")

    y = draw_section(c, y, "NGUỒN CUNG CẤP HÀNG HÓA")
    items = [
        "- Đà Lạt Flower Farm (hoa nội địa) — 60% nguồn hàng",
        "- Nhập khẩu hoa Hà Lan, Ecuador qua Cty XNK Hoa Việt — 25%",
        "- Chợ đầu mối Hóc Môn — 15% (hoa lẻ, cây cảnh)",
    ]
    c.setFont(FONT, 9)
    for item in items:
        y = draw_field(c, y, "", item)

    y = draw_section(c, y, "DỊCH VỤ")
    services = [
        "- Bán lẻ hoa tươi các loại",
        "- Dịch vụ trang trí hoa sự kiện cưới hỏi, hội nghị",
        "- Cho thuê cây cảnh văn phòng",
        "- Đặt hoa online qua Facebook/Zalo",
    ]
    for s in services:
        y = draw_field(c, y, "", s)

    y = draw_section(c, y, "GHI CHÚ")
    y = draw_field(c, y, "Bảo hiểm", CUSTOMER["bao_hiem"])
    y = draw_field(c, y, "Thuế", "Kê khai thuế GTGT theo quý, thuế suất 1%")

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "Cơ quan cấp: UBND Quận Gò Vấp")
    y -= 0.5*cm
    c.drawString(3*cm, y, f"Ngày cấp: {CUSTOMER['ngay_bat_dau']}")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 3. Sổ hộ khẩu
# ============================================================
def generate_household_pdf():
    path = os.path.join(CUSTOMER_DIR, "03_so_ho_khau.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "SỔ HỘ KHẨU", "So: HK-2010-005432")
    y = draw_section(c, y, "THÔNG TIN CHỦ HỘ")
    y = draw_field(c, y, "Họ và tên chủ hộ", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ thường trú", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2010-05-20")

    y = draw_section(c, y, "THÀNH VIÊN TRONG HỘ")
    # Wife
    y = draw_field(c, y, "1. Họ tên", "Nguyễn Thị Hồng")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Vợ")
    y = draw_field(c, y, "   Ngày sinh", "1986-02-14")
    y = draw_field(c, y, "   Số CCCD", "038086007890")
    y = draw_field(c, y, "   Nghề nghiệp", "Phụ giúp cửa hàng Hoa Lan")

    # Child 1
    y -= 0.3*cm
    y = draw_field(c, y, "2. Họ tên", "Trần Minh Khôi")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Con trai")
    y = draw_field(c, y, "   Ngày sinh", "2012-09-15")
    y = draw_field(c, y, "   Nghề nghiệp", "Học sinh lớp 8")

    # Child 2
    y -= 0.3*cm
    y = draw_field(c, y, "3. Họ tên", "Trần Ngọc Mai")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Con gái")
    y = draw_field(c, y, "   Ngày sinh", "2016-03-28")
    y = draw_field(c, y, "   Nghề nghiệp", "Học sinh lớp 4")

    c.showPage()
    y = draw_header(c, "SỔ HỘ KHẨU (tiếp theo)")
    y = draw_section(c, y, "TỔNG HỢP")
    y = draw_field(c, y, "Tổng số nhân khẩu", "4 người")
    y = draw_field(c, y, "Chủ hộ", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Số con", str(CUSTOMER["so_con"]))

    y = draw_section(c, y, "XÁC NHẬN")
    y = draw_field(c, y, "Ngày xác nhận", "2024-09-10")
    y = draw_field(c, y, "Cơ quan", "Công an Phường 1, Quận Gò Vấp, TP.HCM")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 4. Thẩm định nhà ở (nhà mặt tiền)
# ============================================================
def generate_housing_pdf():
    path = os.path.join(CUSTOMER_DIR, "04_tham_dinh_nha_o.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "PHIẾU THẨM ĐỊNH TÀI SẢN / NHÀ Ở", "Ma ho so: TD-2026-SME003")
    y = draw_section(c, y, "1. THÔNG TIN CHỦ SỞ HỮU")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])

    y = draw_section(c, y, "2. THÔNG TIN BẤT ĐỘNG SẢN")
    y = draw_field(c, y, "Địa chỉ BĐS", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Loại hình", "Nhà phố mặt tiền (vừa ở vừa kinh doanh)")
    y = draw_field(c, y, "Diện tích đất", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Diện tích xây dựng", "165 m2 (3 tầng)")
    y = draw_field(c, y, "Năm xây dựng", str(CUSTOMER["nam_xay"]))
    y = draw_field(c, y, "Số tầng", str(CUSTOMER["so_tang_toa_nha"]))
    y = draw_field(c, y, "Có thang máy", "Không")
    y = draw_field(c, y, "Vật liệu tường", CUSTOMER["vat_lieu_tuong"])
    y = draw_field(c, y, "Tình trạng", CUSTOMER["tinh_trang"])
    y = draw_field(c, y, "Mô tả", "Tầng 1: cửa hàng hoa. Tầng 2-3: ở")

    y = draw_section(c, y, "3. THÔNG SỐ KỸ THUẬT")
    y = draw_field(c, y, "Chất lượng (1-10)", "6.0 / 10")
    y = draw_field(c, y, "Diện tích sinh hoạt", "110 m2 (tầng 2-3)")
    y = draw_field(c, y, "Số lối vào", "2 (mặt tiền + sau)")

    c.showPage()
    y = draw_header(c, "PHIẾU THẨM ĐỊNH (tiếp theo)")
    y = draw_section(c, y, "4. THÔNG TIN KHU VỰC")
    y = draw_field(c, y, "Quận/Huyện", "Gò Vấp, TP.HCM")
    y = draw_field(c, y, "Mật độ dân số (tương đối)", "0.038 (khu dân cư)")
    y = draw_field(c, y, "Xếp hạng khu vực", "2 (Tốt)")
    y = draw_field(c, y, "Xếp hạng khu vực (TP)", "2 (Tốt)")
    y = draw_field(c, y, "Đăng ký cùng vùng sống", "Có")
    y = draw_field(c, y, "Đăng ký cùng TP làm việc", "Có")
    y = draw_field(c, y, "Sống cùng vùng làm việc", "Có")

    y = draw_section(c, y, "5. ĐÁNH GIÁ GIÁ TRỊ")
    y = draw_field(c, y, "Giá trị ước tính", f"{CUSTOMER['gia_tri_bds']:,.0f} VND")
    y = draw_field(c, y, "Giá thị trường", "2,500,000,000 - 3,200,000,000 VND")
    y = draw_field(c, y, "Phương pháp", "So sánh thị trường")
    y = draw_field(c, y, "Ngày định giá", "2026-03-05")

    y = draw_section(c, y, "6. TÌNH TRẠNG PHÁP LÝ")
    y = draw_field(c, y, "Sổ hồng/sổ đỏ", "Có - Số GCN.QSDD-GV.005432")
    y = draw_field(c, y, "Trạng thái", "Đã cấp giấy CN QSDĐ và sở hữu nhà ở")
    y = draw_field(c, y, "Tranh chấp", "Không")
    y = draw_field(c, y, "Thế chấp hiện tại", "Không")

    y = draw_section(c, y, "7. KẾT LUẬN")
    y = draw_field(c, y, "Kết luận", "Đủ điều kiện thế chấp")
    y = draw_field(c, y, "Giá trị cho vay tối đa", "1,960,000,000 VND (70% gia tri)")
    y = draw_field(c, y, "Ghi chú", "Nhà mặt tiền, thanh khoản khá. Đã cũ (2000), cần xem xét chi phí sửa chữa")

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "Nhân viên thẩm định: Nguyễn Hoàng Phúc")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 5. Đơn vay vốn lưu động
# ============================================================
def generate_loan_application_pdf():
    path = os.path.join(CUSTOMER_DIR, "05_don_vay.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "ĐƠN ĐỀ NGHỊ VAY VỐN", "Ma don: DV-2026-SME003")
    y = draw_section(c, y, "I. THÔNG TIN NGƯỜI VAY")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Trình độ học vấn", CUSTOMER["trinh_do_hoc_van"])
    y = draw_field(c, y, "Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "SDT", "0903 456 789")
    y = draw_field(c, y, "Email", CUSTOMER["email_cty"])
    y = draw_field(c, y, "Có xe ô tô", "Không")
    y = draw_field(c, y, "Có bất động sản", "Có")
    y = draw_field(c, y, "Người đồng hành", CUSTOMER["nguoi_dong_hanh"])
    y = draw_field(c, y, "Số di động liên lạc được", "Có")
    y = draw_field(c, y, "Số điện thoại bàn", "Có")
    y = draw_field(c, y, "Email", "Có")
    y = draw_field(c, y, "Ngày đổi SĐT gần nhất", "2023-01-10 (khoảng 800 ngày trước)")

    y = draw_section(c, y, "II. THÔNG TIN KHOẢN VAY")
    y = draw_field(c, y, "Loại hợp đồng", CUSTOMER["loai_hop_dong_vay"])
    y = draw_field(c, y, "Số tiền vay", f"{CUSTOMER['so_tien_vay']:,.0f} VND")
    y = draw_field(c, y, "Kỳ hạn", f"{CUSTOMER['ky_han']} thang")
    y = draw_field(c, y, "Trả hàng tháng (dự kiến)", f"{CUSTOMER['tra_hang_thang']:,.0f} VND")

    c.showPage()
    y = draw_header(c, "ĐƠN ĐỀ NGHỊ VAY VỐN (tiếp theo)")
    y = draw_section(c, y, "III. MỤC ĐÍCH VAY")
    y = draw_field(c, y, "Mục đích", CUSTOMER["muc_dich_vay"])
    y = draw_field(c, y, "Chi tiết", "Nhập hoa nhập khẩu (Holland, Ecuador) cho mùa cưới T5-T10/2026")
    y = draw_field(c, y, "", "Mua thêm vật tư trang trí sự kiện và xe tải nhỏ vận chuyển")

    y = draw_section(c, y, "IV. TÀI SẢN THẾ CHẤP")
    y = draw_field(c, y, "Loại TSĐB", "Nhà phố mặt tiền")
    y = draw_field(c, y, "Giá trị BĐS", f"{CUSTOMER['gia_tri_bds']:,.0f} VND")
    y = draw_field(c, y, "LTV dự kiến", f"{CUSTOMER['so_tien_vay']/CUSTOMER['gia_tri_bds']*100:.1f}%")

    y = draw_section(c, y, "V. NGUỒN TRẢ NỢ")
    y = draw_field(c, y, "Thu nhập (TB tháng)", f"{CUSTOMER['luong_thang']:,.0f} VND")
    y = draw_field(c, y, "Nguồn", "Doanh thu cửa hàng Hoa Lan")
    y = draw_field(c, y, "Chi phí sinh hoạt", "12,000,000 VND/thang")
    y = draw_field(c, y, "Chi phí kinh doanh", "Đã trừ trong thu nhập ròng")
    dti = CUSTOMER["tra_hang_thang"] / CUSTOMER["luong_thang"] * 100
    y = draw_field(c, y, "Tỷ lệ nợ/thu nhập (DTI)", f"{dti:.1f}%")

    y = draw_section(c, y, "VI. CAM KẾT")
    c.setFont(FONT, 9)
    for txt in [
        "- Tôi cam kết các thông tin trên là đúng sự thật.",
        "- Tôi đồng ý để ngân hàng xác minh thông tin.",
        "- Tôi cam kết sử dụng vốn vay đúng mục đích kinh doanh.",
    ]:
        y = draw_field(c, y, "", txt)

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "TP.HCM, ngay 10 thang 03 nam 2026")
    y -= 1.5*cm
    c.drawString(3*cm, y, CUSTOMER["ho_ten"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 6. Bank Statement — Business income, declining trend
# ============================================================
def generate_bank_statement():
    path = os.path.join(CUSTOMER_DIR, "06_sao_ke_ngan_hang.csv")
    transactions = []
    base_date = date(2025, 9, 1)
    balance = 28000000
    random.seed(303)

    # Revenue declining over 6 months (seasonal + competition)
    monthly_revenues = [45000000, 42000000, 38000000, 35000000, 33000000, 30000000]

    for month in range(6):
        month_date = base_date + timedelta(days=30 * month)
        revenue = monthly_revenues[month]

        # Multiple revenue transactions per month (retail business)
        # Split into 8-12 smaller transactions
        n_sales = random.randint(8, 12)
        per_sale = revenue // n_sales
        for sale_idx in range(n_sales):
            sale_date = month_date + timedelta(days=random.randint(1, 28))
            amt = per_sale + random.randint(-500000, 500000)
            balance += amt
            descs = [
                "BAN HOA TUOI - KHACH LE",
                "DICH VU TRANG TRI SU KIEN",
                "BAN HOA ONLINE - ZALO/FB",
                "BAN CHAU CAY CANH",
                "DICH VU HOA CUOI HOI",
                "BAN HOA SI - CONG TY",
            ]
            transactions.append({
                "date": sale_date.isoformat(),
                "description": descs[sale_idx % len(descs)],
                "credit": max(0, amt), "debit": 0, "balance": balance
            })

        # COGS — buying flowers/materials (~60% of revenue)
        for purchase_day in [3, 10, 17, 24]:
            pdate = month_date + timedelta(days=purchase_day)
            cost = revenue * 0.15 + random.randint(-300000, 300000)
            balance -= int(cost)
            purchase_descs = [
                "MUA HOA DA LAT - FARM",
                "NHAP HOA NHAP KHAU - HOA VIET",
                "MUA VAT TU TRANG TRI",
                "CHO DAU MOI HOC MON - HOA LE",
            ]
            transactions.append({
                "date": pdate.isoformat(),
                "description": purchase_descs[purchase_day % 4],
                "credit": 0, "debit": int(cost), "balance": balance
            })

        # Employee wages — 5th
        wage_date = month_date.replace(day=5)
        balance -= 6000000
        transactions.append({
            "date": wage_date.isoformat(),
            "description": "LUONG NHAN VIEN T" + str(month_date.month) + " (2 nguoi)",
            "credit": 0, "debit": 6000000, "balance": balance
        })

        # Bills
        bill_date = month_date + timedelta(days=14)
        for desc_b, amt_b in [("TIEN DIEN EVN HCMC", 1200000), ("TIEN NUOC SAWACO", 280000), ("CUOC INTERNET VNPT", 250000)]:
            balance -= amt_b
            transactions.append({
                "date": bill_date.isoformat(),
                "description": desc_b,
                "credit": 0, "debit": amt_b, "balance": balance
            })

        # Personal living expenses
        for day_off in [7, 14, 21, 28]:
            sp_date = month_date + timedelta(days=day_off)
            items = [
                ("COOPMART - SIEU THI", 650000),
                ("HOC PHI CON - TRUONG TH", 2000000),
                ("GRAB - DI CHUYEN", 180000),
                ("TIEN AN TRUA - GIA DINH", 400000),
            ]
            d, a = items[day_off // 7 - 1] if day_off // 7 <= 4 else items[0]
            balance -= a
            transactions.append({
                "date": sp_date.isoformat(),
                "description": d, "credit": 0, "debit": a, "balance": balance
            })

    transactions.sort(key=lambda x: x["date"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "description", "credit", "debit", "balance"])
        writer.writeheader()
        writer.writerows(transactions)
    print(f"  Created: {path} ({len(transactions)} transactions)")


# ============================================================
# 7. CIC — Light history (1 closed loan, paid on time)
# ============================================================
def generate_cic_mock():
    path = os.path.join(CUSTOMER_DIR, "07_cic_api_response.json")

    def make_monthly_status(n_months, credit_active):
        statuses = []
        for m in range(n_months):
            if credit_active == "Closed" and m < 3:
                s = "C"
            elif m < 24:
                s = "0"
            else:
                s = "X"
            statuses.append({"MONTHS_BALANCE": -m, "STATUS": s})
        return statuses

    cic_data = {
        "api_version": "CIC-VN-v2.1",
        "query_timestamp": "2026-03-10T14:30:00+07:00",
        "customer_id": CUSTOMER["cccd"],
        "customer_name": CUSTOMER["ho_ten"],

        "ext_source_scores": {
            "EXT_SOURCE_1": 0.380,
            "EXT_SOURCE_2": 0.420,
            "EXT_SOURCE_3": 0.350,
            "_note": "Scores lower — limited credit history, SME"
        },

        "credit_inquiry_counts": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
            "AMT_REQ_CREDIT_BUREAU_DAY": 0,
            "AMT_REQ_CREDIT_BUREAU_WEEK": 1,
            "AMT_REQ_CREDIT_BUREAU_MON": 1,
            "AMT_REQ_CREDIT_BUREAU_QRT": 2,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 3
        },

        "social_circle": {
            "OBS_30_CNT_SOCIAL_CIRCLE": 1,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "OBS_60_CNT_SOCIAL_CIRCLE": 1,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0
        },

        "bureau_records": [
            {
                "SK_ID_BUREAU": 5003001,
                "CREDIT_ACTIVE": "Closed",
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -900,
                "CREDIT_DAY_OVERDUE": 0,
                "DAYS_CREDIT_ENDDATE": -100,
                "DAYS_ENDDATE_FACT": -110,
                "AMT_CREDIT_MAX_OVERDUE": 0,
                "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": 100000000,
                "AMT_CREDIT_SUM_DEBT": 0,
                "AMT_CREDIT_SUM_LIMIT": 0,
                "AMT_CREDIT_SUM_OVERDUE": 0,
                "CREDIT_TYPE": "Consumer credit",
                "DAYS_CREDIT_UPDATE": -100,
                "AMT_ANNUITY": 4500000,
                "monthly_status": make_monthly_status(30, "Closed")
            }
        ],

        "thin_file_flag": False,
        "cic_score_equivalent": 580,
        "debt_group": 1
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cic_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")


# ============================================================
# 8. Internal DB — 1 previous app (approved, completed)
# ============================================================
def generate_internal_db_mock():
    sk_id_curr = 300003

    prev_apps = [{
        "SK_ID_PREV": 3001001,
        "SK_ID_CURR": sk_id_curr,
        "NAME_CONTRACT_TYPE": "Consumer loans",
        "AMT_ANNUITY": 4500000,
        "AMT_APPLICATION": 100000000,
        "AMT_CREDIT": 100000000,
        "AMT_DOWN_PAYMENT": 0,
        "AMT_GOODS_PRICE": 100000000,
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "HOUR_APPR_PROCESS_START": 9,
        "FLAG_LAST_APPL_PER_CONTRACT": "Y",
        "NFLAG_LAST_APPL_IN_DAY": 1,
        "RATE_DOWN_PAYMENT": 0.0,
        "RATE_INTEREST_PRIMARY": 0.13,
        "RATE_INTEREST_PRIVILEGED": 0.11,
        "NAME_CASH_LOAN_PURPOSE": "Repairs",
        "NAME_CONTRACT_STATUS": "Approved",
        "DAYS_DECISION": -900,
        "NAME_PAYMENT_TYPE": "Cash through the bank",
        "CODE_REJECT_REASON": "XAP",
        "NAME_TYPE_SUITE": "Spouse, partner",
        "NAME_CLIENT_TYPE": "New",
        "NAME_GOODS_CATEGORY": "XNA",
        "NAME_PORTFOLIO": "Cash",
        "NAME_PRODUCT_TYPE": "walk-in",
        "CHANNEL_TYPE": "Credit and cash offices",
        "SELLERPLACE_AREA": -1,
        "NAME_SELLER_INDUSTRY": "XNA",
        "CNT_PAYMENT": 24,
        "NAME_YIELD_GROUP": "middle",
        "PRODUCT_COMBINATION": "Cash",
        "DAYS_FIRST_DRAWING": -890,
        "DAYS_FIRST_DUE": -860,
        "DAYS_LAST_DUE_1ST_VERSION": -140,
        "DAYS_LAST_DUE": -140,
        "DAYS_TERMINATION": -110,
        "NFLAG_INSURED_ON_APPROVAL": 0
    }]

    # Installments — all paid on time
    installments = []
    for i in range(1, 25):
        days_inst = -900 + 30 * i
        installments.append({
            "SK_ID_PREV": 3001001,
            "SK_ID_CURR": sk_id_curr,
            "NUM_INSTALMENT_VERSION": 1,
            "NUM_INSTALMENT_NUMBER": i,
            "DAYS_INSTALMENT": days_inst,
            "DAYS_ENTRY_PAYMENT": days_inst - 1,
            "AMT_INSTALMENT": 4500000,
            "AMT_PAYMENT": 4500000
        })

    # POS Cash
    pos_cash = []
    for m in range(24):
        pos_cash.append({
            "SK_ID_PREV": 3001001,
            "SK_ID_CURR": sk_id_curr,
            "MONTHS_BALANCE": -m,
            "CNT_INSTALMENT": 24,
            "CNT_INSTALMENT_FUTURE": max(0, 24 - m),
            "NAME_CONTRACT_STATUS": "Completed" if m < 2 else "Active",
            "SK_DPD": 0,
            "SK_DPD_DEF": 0
        })

    internal_data = {
        "SK_ID_CURR": sk_id_curr,
        "previous_applications": prev_apps,
        "pos_cash_balance": pos_cash,
        "installments_payments": installments,
        "credit_card_balance": []
    }

    path = os.path.join(CUSTOMER_DIR, "08_internal_db.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(internal_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")
    print(f"    - 1 previous application (Completed)")
    print(f"    - {len(installments)} installment records (all on time)")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Generating mock data for: {CUSTOMER['ho_ten']}")
    print(f"  Profile: Micro SME — Cửa hàng Hoa Lan")
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
