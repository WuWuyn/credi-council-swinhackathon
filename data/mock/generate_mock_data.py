"""
Generate realistic demo mock data for 1 customer: Nguyen Van Minh.
Creates PDFs (CCCD, labor contract, household, housing, loan app) + CIC JSON + Internal DB JSON.

Usage:
    python data/mock/generate_mock_data.py
"""
import json, os, csv
from datetime import date, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMER_DIR = os.path.join(OUTPUT_DIR, "customer_001")
os.makedirs(CUSTOMER_DIR, exist_ok=True)

# ============================================================
# Registering a Unicode font for Vietnamese text
# ============================================================
# Try common system fonts that support Vietnamese
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

if not _font_registered:
    print("WARNING: No Vietnamese-capable font found. PDFs will use Helvetica (no diacritics).")
    FONT = "Helvetica"
else:
    FONT = "VNFont"


# ============================================================
# Customer Profile — Nguyen Van Minh
# ============================================================
CUSTOMER = {
    "id": "CUST_100002",
    "cccd": "079099001234",
    "ho_ten": "Nguyễn Văn Minh",
    "ngay_sinh": "1988-05-15",
    "gioi_tinh": "Nam",
    "quoc_tich": "Việt Nam",
    "dan_toc": "Kinh",
    "ton_giao": "Không",
    "que_quan": "Xã Bình Hòa, Huyện Thuận An, Tỉnh Bình Dương",
    "thuong_tru": "Số 45, Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP Hồ Chí Minh",
    "ngay_cap_cccd": "2021-03-20",
    "noi_cap": "Cục Cảnh sát ĐKQL cư trú và DLQG về dân cư",
    "tinh_trang_hon_nhan": "Đã kết hôn",
    "so_con": 1,
    "so_thanh_vien_gia_dinh": 3,

    # Employment
    "ten_cong_ty": "Công ty TNHH Công nghệ ABC Việt Nam",
    "loai_doanh_nghiep": "Business Entity Type",
    "dia_chi_cty": "Tầng 15, Tòa nhà Landmark 81, 720A Điện Biên Phủ, P.22, Q.Bình Thạnh, TP.HCM",
    "chuc_vu": "Senior Software Engineer",
    "phong_ban": "Phòng Phát triển Sản phẩm",
    "ngay_bat_dau": "2019-06-01",
    "loai_hop_dong": "Không xác định thời hạn",
    "luong_thang": 35000000,  # 35M VND/month
    "luong_nam": 420000000,   # 420M VND/year
    "phu_cap": 5000000,
    "bao_hiem": "BHXH, BHYT, BHTN đầy đủ",
    "sdt_cong_ty": "028 1234 5678",
    "email_cty": "minh.nguyenvan@abctech.vn",
    "ma_so_thue_cty": "0316789012",
    "loai_thu_nhap": "Working",

    # Housing
    "loai_nha": "Block of flats",
    "dia_chi_nha": "Căn hộ 12A05, Chung cư Vinhomes Central Park, 208 Nguyễn Hữu Cảnh, P.22, Q.Bình Thạnh, TP.HCM",
    "dien_tich": 72.5,  # m2
    "nam_xay": 2018,
    "so_tang_toa_nha": 40,
    "tang_can_ho": 12,
    "co_thang_may": True,
    "vat_lieu_tuong": "Panel",
    "tinh_trang": "Bình thường (không khẩn cấp)",
    "gia_tri_bds": 4500000000,  # 4.5B VND

    # Loan request
    "loai_hop_dong_vay": "Cash loans",
    "so_tien_vay": 300000000,    # 300M VND
    "ky_han": 36,                # 36 months
    "lai_suat": 0.12,            # 12%/year
    "muc_dich_vay": "Mua xe ô tô phục vụ đi lại và công việc",
    "tra_hang_thang": 9964000,   # ~10M VND/month
    "gia_tri_hang_hoa": 350000000,  # 350M VND (car value)
    "co_xe_oto": True,
    "tuoi_xe": 3,
    "co_bat_dong_san": True,
    "nguoi_dong_hanh": "Spouse, partner",
    "trinh_do_hoc_van": "Higher education",
}


def draw_header(c, title, subtitle=""):
    """Draw a professional header for each PDF page."""
    c.setFont(FONT, 14)
    c.drawCentredString(A4[0]/2, A4[1] - 2*cm, title)
    if subtitle:
        c.setFont(FONT, 10)
        c.drawCentredString(A4[0]/2, A4[1] - 2.8*cm, subtitle)
    c.line(2*cm, A4[1] - 3.2*cm, A4[0] - 2*cm, A4[1] - 3.2*cm)
    return A4[1] - 4*cm  # return y position after header


def draw_field(c, y, label, value, x=2.5*cm, label_width=6*cm):
    """Draw a label: value pair."""
    c.setFont(FONT, 10)
    c.drawString(x, y, f"{label}:")
    c.setFont(FONT, 10)
    c.drawString(x + label_width, y, str(value))
    return y - 0.6*cm


def draw_section(c, y, title):
    """Draw a section title."""
    y -= 0.3*cm
    c.setFont(FONT, 11)
    c.setFillColor(colors.HexColor("#1a5276"))
    c.drawString(2.2*cm, y, title)
    c.setFillColor(colors.black)
    c.line(2.2*cm, y - 0.15*cm, A4[0] - 2*cm, y - 0.15*cm)
    return y - 0.7*cm


# ============================================================
# 1. CCCD PDF (2 pages)
# ============================================================
def generate_cccd_pdf():
    path = os.path.join(CUSTOMER_DIR, "01_cccd.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    # Page 1 — Front
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
    y = draw_field(c, y, "Có giá trị đến", "2031-03-20")

    y = draw_section(c, y, "ĐẶC ĐIỂM NHẬN DẠNG")
    y = draw_field(c, y, "Chiều cao", "172 cm")
    y = draw_field(c, y, "Màu mắt", "Đen")
    y = draw_field(c, y, "Đặc điểm nhận dạng", "Không có đặc điểm nổi bật")

    # Page 2 — Back / additional info
    c.showPage()
    y = draw_header(c, "CĂN CƯỚC CÔNG DÂN - MẶT SAU")
    y = draw_section(c, y, "MÃ VẠCH VÀ CHIP")
    y = draw_field(c, y, "Mã QR", "[QR Code - encoded personal info]")
    y = draw_field(c, y, "Chip NFC", "Có - Chưa kết nối VNEID")

    y = draw_section(c, y, "LỊCH SỬ CẤP ĐỔI")
    y = draw_field(c, y, "CMND cũ", "079088654321")
    y = draw_field(c, y, "Ngày cấp CMND", "2010-08-12")
    y = draw_field(c, y, "Lý do đổi", "Chuyển đổi từ CMND sang CCCD gắn chip")

    y = draw_section(c, y, "XÁC NHẬN ĐĂNG KÝ CƯ TRÚ")
    y = draw_field(c, y, "Địa chỉ đăng ký", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2015-07-20")
    y = draw_field(c, y, "Địa chỉ hiện tại", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Địa chỉ làm việc", CUSTOMER["dia_chi_cty"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 2. Labor Contract PDF (4 pages)
# ============================================================
def generate_labor_contract_pdf():
    path = os.path.join(CUSTOMER_DIR, "02_hop_dong_lao_dong.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    # Page 1 — Header
    y = draw_header(c, "HỢP ĐỒNG LAO ĐỘNG", f"So: HDLD-2019-{CUSTOMER['cccd'][-4:]}")
    y = draw_section(c, y, "BÊN SỬ DỤNG LAO ĐỘNG (BÊN A)")
    y = draw_field(c, y, "Tên doanh nghiệp", CUSTOMER["ten_cong_ty"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_cty"])
    y = draw_field(c, y, "Mã số thuế", CUSTOMER["ma_so_thue_cty"])
    y = draw_field(c, y, "Điện thoại", CUSTOMER["sdt_cong_ty"])
    y = draw_field(c, y, "Đại diện", "Ông Trần Quốc Hưng - Giám đốc")
    y = draw_field(c, y, "Chức vụ", "Giám đốc điều hành")

    y = draw_section(c, y, "NGƯỜI LAO ĐỘNG (BÊN B)")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Điện thoại", "0901 234 567")
    y = draw_field(c, y, "Email", CUSTOMER["email_cty"])

    y = draw_section(c, y, "ĐIỀU 1: CÔNG VIỆC VÀ ĐỊA ĐIỂM")
    y = draw_field(c, y, "Chức danh", CUSTOMER["chuc_vu"])
    y = draw_field(c, y, "Phòng ban", CUSTOMER["phong_ban"])
    y = draw_field(c, y, "Địa điểm làm việc", CUSTOMER["dia_chi_cty"])
    y = draw_field(c, y, "Ngày bắt đầu", CUSTOMER["ngay_bat_dau"])
    y = draw_field(c, y, "Loại hợp đồng", CUSTOMER["loai_hop_dong"])

    # Page 2 — Salary details
    c.showPage()
    y = draw_header(c, "HỢP ĐỒNG LAO ĐỘNG (tiếp theo)")
    y = draw_section(c, y, "ĐIỀU 2: LƯƠNG VÀ CHẾ ĐỘ")
    y = draw_field(c, y, "Lương cơ bản", f"{CUSTOMER['luong_thang']:,.0f} VND/thang")
    y = draw_field(c, y, "Phụ cấp", f"{CUSTOMER['phu_cap']:,.0f} VND/thang")
    y = draw_field(c, y, "Tổng thu nhập", f"{CUSTOMER['luong_thang'] + CUSTOMER['phu_cap']:,.0f} VND/thang")
    y = draw_field(c, y, "Thu nhập năm", f"{CUSTOMER['luong_nam']:,.0f} VND/nam")
    y = draw_field(c, y, "Hình thức trả", "Chuyển khoản qua ngân hàng")
    y = draw_field(c, y, "Ngân hàng nhận lương", "Vietcombank - CN Hồ Chí Minh")
    y = draw_field(c, y, "Số tài khoản", "0071001234567")
    y = draw_field(c, y, "Ngày trả lương", "Mùng 5 hàng tháng")

    y = draw_section(c, y, "ĐIỀU 3: THỜI GIAN LÀM VIỆC")
    y = draw_field(c, y, "Giờ làm việc", "8h00 - 17h30, Thứ 2 - Thứ 6")
    y = draw_field(c, y, "Nghỉ trưa", "12h00 - 13h30")
    y = draw_field(c, y, "Ngày nghỉ phép", "12 ngày/năm (theo thâm niên)")

    y = draw_section(c, y, "ĐIỀU 4: BẢO HIỂM XÃ HỘI")
    y = draw_field(c, y, "BHXH", "Đóng đầy đủ (8% NLĐ + 17.5% NSDLĐ)")
    y = draw_field(c, y, "BHYT", "Đóng đầy đủ (1.5% NLĐ + 3% NSDLĐ)")
    y = draw_field(c, y, "BHTN", "Đóng đầy đủ (1% NLĐ + 1% NSDLĐ)")
    y = draw_field(c, y, "Số BHXH", "7901234567")

    # Page 3 — Terms and conditions
    c.showPage()
    y = draw_header(c, "HỢP ĐỒNG LAO ĐỘNG (tiếp theo)")
    y = draw_section(c, y, "ĐIỀU 5: QUYỀN VÀ NGHĨA VỤ CỦA BÊN A")
    c.setFont(FONT, 9)
    duties = [
        "- Bảo đảm điều kiện làm việc an toàn, vệ sinh lao động.",
        "- Trả lương đầy đủ, đúng hạn theo quy định.",
        "- Thực hiện đầy đủ các chế độ BHXH, BHYT, BHTN cho người lao động.",
        "- Tạo điều kiện để người lao động nâng cao trình độ chuyên môn.",
        "- Không được phân biệt đối xử, cưỡng bức lao động.",
    ]
    for d in duties:
        y = draw_field(c, y, "", d)

    y = draw_section(c, y, "ĐIỀU 6: QUYỀN VÀ NGHĨA VỤ CỦA BÊN B")
    rights = [
        "- Hoàn thành công việc theo hợp đồng lao động.",
        "- Chấp hành lệnh điều hành sản xuất kinh doanh hợp pháp.",
        "- Bảo mật thông tin kỹ thuật, kinh doanh của công ty.",
        "- Được hưởng đầy đủ lương, thưởng và các chế độ phúc lợi.",
        "- Được tham gia đào tạo nâng cao trình độ nghiệp vụ.",
    ]
    for r in rights:
        y = draw_field(c, y, "", r)

    y = draw_section(c, y, "ĐIỀU 7: CHẤM DỨT HỢP ĐỒNG")
    c.setFont(FONT, 9)
    terms = [
        "- Hai bên có thể chấm dứt hợp đồng theo quy định của Bộ luật Lao động.",
        "- Thời hạn báo trước: 45 ngày đối với hợp đồng không xác định thời hạn.",
        "- Trợ cấp thôi việc: 0.5 tháng lương/năm làm việc.",
    ]
    for t in terms:
        y = draw_field(c, y, "", t)

    # Page 4 — Signatures
    c.showPage()
    y = draw_header(c, "HỢP ĐỒNG LAO ĐỘNG (Trang cuối)")
    y = draw_section(c, y, "ĐIỀU 8: ĐIỀU KHOẢN THI HÀNH")
    c.setFont(FONT, 9)
    clauses = [
        "- Hợp đồng này có hiệu lực từ ngày ký.",
        "- Hợp đồng được lập thành 02 bản, mỗi bên giữ 01 bản có giá trị pháp lý như nhau.",
        "- Mọi tranh chấp phát sinh từ hợp đồng này được giải quyết theo Bộ luật Lao động.",
        "- Trong quá trình thực hiện, nếu cần bổ sung hoặc sửa đổi, hai bên sẽ thỏa thuận",
        "  bằng văn bản và lập phụ lục hợp đồng.",
    ]
    for cl in clauses:
        y = draw_field(c, y, "", cl)

    y -= 2*cm
    c.setFont(FONT, 11)
    c.drawString(3*cm, y, "ĐẠI DIỆN BÊN A")
    c.drawString(12*cm, y, "BÊN B")
    y -= 0.6*cm
    c.setFont(FONT, 9)
    c.drawString(3*cm, y, "(Ký, ghi rõ họ tên, đóng dấu)")
    c.drawString(12*cm, y, "(Ký, ghi rõ họ tên)")
    y -= 2*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "Trần Quốc Hưng")
    c.drawString(12*cm, y, CUSTOMER["ho_ten"])
    y -= 0.5*cm
    c.drawString(3*cm, y, "Giám đốc điều hành")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 3. Household Registration PDF (2 pages)
# ============================================================
def generate_household_pdf():
    path = os.path.join(CUSTOMER_DIR, "03_so_ho_khau.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "SỔ HỘ KHẨU", "So: HK-2015-001234")
    y = draw_section(c, y, "THÔNG TIN CHỦ HỘ")
    y = draw_field(c, y, "Họ và tên chủ hộ", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ thường trú", CUSTOMER["thuong_tru"])
    y = draw_field(c, y, "Ngày đăng ký", "2015-07-20")

    y = draw_section(c, y, "THÀNH VIÊN TRONG HỘ")

    # Member 1: Wife
    y = draw_field(c, y, "1. Họ tên", "Lê Thị Hương")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Vợ")
    y = draw_field(c, y, "   Ngày sinh", "1990-12-08")
    y = draw_field(c, y, "   Số CCCD", "079090005678")
    y = draw_field(c, y, "   Nghề nghiệp", "Kế toán - Công ty XYZ")
    y = draw_field(c, y, "   Ngày đăng ký về hộ", "2017-03-15")

    # Member 2: Child
    y -= 0.3*cm
    y = draw_field(c, y, "2. Họ tên", "Nguyễn Minh Anh")
    y = draw_field(c, y, "   Quan hệ với chủ hộ", "Con")
    y = draw_field(c, y, "   Ngày sinh", "2021-06-20")
    y = draw_field(c, y, "   Giới tính", "Nữ")
    y = draw_field(c, y, "   Ngày đăng ký về hộ", "2021-07-01")

    # Page 2 — summary
    c.showPage()
    y = draw_header(c, "SỔ HỘ KHẨU (tiếp theo)")
    y = draw_section(c, y, "TỔNG HỢP")
    y = draw_field(c, y, "Tổng số nhân khẩu", f"{CUSTOMER['so_thanh_vien_gia_dinh']} người")
    y = draw_field(c, y, "Chủ hộ", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Tình trạng hôn nhân chủ hộ", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Số con", str(CUSTOMER["so_con"]))

    y = draw_section(c, y, "XÁC NHẬN")
    y = draw_field(c, y, "Ngày xác nhận", "2024-01-15")
    y = draw_field(c, y, "Cơ quan xác nhận", "Công an Phường Bến Nghé, Quận 1, TP.HCM")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 4. Housing Survey / Collateral PDF (3 pages)
# ============================================================
def generate_housing_pdf():
    path = os.path.join(CUSTOMER_DIR, "04_tham_dinh_nha_o.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "PHIẾU THẨM ĐỊNH TÀI SẢN / NHÀ Ở", "Ma ho so: TD-2026-001234")
    y = draw_section(c, y, "1. THÔNG TIN CHỦ SỞ HỮU")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Địa chỉ liên hệ", CUSTOMER["dia_chi_nha"])

    y = draw_section(c, y, "2. THÔNG TIN BẤT ĐỘNG SẢN")
    y = draw_field(c, y, "Địa chỉ BĐS", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "Loại hình nhà ở", CUSTOMER["loai_nha"])
    y = draw_field(c, y, "Diện tích sử dụng", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Năm xây dựng", str(CUSTOMER["nam_xay"]))
    y = draw_field(c, y, "Số tầng tòa nhà", str(CUSTOMER["so_tang_toa_nha"]))
    y = draw_field(c, y, "Tầng căn hộ", str(CUSTOMER["tang_can_ho"]))
    y = draw_field(c, y, "Có thang máy", "Có" if CUSTOMER["co_thang_may"] else "Không")
    y = draw_field(c, y, "Vật liệu tường", CUSTOMER["vat_lieu_tuong"])
    y = draw_field(c, y, "Tình trạng khẩn cấp", CUSTOMER["tinh_trang"])

    y = draw_section(c, y, "3. THÔNG SỐ KỸ THUẬT CHI TIẾT")
    # These map to the normalized AVG/MODE/MEDI columns in Home Credit
    y = draw_field(c, y, "Chất lượng căn hộ (1-10)", "7.5 / 10")
    y = draw_field(c, y, "Diện tích tầng hầm", "5.2 m2 (phan chung)")
    y = draw_field(c, y, "Diện tích đất (toàn khu)", "12,500 m2")
    y = draw_field(c, y, "Diện tích sinh hoạt", f"{CUSTOMER['dien_tich']} m2")
    y = draw_field(c, y, "Diện tích chung", "350 m2 (sảnh, hành lang)")
    y = draw_field(c, y, "Số lối vào", "2 (chính + phía sau)")

    # Page 2 — More details
    c.showPage()
    y = draw_header(c, "PHIẾU THẨM ĐỊNH (tiếp theo)")
    y = draw_section(c, y, "4. THÔNG TIN KHU VỰC")
    y = draw_field(c, y, "Quận/Huyện", "Binh Thanh, TP.HCM")
    y = draw_field(c, y, "Mật độ dân số (tương đối)", "0.035 (khu trung tâm)")
    y = draw_field(c, y, "Xếp hạng khu vực", "2 (Tốt)")
    y = draw_field(c, y, "Xếp hạng khu vực (TP)", "2 (Tốt)")
    y = draw_field(c, y, "Đăng ký cùng vùng sống", "Có")
    y = draw_field(c, y, "Đăng ký cùng TP làm việc", "Có")
    y = draw_field(c, y, "Sống cùng vùng làm việc", "Có")

    y = draw_section(c, y, "5. ĐÁNH GIÁ GIÁ TRỊ")
    y = draw_field(c, y, "Giá trị ước tính", f"{CUSTOMER['gia_tri_bds']:,.0f} VND")
    y = draw_field(c, y, "Giá thị trường tham khảo", "4,200,000,000 - 4,800,000,000 VND")
    y = draw_field(c, y, "Phương pháp định giá", "So sánh thị trường + Chi phí")
    y = draw_field(c, y, "Ngày định giá", "2026-03-10")
    y = draw_field(c, y, "Đơn vị định giá", "Công ty Thẩm định giá Hoàng Quân")

    y = draw_section(c, y, "6. TÌNH TRẠNG PHÁP LÝ")
    y = draw_field(c, y, "Sổ hồng/sổ đỏ", "Có - Số HD.123456.BKHBD")
    y = draw_field(c, y, "Trạng thái pháp lý", "Đã được cấp giấy CN QSDĐ")
    y = draw_field(c, y, "Tranh chấp", "Không có tranh chấp")
    y = draw_field(c, y, "Thế chấp hiện tại", "Không")

    # Page 3 — Photos and conclusion
    c.showPage()
    y = draw_header(c, "PHIẾU THẨM ĐỊNH (Trang cuối)")
    y = draw_section(c, y, "7. HÌNH ẢNH BẤT ĐỘNG SẢN")
    y = draw_field(c, y, "[Ảnh mặt tiền]", "Chung cư Vinhomes Central Park - Mặt ngoài")
    y = draw_field(c, y, "[Ảnh mặt trong]", "Phòng khách - 25m2, nội thất hiện đại")
    y = draw_field(c, y, "[Ảnh phòng ngủ]", "2 phòng ngủ, đầy đủ nội thất")
    y = draw_field(c, y, "[Ảnh khu vực chung]", "Hồ bơi, phòng gym, sảnh")

    y = draw_section(c, y, "8. KẾT LUẬN THẨM ĐỊNH")
    y = draw_field(c, y, "Kết luận", "Đủ điều kiện thế chấp")
    y = draw_field(c, y, "Giá trị cho vay tối đa", "3,150,000,000 VND (70% gia tri)")
    y = draw_field(c, y, "Ghi chú", "BĐS ở vị trí tốt, thanh khoản cao")

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, "Nhân viên thẩm định:")
    c.drawString(12*cm, y, "Quản lý phê duyệt:")
    y -= 1.5*cm
    c.drawString(3*cm, y, "Phạm Văn Đức")
    c.drawString(12*cm, y, "Hoàng Thị Mai")

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 5. Loan Application PDF (2 pages)
# ============================================================
def generate_loan_application_pdf():
    path = os.path.join(CUSTOMER_DIR, "05_don_vay.pdf")
    c = canvas.Canvas(path, pagesize=A4)

    y = draw_header(c, "ĐƠN ĐỀ NGHỊ VAY VỐN", "Ma don: DV-2026-001234")
    y = draw_section(c, y, "I. THÔNG TIN NGƯỜI VAY")
    y = draw_field(c, y, "Họ và tên", CUSTOMER["ho_ten"])
    y = draw_field(c, y, "Số CCCD", CUSTOMER["cccd"])
    y = draw_field(c, y, "Ngày sinh", CUSTOMER["ngay_sinh"])
    y = draw_field(c, y, "Giới tính", CUSTOMER["gioi_tinh"])
    y = draw_field(c, y, "Trình độ học vấn", CUSTOMER["trinh_do_hoc_van"])
    y = draw_field(c, y, "Tình trạng hôn nhân", CUSTOMER["tinh_trang_hon_nhan"])
    y = draw_field(c, y, "Địa chỉ", CUSTOMER["dia_chi_nha"])
    y = draw_field(c, y, "SDT", "0901 234 567")
    y = draw_field(c, y, "Email", "minh.nguyenvan@gmail.com")
    y = draw_field(c, y, "Có xe ô tô", "Có" if CUSTOMER["co_xe_oto"] else "Không")
    y = draw_field(c, y, "Tuổi xe (năm)", str(CUSTOMER["tuoi_xe"]))
    y = draw_field(c, y, "Có bất động sản", "Có" if CUSTOMER["co_bat_dong_san"] else "Không")
    y = draw_field(c, y, "Người đồng hành", CUSTOMER["nguoi_dong_hanh"])
    y = draw_field(c, y, "Số di động liên lạc được", "Có")
    y = draw_field(c, y, "Số điện thoại bàn", "Không")
    y = draw_field(c, y, "Email", "Có")
    y = draw_field(c, y, "Ngày đổi SĐT gần nhất", "2024-06-15 (khoảng 270 ngày trước)")

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
    y = draw_field(c, y, "Mô tả chi tiết", "Mua xe Toyota Vios 2024 để phục vụ đi lại và công việc.")
    y = draw_field(c, y, "", "Xe mới 100%, mua tại Toyota Bến Thành, Q.1, TP.HCM.")

    y = draw_section(c, y, "IV. TÀI SẢN THẾ CHẤP")
    y = draw_field(c, y, "Loại TSĐB", "Xe ô tô mua mới + Căn hộ chung cư")
    y = draw_field(c, y, "Giá trị xe", f"{CUSTOMER['gia_tri_hang_hoa']:,.0f} VND")
    y = draw_field(c, y, "Giá trị căn hộ", f"{CUSTOMER['gia_tri_bds']:,.0f} VND")

    y = draw_section(c, y, "V. NGUỒN TRẢ NỢ")
    y = draw_field(c, y, "Thu nhập hàng tháng", f"{CUSTOMER['luong_thang'] + CUSTOMER['phu_cap']:,.0f} VND")
    y = draw_field(c, y, "Chi phí sinh hoạt", "15,000,000 VND/thang")
    y = draw_field(c, y, "Thu nhập khả dụng", f"{CUSTOMER['luong_thang'] + CUSTOMER['phu_cap'] - 15000000:,.0f} VND")
    y = draw_field(c, y, "Tỷ lệ nợ/thu nhập (DTI)", f"{CUSTOMER['tra_hang_thang']/(CUSTOMER['luong_thang']+CUSTOMER['phu_cap'])*100:.1f}%")

    y = draw_section(c, y, "VI. CAM KẾT")
    c.setFont(FONT, 9)
    commitments = [
        "- Tôi cam kết các thông tin trên là đúng sự thật.",
        "- Tôi đồng ý để ngân hàng xác minh thông tin và tra cứu CIC.",
        "- Tôi cam kết sử dụng vốn vay đúng mục đích.",
        "- Tôi cam kết trả nợ gốc và lãi đúng hạn.",
    ]
    for cm_text in commitments:
        y = draw_field(c, y, "", cm_text)

    y -= 1.5*cm
    c.setFont(FONT, 10)
    c.drawString(3*cm, y, f"TP.HCM, ngay 15 thang 03 nam 2026")
    y -= 1.5*cm
    c.drawString(3*cm, y, "Người vay ký tên:")
    y -= 1.5*cm
    c.drawString(3*cm, y, CUSTOMER["ho_ten"])

    c.save()
    print(f"  Created: {path}")


# ============================================================
# 6. Bank Statement CSV (6 months)
# ============================================================
def generate_bank_statement():
    path = os.path.join(CUSTOMER_DIR, "06_sao_ke_ngan_hang.csv")
    transactions = []
    base_date = date(2025, 9, 1)  # 6 months ago from ~March 2026

    balance = 45000000  # Starting balance 45M

    for month in range(6):
        month_date = base_date + timedelta(days=30 * month)

        # Salary — 5th of each month
        salary_date = month_date.replace(day=5)
        balance += CUSTOMER["luong_thang"] + CUSTOMER["phu_cap"]
        transactions.append({
            "date": salary_date.isoformat(),
            "description": "LUONG T" + str(month_date.month) + "/2025 - CTY TNHH CONG NGHE ABC VIET NAM",
            "credit": CUSTOMER["luong_thang"] + CUSTOMER["phu_cap"],
            "debit": 0,
            "balance": balance
        })

        # Rent — 10th
        rent_date = month_date.replace(day=10)
        rent = 8000000
        balance -= rent
        transactions.append({
            "date": rent_date.isoformat(),
            "description": "TIEN THUE NHA T" + str(month_date.month),
            "credit": 0, "debit": rent, "balance": balance
        })

        # Bills — 15th
        bill_date = month_date.replace(day=15)
        for desc, amt in [("TIEN DIEN EVN HCMC", 850000), ("TIEN NUOC SAWACO", 180000), ("CUOC INTERNET FPT", 250000)]:
            balance -= amt
            transactions.append({
                "date": bill_date.isoformat(),
                "description": desc,
                "credit": 0, "debit": amt, "balance": balance
            })

        # Daily spending
        for day_offset in [3, 7, 12, 18, 22, 25, 28]:
            spend_date = month_date + timedelta(days=day_offset)
            desc_options = [
                ("VINMART - SIEU THI", 450000 + month * 10000),
                ("GRAB - DI CHUYEN", 180000),
                ("GD ONLINE - SHOPEE", 320000),
                ("NHA HANG - AN TOI", 550000),
                ("CAFE HIGHLANDS", 85000),
                ("GRAB FOOD", 120000),
                ("GYM - CALIFORNIA", 1200000),
            ]
            desc, amt = desc_options[day_offset % len(desc_options)]
            balance -= int(amt)
            transactions.append({
                "date": spend_date.isoformat(),
                "description": desc,
                "credit": 0, "debit": int(amt), "balance": balance
            })

        # Savings transfer — 20th
        save_date = month_date.replace(day=20)
        save_amt = 10000000
        balance -= save_amt
        transactions.append({
            "date": save_date.isoformat(),
            "description": "CK TIET KIEM - TK 0071009999999",
            "credit": 0, "debit": save_amt, "balance": balance
        })

    # Sort by date
    transactions.sort(key=lambda x: x["date"])

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "description", "credit", "debit", "balance"])
        writer.writeheader()
        writer.writerows(transactions)
    print(f"  Created: {path} ({len(transactions)} transactions)")


# ============================================================
# 7. CIC API Mock Response
# ============================================================
def generate_cic_mock():
    path = os.path.join(CUSTOMER_DIR, "07_cic_api_response.json")

    # Generate realistic bureau_balance monthly_status (40 months each)
    # STATUS: C=Closed, 0=Current, X=Unknown, 1=DPD_1-30, 2=DPD_31-60, etc.
    def make_monthly_status(n_months, credit_active):
        """Generate realistic monthly payment statuses."""
        statuses = []
        for m in range(n_months):
            mb = -m  # MONTHS_BALANCE: 0 (current), -1 (last month), etc.
            if credit_active == "Closed" and m < 4:
                s = "C"  # Closed months
            elif m < 2:
                s = "0"  # Current - on time
            elif m < 30:
                s = "0"  # Mostly on time
            else:
                s = "X"  # Old/unknown
            statuses.append({"MONTHS_BALANCE": mb, "STATUS": s})
        return statuses

    cic_data = {
        "api_version": "CIC-VN-v2.1",
        "query_timestamp": "2026-03-15T10:23:41+07:00",
        "customer_id": CUSTOMER["cccd"],
        "customer_name": CUSTOMER["ho_ten"],

        "ext_source_scores": {
            "EXT_SOURCE_1": 0.502,
            "EXT_SOURCE_2": 0.654,
            "EXT_SOURCE_3": 0.571,
            "_note": "Normalized external credit scores from partner bureaus"
        },

        "credit_inquiry_counts": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
            "AMT_REQ_CREDIT_BUREAU_DAY": 0,
            "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
            "AMT_REQ_CREDIT_BUREAU_MON": 1,
            "AMT_REQ_CREDIT_BUREAU_QRT": 2,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 4
        },

        "social_circle": {
            "OBS_30_CNT_SOCIAL_CIRCLE": 2,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "OBS_60_CNT_SOCIAL_CIRCLE": 1,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0
        },

        "bureau_records": [
            {
                "SK_ID_BUREAU": 5001001,  # Integer, not string
                "CREDIT_ACTIVE": "Closed",
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -1200,
                "CREDIT_DAY_OVERDUE": 0,
                "DAYS_CREDIT_ENDDATE": -200,
                "DAYS_ENDDATE_FACT": -210,
                "AMT_CREDIT_MAX_OVERDUE": 0,
                "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": 150000000,
                "AMT_CREDIT_SUM_DEBT": 0,
                "AMT_CREDIT_SUM_LIMIT": 0,
                "AMT_CREDIT_SUM_OVERDUE": 0,
                "CREDIT_TYPE": "Consumer credit",
                "DAYS_CREDIT_UPDATE": -200,
                "AMT_ANNUITY": 5500000,
                "monthly_status": make_monthly_status(40, "Closed")
            },
            {
                "SK_ID_BUREAU": 5001002,  # Integer, not string
                "CREDIT_ACTIVE": "Active",
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -600,
                "CREDIT_DAY_OVERDUE": 0,
                "DAYS_CREDIT_ENDDATE": 400,
                "DAYS_ENDDATE_FACT": None,
                "AMT_CREDIT_MAX_OVERDUE": 0,
                "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": 200000000,
                "AMT_CREDIT_SUM_DEBT": 80000000,
                "AMT_CREDIT_SUM_LIMIT": 0,
                "AMT_CREDIT_SUM_OVERDUE": 0,
                "CREDIT_TYPE": "Consumer credit",
                "DAYS_CREDIT_UPDATE": -30,
                "AMT_ANNUITY": 7200000,
                "monthly_status": make_monthly_status(20, "Active")
            }
        ],

        "thin_file_flag": False,
        "cic_score_equivalent": 680,
        "debt_group": 1
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cic_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")


# ============================================================
# 8. Internal DB Mock — Previous Applications, POS, Installments, Credit Card
# ============================================================
def generate_internal_db_mock():
    sk_id_curr = 100002

    # -- Previous Applications --
    prev_apps = [
        {
            "SK_ID_PREV": 2001001,
            "SK_ID_CURR": sk_id_curr,
            "NAME_CONTRACT_TYPE": "Consumer loans",
            "AMT_ANNUITY": 5500000,
            "AMT_APPLICATION": 150000000,
            "AMT_CREDIT": 150000000,
            "AMT_DOWN_PAYMENT": 0,
            "AMT_GOODS_PRICE": 150000000,
            "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
            "HOUR_APPR_PROCESS_START": 10,
            "FLAG_LAST_APPL_PER_CONTRACT": "Y",
            "NFLAG_LAST_APPL_IN_DAY": 1,
            "RATE_DOWN_PAYMENT": 0.0,
            "RATE_INTEREST_PRIMARY": 0.12,
            "RATE_INTEREST_PRIVILEGED": 0.10,
            "NAME_CASH_LOAN_PURPOSE": "XAP",
            "NAME_CONTRACT_STATUS": "Approved",
            "DAYS_DECISION": -800,
            "NAME_PAYMENT_TYPE": "Cash through the bank",
            "CODE_REJECT_REASON": "XAP",
            "NAME_TYPE_SUITE": "Unaccompanied",
            "NAME_CLIENT_TYPE": "Repeater",
            "NAME_GOODS_CATEGORY": "Consumer Electronics",
            "NAME_PORTFOLIO": "POS",
            "NAME_PRODUCT_TYPE": "x-sell",
            "CHANNEL_TYPE": "Country-wide",
            "SELLERPLACE_AREA": 500,
            "NAME_SELLER_INDUSTRY": "Consumer electronics",
            "CNT_PAYMENT": 24,
            "NAME_YIELD_GROUP": "middle",
            "PRODUCT_COMBINATION": "POS industry with interest",
            "DAYS_FIRST_DRAWING": -790,
            "DAYS_FIRST_DUE": -760,
            "DAYS_LAST_DUE_1ST_VERSION": -40,
            "DAYS_LAST_DUE": -40,
            "DAYS_TERMINATION": -40,
            "NFLAG_INSURED_ON_APPROVAL": 0
        },
        {
            "SK_ID_PREV": 2001002,
            "SK_ID_CURR": sk_id_curr,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "AMT_ANNUITY": 7200000,
            "AMT_APPLICATION": 200000000,
            "AMT_CREDIT": 200000000,
            "AMT_DOWN_PAYMENT": 0,
            "AMT_GOODS_PRICE": 200000000,
            "WEEKDAY_APPR_PROCESS_START": "FRIDAY",
            "HOUR_APPR_PROCESS_START": 14,
            "FLAG_LAST_APPL_PER_CONTRACT": "Y",
            "NFLAG_LAST_APPL_IN_DAY": 1,
            "RATE_DOWN_PAYMENT": 0.0,
            "RATE_INTEREST_PRIMARY": 0.11,
            "RATE_INTEREST_PRIVILEGED": 0.09,
            "NAME_CASH_LOAN_PURPOSE": "Repairs",
            "NAME_CONTRACT_STATUS": "Approved",
            "DAYS_DECISION": -500,
            "NAME_PAYMENT_TYPE": "Cash through the bank",
            "CODE_REJECT_REASON": "XAP",
            "NAME_TYPE_SUITE": "Spouse, partner",
            "NAME_CLIENT_TYPE": "Repeater",
            "NAME_GOODS_CATEGORY": "XNA",
            "NAME_PORTFOLIO": "Cash",
            "NAME_PRODUCT_TYPE": "x-sell",
            "CHANNEL_TYPE": "Credit and cash offices",
            "SELLERPLACE_AREA": -1,
            "NAME_SELLER_INDUSTRY": "XNA",
            "CNT_PAYMENT": 36,
            "NAME_YIELD_GROUP": "low_normal",
            "PRODUCT_COMBINATION": "Cash",
            "DAYS_FIRST_DRAWING": -490,
            "DAYS_FIRST_DUE": -460,
            "DAYS_LAST_DUE_1ST_VERSION": 640,
            "DAYS_LAST_DUE": None,
            "DAYS_TERMINATION": None,
            "NFLAG_INSURED_ON_APPROVAL": 1
        }
    ]

    # -- POS Cash Balance (monthly for each prev app) --
    pos_cash = []
    for month in range(24):
        pos_cash.append({
            "SK_ID_PREV": 2001001,
            "SK_ID_CURR": sk_id_curr,
            "MONTHS_BALANCE": -month,
            "CNT_INSTALMENT": 24,
            "CNT_INSTALMENT_FUTURE": max(0, 24 - month),
            "NAME_CONTRACT_STATUS": "Completed" if month < 2 else "Active",
            "SK_DPD": 0,
            "SK_DPD_DEF": 0
        })
    for month in range(18):
        pos_cash.append({
            "SK_ID_PREV": 2001002,
            "SK_ID_CURR": sk_id_curr,
            "MONTHS_BALANCE": -month,
            "CNT_INSTALMENT": 36,
            "CNT_INSTALMENT_FUTURE": max(0, 36 - month),
            "NAME_CONTRACT_STATUS": "Active",
            "SK_DPD": 0,
            "SK_DPD_DEF": 0
        })

    # -- Installments Payments --
    installments = []
    for i in range(1, 25):  # Loan 1: 24 installments
        days_inst = -800 + 30 * i
        installments.append({
            "SK_ID_PREV": 2001001,
            "SK_ID_CURR": sk_id_curr,
            "NUM_INSTALMENT_VERSION": 1,
            "NUM_INSTALMENT_NUMBER": i,
            "DAYS_INSTALMENT": days_inst,
            "DAYS_ENTRY_PAYMENT": days_inst - 2,  # paid 2 days early
            "AMT_INSTALMENT": 5500000,
            "AMT_PAYMENT": 5500000
        })
    for i in range(1, 19):  # Loan 2: 18 of 36 paid so far
        days_inst = -500 + 30 * i
        installments.append({
            "SK_ID_PREV": 2001002,
            "SK_ID_CURR": sk_id_curr,
            "NUM_INSTALMENT_VERSION": 1,
            "NUM_INSTALMENT_NUMBER": i,
            "DAYS_INSTALMENT": days_inst,
            "DAYS_ENTRY_PAYMENT": days_inst - 1,
            "AMT_INSTALMENT": 7200000,
            "AMT_PAYMENT": 7200000
        })

    # -- Credit Card Balance --
    credit_card = []
    for month in range(12):
        credit_card.append({
            "SK_ID_PREV": 2001003,
            "SK_ID_CURR": sk_id_curr,
            "MONTHS_BALANCE": -month,
            "AMT_BALANCE": 5000000 + month * 200000,
            "AMT_CREDIT_LIMIT_ACTUAL": 50000000,
            "AMT_DRAWINGS_ATM_CURRENT": 0,
            "AMT_DRAWINGS_CURRENT": 3500000 + month * 100000,
            "AMT_DRAWINGS_OTHER_CURRENT": 0,
            "AMT_DRAWINGS_POS_CURRENT": 3500000 + month * 100000,
            "AMT_INST_MIN_REGULARITY": 2500000,
            "AMT_PAYMENT_CURRENT": 3000000,
            "AMT_PAYMENT_TOTAL_CURRENT": 3000000,
            "AMT_RECEIVABLE_PRINCIPAL": 4500000 + month * 150000,
            "AMT_RECIVABLE": 5000000 + month * 200000,
            "AMT_TOTAL_RECEIVABLE": 5000000 + month * 200000,
            "CNT_DRAWINGS_ATM_CURRENT": 0,
            "CNT_DRAWINGS_CURRENT": 3,
            "CNT_DRAWINGS_OTHER_CURRENT": 0,
            "CNT_DRAWINGS_POS_CURRENT": 3,
            "CNT_INSTALMENT_MATURE_CUM": month,
            "NAME_CONTRACT_STATUS": "Active",
            "SK_DPD": 0,
            "SK_DPD_DEF": 0
        })

    # Save all
    internal_data = {
        "SK_ID_CURR": sk_id_curr,
        "previous_applications": prev_apps,
        "pos_cash_balance": pos_cash,
        "installments_payments": installments,
        "credit_card_balance": credit_card
    }
    path = os.path.join(CUSTOMER_DIR, "08_internal_db.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(internal_data, f, ensure_ascii=False, indent=2)
    print(f"  Created: {path}")
    print(f"    - {len(prev_apps)} previous applications")
    print(f"    - {len(pos_cash)} POS cash records")
    print(f"    - {len(installments)} installment records")
    print(f"    - {len(credit_card)} credit card records")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Generating mock data for: {CUSTOMER['ho_ten']}")
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
    print(f"  All mock data generated successfully!")
    print(f"{'='*60}")
