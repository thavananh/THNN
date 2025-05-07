import streamlit as st

DEFAULT_CHART_COLORS = {
    "Ham": "#2ECC71",
    "Spam": "#E74C3C",
}

def setting_page():
    st.header("⚙️ Cài đặt")

    st.subheader("🎨 Tuỳ chỉnh màu biểu đồ cảm xúc")

    if "chart_colors" not in st.session_state:
        st.session_state.chart_colors = DEFAULT_CHART_COLORS.copy()

    changed = False
    for label, default_color in st.session_state.chart_colors.items():
        new_color = st.color_picker(
            f"Màu cho **{label}**",
            value=default_color,
            key=f"color_{label}"
        )
        if new_color != st.session_state.chart_colors[label]:
            st.session_state.chart_colors[label] = new_color
            changed = True

    if changed:
        st.success("✅ Đã lưu màu mới. Mở lại trang **Thống kê** để xem thay đổi!")

    st.subheader("🖋️ Tuỳ chỉnh font chữ (toàn ứng dụng)")

    fonts = [
        "Roboto", "Inter", "Helvetica", "Arial",
        "Times New Roman", "Georgia", "Courier New"
    ]
    if "app_font" not in st.session_state:
        st.session_state.app_font = fonts[0]

    font_sel = st.selectbox(
        "Chọn font hiển thị",
        fonts,
        index=fonts.index(st.session_state.app_font)
    )
    if font_sel != st.session_state.app_font:
        st.session_state.app_font = font_sel
        st.rerun()

    st.markdown(
        f"""
        <style>
            html, body, [class*="css"] {{
                font-family: '{st.session_state.app_font}', sans-serif !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.info("Các tuỳ chọn khác sẽ được bổ sung trong phiên bản sắp tới.")
