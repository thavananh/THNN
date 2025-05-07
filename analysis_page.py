# analysis_page.py
import streamlit as st
from utils import get_prediction


def clear_input():
    st.session_state["input_text"] = ""
    st.session_state["stop_file"] = False


def analysis_page():
    st.header("📊 Phân Loại Email")

    # Dropdown để chọn model, gọn và căn trái
    col_group, _ = st.columns([2, 8])
    with col_group:
        group = st.selectbox(
            "Chọn nhóm model", 
            ["Deep Learning", "Model ML"],
            help="Chọn nhóm mô hình",
            key="group_choice"
        )
        if group == "Deep Learning":
            models = ["cnn_lstm_attention",]
        else:
            models = ["logistic", "random_forest", "xgboost", "naive_bayes", "extra_trees", "lightgbm", "voting"]
        model_choice = st.selectbox(
            "Chọn model", models,
            key="model_choice"
        )

    # Nhập câu hoặc tải file
    raw_text = st.text_area("Nhập câu (tiếng Việt) hoặc tiếng Anh:", key="input_text", height=150)
    uploaded = st.file_uploader("… hoặc chọn file .txt", type=["txt"])

    # Nút Clear, phân tích câu và file
    col_clear, col_spacer, col_clf, col_file = st.columns([1, 6, 2.5, 2.5])

    with col_clear:
        st.button(
            "Clear 🧹",
            help="Xoá nội dung",
            use_container_width=True,
            on_click=clear_input
        )

    with col_clf:
        btn_sentence = st.button("Phân tích câu", use_container_width=True)

    with col_file:
        btn_file = st.button("Phân tích file", use_container_width=True)

    # Phân tích từng câu
    if btn_sentence:
        if not raw_text.strip():
            st.warning("Vui lòng nhập văn bản")
        else:
            pred = get_prediction(raw_text, model_choice)
            if pred:
                st.success("Kết quả phân loại:")
                st.write(f"• **{pred}**")
            else:
                st.error("Không thể lấy kết quả từ API.")

    # Phân tích file từng dòng
    if btn_file:
        if uploaded is None:
            st.warning("Chưa chọn file")
        else:
            fname = uploaded.name
            content = uploaded.read().decode("utf-8")
            lines = [l for l in content.splitlines() if l.strip()]
            total = len(lines)

            if "stop_file" not in st.session_state:
                st.session_state["stop_file"] = False

            progress = st.progress(0)
            if st.button("Stop", key="stop_file_btn"):
                st.session_state["stop_file"] = True

            exp = st.expander(f"Kết quả phân tích file: {fname}", expanded=True)
            with exp:
                for i, line in enumerate(lines, start=1):
                    if st.session_state.get("stop_file"):
                        st.warning("Quá trình đã bị dừng bởi người dùng.")
                        break

                    st.write(f"**{line}**")
                    pred = get_prediction(line, model_choice)
                    if pred:
                        st.write(f"  ↳ {pred}")
                    else:
                        st.write("  ↳ Lỗi phân loại.")

                    progress.progress(i / total)
                else:
                    st.success("Hoàn tất phân tích file!")