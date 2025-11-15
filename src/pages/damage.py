"""
Mango fruit damage/grading page.
"""
import streamlit as st
import tempfile
import os
from ..utils import predict_damage
from ..config import IMAGES_DIR, TEMP_DIR


def render():
    """Render the damage detection page."""
    st.title("🥭 Mango Grading System")
    st.write(
        "Protocols for Mango Grading According to UN/ECE Standards mangoes are divided into three "
        "grades Extra Class, Class I, and Class II. The model classifies your mango image into one "
        "of these three classes based on the level of damage."
    )
    st.image(str(IMAGES_DIR / "fruit_grade.jpeg"), use_container_width=True)
    
    uploaded_file = st.file_uploader("Upload a mango image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Damage"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(TEMP_DIR)) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                img_rgb, pred_rf_label, pred_rf = predict_damage(tmp_file_path)

                st.subheader("🧠 Model Prediction:")
                st.write(f"🟢 Random Forest Prediction: **{pred_rf_label}**")

                st.image(img_rgb, caption="Processed Image", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

