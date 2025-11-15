"""
Mango variety classification page.
"""
import streamlit as st
import tempfile
import os
from ..utils import predict_and_annotate
from ..config import IMAGES_DIR, TEMP_DIR


def render():
    """Render the variety classification page."""
    st.title("Mango Variety Classifier 🥭")
    st.write(
        "Find your mango variety from alphonsa, Ambika, Amrapalli, Banganpalli, Chausa, Dasheri, "
        "Himsagar, Kesar, Langra, Malgova, Mallika, Neelam, Raspuri, totapuri, Vanraj"
    )
    st.image(str(IMAGES_DIR / "fruit_variety.jpeg"), use_container_width=True)
    st.write("Upload an image to predict the mango variety.")

    uploaded_file = st.file_uploader("Choose a mango image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(TEMP_DIR)) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            img_path = tmp_file.name

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Find Variety"):
            try:
                result_img, labels = predict_and_annotate(img_path)
                # result_img is a PIL Image, so we can display it directly
                st.image(result_img, caption="Detected Mango(s)", use_container_width=True)

                if labels:
                    st.success("Prediction(s):")
                    for label in labels:
                        st.write(f"- {label}")
                else:
                    st.warning("No mango detected confidently.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)

