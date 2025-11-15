"""
Mango disease detection page.
"""
import streamlit as st
import tempfile
import os
import numpy as np
from ..utils import predict_disease
from ..config import IMAGES_DIR, TEMP_DIR


def render():
    """Render the disease detection page."""
    st.title("Mango disease detector")
    st.write(
        "Mango leaf diseases can be identified by observing symptoms like discoloration, spots, "
        "and deformities. Common diseases include anthracnose, powdery mildew, and bacterial black spot, "
        "each showing distinct patterns on the leaves. Early detection helps in managing and controlling "
        "the spread of these diseases."
    )
    st.image(str(IMAGES_DIR / "leave_damage.jpeg"), use_container_width=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Disease"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(TEMP_DIR)) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                label, confidence, heatmap = predict_disease(tmp_file_path)

                st.success(f"Predicted Disease: **{label}**")
                st.write(f"Confidence Score: **{confidence * 100:.2f}%**")

                st.subheader("Interpretation (LayerCAM Heatmap)")
                # Ensure heatmap is in a format st.image can display (e.g., numpy array)
                if isinstance(heatmap, np.ndarray) and heatmap.ndim == 2:  # Grayscale heatmap
                    st.image(heatmap, use_container_width=True, 
                            caption="LayerCAM visualization highlighting key regions.")
                elif isinstance(heatmap, np.ndarray) and heatmap.ndim == 3 and heatmap.shape[2] == 3:  # RGB heatmap
                    st.image(heatmap, use_container_width=True, 
                            caption="LayerCAM visualization highlighting key regions.")
                else:
                    st.warning("Heatmap could not be displayed. Ensure it's a valid image format.")
            except Exception as e:
                st.error(f"Error during disease detection: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

