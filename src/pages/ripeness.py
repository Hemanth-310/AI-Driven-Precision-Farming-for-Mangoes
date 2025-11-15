"""
Mango ripeness analysis page.
"""
import streamlit as st
import tempfile
import os
from ..utils import ImageProcessing, predict_tss, predict_sensory_from_tss, interpret_score, plot_radar_chart
from ..config import IMAGES_DIR, TEMP_DIR


def render():
    """Render the ripeness analysis page."""
    st.title("🍋 Mango Ripeness Detector")
    st.write(
        "This tool helps you assess the ripeness of mangoes based on image analysis and key "
        "physiological parameters. Upload a mango image along with its storage time, days after "
        "flowering (DAFS), and weight to get insights into its ripeness stage and sensory attributes."
    )
    st.image(str(IMAGES_DIR / "fruit_ripeness.jpeg"), use_container_width=True)
    

    uploaded_file = st.file_uploader("Upload a mango image...", type=["jpg", "jpeg", "png"])
    storage_time = st.number_input("Storage Time (days)", min_value=0.0, step=1.0)
    dafs = st.number_input("Days After Flowering (DAFS)", min_value=0.0, step=1.0)
    weight = st.number_input("Weight (grams)", min_value=0.0, step=1.0)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Mango Image", use_container_width=True)

        if st.button("Check for Ripeness"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(TEMP_DIR)) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                processor = ImageProcessing(tmp_file_path)
                rgb_stage = processor.classify_rgb_stage()
                avg_rgb = processor.get_avg_rgb()

                if avg_rgb:
                    avg_r, avg_g, avg_b = avg_rgb
                    predicted_tss = predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b)

                    if predicted_tss:
                        tss_stage = "Unripe" if predicted_tss < 8 else "Mid-Ripe" if predicted_tss < 11 else "Ripe"

                        st.subheader("🎯 Ripeness Assessment")
                        st.write(f"**Fruit stage**: {rgb_stage}")
                        st.write(f"**Predicted TSS**: {predicted_tss:.2f} °Brix")

                        sensory_result = predict_sensory_from_tss(predicted_tss)

                        st.subheader("🌟 Sensory Attributes:")
                        radar_scores = {}
                        for k, v in sensory_result.items():
                            if k in ["Taste", "Flavour", "Appearance"]:
                                label = interpret_score(k, v)
                                st.write(f"{k}: {v:.2f} ({label})")
                                radar_scores[k] = v
                            else:
                                st.write(f"{k}: {v:.2f}")

                        st.subheader("📊 Sensory Quality Radar Chart")
                        plot_radar_chart(radar_scores)

                        fig_rgb = processor.plot_rgb_with_image()
                        if fig_rgb:
                            st.markdown("### 🎨 RGB Region Analysis")
                            st.pyplot(fig_rgb)
                    else:
                        st.error("TSS prediction failed. Please check inputs or model path.")
                else:
                    st.error("Could not get average RGB from the image.")
            except Exception as e:
                st.error(f"Error during ripeness analysis: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

