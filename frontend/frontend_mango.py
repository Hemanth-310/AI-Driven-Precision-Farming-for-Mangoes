import streamlit as st
import os
import tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np # Added numpy import for image handling

# Assume these are correctly implemented and available in your environment
from predict_and_annotate import predict_and_annotate
from disease_detection_utils import predict_disease
from ripeness_utils import ImageProcessing, predict_tss, predict_sensory_from_tss, interpret_score, plot_radar_chart
from fruit_grading_utils import predict_damage


# --- Navigation UI: Just Buttons on Top ---
def render_navigation_buttons():
    # Define pages and their display names/icons
    tabs = {
        "🏠 Welcome": "welcome",
        "📖 About mangoo": "about",
        "🍋 Variety": "variety",
        "🌿 Disease": "disease",
        "🍈 Ripeness": "ripeness",
        "🍂 Damage": "damage"
    }

    # Set default page if not already in session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'

    # Inject custom CSS for basic styling and to hide Streamlit's default header
    st.markdown("""
    <style>
        /* Hide Streamlit's default header bar */
        header {
            display: none !important;
        }
        /* General styling for the main content block to give space below the buttons */
        .block-container {
            padding-top: 1rem; /* Adjust this value as needed */
        }
        /* Style for standard Streamlit buttons (optional, for a cleaner look) */
        .stButton>button {
            border-radius: 8px; /* Slightly rounded corners */
            box-shadow: 1px 1px 3px rgba(0,0,0,0.1); /* Subtle shadow */
            margin: 0 5px; /* Space between buttons */
            transition: all 0.2s ease-in-out; /* Smooth transitions */
            font-weight: normal; /* Default font weight */
            background-color: #f0f2f6; /* Light background for buttons */
            color: #333; /* Darker text color */
        }
        .stButton>button:hover {
            background-color: #e0e2e6; /* Slightly darker on hover */
            border-color: #e0e2e6;
        }
        /* Style for the active/selected button */
        /* Note: Streamlit's internal classes can change, inspect with browser dev tools if this breaks */
        .stButton>button[kind="primary"] { /* Targeting the primary button style often works for active */
             background-color: #007bff; /* A distinct color for the active button */
             color: white;
             box-shadow: 0 2px 5px rgba(0, 123, 255, 0.3); /* More prominent shadow for active */
        }
    </style>
    """, unsafe_allow_html=True)

    # Use st.columns to lay out the buttons horizontally
    cols = st.columns(len(tabs))

    # Render buttons within the columns
    for i, (label, key) in enumerate(tabs.items()):
        with cols[i]:
            # If the current page matches this button's key, make it a primary button
            # Primary buttons have a distinct default style that we can then override with CSS
            is_active = st.session_state['page'] == key
            button_style = "primary" if is_active else "secondary" # Use Streamlit's button 'kind'

            if st.button(label, key=f"nav_btn_{key}", use_container_width=True, type=button_style):
                st.session_state["page"] = key
                st.rerun() # Force rerun to switch page content and update button style

    # Add a horizontal line for visual separation below the navigation buttons
    st.markdown("---")


# --- Pages (functions remain largely the same) ---

def welcome_page():
    st.title("AI for your mango trees")
    st.image(r"images\welcome.jpeg", use_container_width=True)
    st.write("Hello!! Cultivate the tastiest mangoes in your yard with the help of AI! 🍋🌱")
    st.write("Use the navigation buttons above to explore different features.")
    st.title("About Mango 🌱")
    st.write("""
        ### Soil:
        - Mango grows well on all types of soil provided they are deep and well drained.
        - Red loamy soils are quite ideal.
        - Alkaline, ill-drained and soils with rocky substratum are not suitable for successful cultivation of mango crop.
        - In India, mango is grown on lateritic, alluvial, kankar, and other types of soil.
        - Rich, medium, and well-drained soils give better results.
        - Very poor, stony, and soils with hard substratum should be avoided.
        - The vigour and cropping behavior of a mango tree are affected by the soil type.
        
        ### Climate:
        - Mango is grown in both tropical and sub-tropical conditions.
        - It can tolerate a wide range of climatic conditions.
        - For growing mango on a commercial and profitable scale, the temperature and rainfall must be within a clearly defined range.
        - Temperature, rainfall, altitude, and wind velocity all influence the growth and production of mango.
        - Mango thrives well under humid and dry conditions.
        - It requires good rainfall during its growing season (June to October) and rainless, dry weather from November onwards.
        - Rainy or cloudy weather during flowering favours the incidence of powdery mildew disease and leafhoppers.
    """)
    st.write("""
        ### Post Harvest Management :
        - Storage : Shelf life of mangoes being short (2 to 3 weeks) they are cooled as soon as possible tostorage temperatue of 13 degree Celcius. 
        - A few varieties can withstand storage temperature of 10 degree Celcius. 
        - Steps involved in post harvest handling include preparation, grading, washing, drying, waxing, packing, pre-cooling, palletisation and transportation.
        - Packaging : Mangoes are generally packed in corrugated fibre board boxes 40 cm x 30 cm x 20cm in size.
        - Fruits are packed in single layer 8 to 20 fruits per carton. The boxes should have sufficient number of air holes (about 8% of the surface area) to allow good ventillation.
        - Financial institutions have also formulated mango financing schemes in potential areas for expansion of area under mango. Individual mango development schemes with farm infrastructure facilities like well, pumpset, fencing and drip irrigation system etc. have also been considered.
        - Farm model for financing one hectare mango orchard is furnished.
            
        ### Unit Cost : The unit cost varies from state to state. The cost presented here is indicative only.
        - The enterpreneurs and the bankers are requested to consult our Regional Offices for the latest information in this regard. The unit cost estimated for this model scheme is Rs.34400/- per ha capitalised upto the fifth year.I.
        - Financial Analysis : Results of financial analysis are indicated below :
            - NPW at 15% DF : Rs.59058 (+)
            - BCR at 15% DF : 2.34
            - IRR : 25.59%
            - Margin Money : The margin money assumed in this model scheme is 5% of the total financial outlay.
            - Interest Rate : Interest rate may be decided by the banks as per the guidelines of RBI.
            - Security : Banks may charge such security as permissible under RBI guidelines.
            - Repayment : The bank loan with interest is repayable within 14 years with 7 years grace period 
            
        ### Cost and Income from Mango Cultivation
        - Spacing : 10m x 10m
        - Plant population : 100
        ## Estimated cost:
    """)
    st.image(r"images/cost.png", use_container_width=True)
    st.write("""
        ## Projected income:
        - Repayment Schedule (Mango Cultivation)
        - Total Financial Outlay(Rs) 34400
        - Margin money @ 5% of TFO((Rs.) 1720
        - Bank Loan(Rs.) 32680 (Amount in Rs.)
        - Repayment period is 14 years including 7 years grace period.""")
    

def about_mango():
    st.title("🍋 About Mangoo")
    st.write("Welcome to Mangoo, your ultimate AI companion for cultivating healthy and delicious mangoes! Our platform leverages cutting-edge artificial intelligence to assist you with various aspects of mango farming, from identifying specific mango varieties to detecting diseases, assessing ripeness, and even checking for fruit damage.")
    st.write("With Mangoo, you can:")
    st.markdown("""
    - **Classify Mango Varieties**: Instantly identify your mango type from a wide range of popular varieties by simply uploading an image.
    - **Detect Leaf Diseases**: Get early warnings about potential diseases like Powdery Mildew or Die Back on your mango leaves, allowing for timely intervention.
    - **Assess Ripeness**: Determine the optimal ripeness of your mangoes using image analysis combined with key physiological data, ensuring perfect taste and texture.
    - **Check for Fruit Damage**: Quickly identify and categorize damage on mango fruits, helping you sort and grade your harvest efficiently.
    """)
    st.write("Our goal is to empower mango farmers and enthusiasts with smart tools, making mango cultivation more efficient, sustainable, and fruitful. Join us in revolutionizing mango farming with AI!")

# st.set_page_config(
#     page_title="Mangoo AI",
#     page_icon="🍋",
#     layout="wide"
# )


def mango_varity():
    st.title("Mango Variety Classifier 🥭")
    st.write("Find your mango variety from alphonsa, Ambika, Amrapalli, Banganpalli, Chausa, Dasheri, Himsagar, Kesar, Langra, Malgova, Mallika, Neelam, 'Raspuri, totapuri, Vanraj")
    st.image(r"images\fruit_variety.jpeg", use_container_width=True)
    st.write("Upload an image to predict the mango variety.")

    uploaded_file = st.file_uploader("Choose a mango image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            img_path = tmp_file.name

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Find Variety"):
            try:
                result_img, labels = predict_and_annotate(img_path)
                # Convert result_img to RGB if it's BGR (OpenCV default) for st.image
                if isinstance(result_img, np.ndarray) and result_img.ndim == 3 and result_img.shape[2] == 3:
                     result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
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
                os.remove(img_path)


def mango_disease():
    st.title("Mango disease detector")
    st.write("""Mango leaf diseases can be identified by observing symptoms like discoloration, spots, and deformities. Common diseases include anthracnose, powdery mildew, and bacterial black spot, each showing distinct patterns on the leaves. Early detection helps in managing and controlling the spread of these diseases.""")
    st.image(r"images\leave_damage.jpeg", use_container_width=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Disease"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                label, confidence, heatmap = predict_disease(tmp_file_path)

                st.success(f"Predicted Disease: **{label}**")
                st.write(f"Confidence Score: **{confidence * 100:.2f}%**")

                st.subheader("Interpretation (LayerCAM Heatmap)")
                # Ensure heatmap is in a format st.image can display (e.g., numpy array)
                if isinstance(heatmap, np.ndarray) and heatmap.ndim == 2: # Grayscale heatmap
                    st.image(heatmap, use_container_width=True, caption="LayerCAM visualization highlighting key regions.")
                elif isinstance(heatmap, np.ndarray) and heatmap.ndim == 3 and heatmap.shape[2] == 3: # RGB heatmap
                    st.image(heatmap, use_container_width=True, caption="LayerCAM visualization highlighting key regions.")
                else:
                    st.warning("Heatmap could not be displayed. Ensure it's a valid image format.")
            except Exception as e:
                st.error(f"Error during disease detection: {str(e)}")
            finally:
                os.remove(tmp_file_path)


def mango_ripeness_analysis():
    st.title("🍋 Mango Ripeness Detector")
    st.write("""This tool helps you assess the ripeness of mangoes based on image analysis and key physiological parameters. Upload a mango image along with its storage time, days after flowering (DAFS), and weight to get insights into its ripeness stage and sensory attributes.""")
    st.image(r"images\fruit_ripeness.jpeg", use_container_width=True)
    

    uploaded_file = st.file_uploader("Upload a mango image...", type=["jpg", "jpeg", "png"])
    storage_time = st.number_input("Storage Time (days)", min_value=0.0, step=1.0)
    dafs = st.number_input("Days After Flowering (DAFS)", min_value=0.0, step=1.0)
    weight = st.number_input("Weight (grams)", min_value=0.0, step=1.0)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Mango Image", use_container_width=True)

        if st.button("Check for Ripeness"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
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
                os.remove(tmp_file_path)


def fruit_damage_checker():
    st.title("🥭 Mango Grading System ")
    st.write("Protocols for Mango Grading According to UN/ECE Standards mangoes are divided into three grades Extra Class, Class I, and Class II. The model classifies your mango image into one of these three classes based on the level of damage.")
    st.image(r"images\fruit_grade.jpeg", use_container_width=True)
    
    uploaded_file = st.file_uploader("Upload a mango image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Damage"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
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
                os.remove(tmp_file_path)


# --- Main App Logic ---
def main():
    # Set wide layout for better visual space
    st.set_page_config(page_title="Mangoo AI", page_icon="🍋", layout="wide")

    # Initialize session state for page navigation if not already set
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'

    # Render the navigation buttons at the top
    render_navigation_buttons()

    # Based on the current page in session state, render the corresponding content
    if st.session_state['page'] == 'welcome':
        welcome_page()
    elif st.session_state['page'] == 'about':
        about_mango()
    elif st.session_state['page'] == 'variety':
        mango_varity()
    elif st.session_state['page'] == 'disease':
        mango_disease()
    elif st.session_state['page'] == 'ripeness':
        mango_ripeness_analysis()
    elif st.session_state['page'] == 'damage':
        fruit_damage_checker()


if __name__ == "__main__":
    main()