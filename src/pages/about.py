"""
About page for the Mango AI application.
"""
import streamlit as st


def render():
    """Render the about page."""
    st.title("🍋 About Mangoo")
    st.write(
        "Welcome to Mangoo, your ultimate AI companion for cultivating healthy and delicious mangoes! "
        "Our platform leverages cutting-edge artificial intelligence to assist you with various aspects "
        "of mango farming, from identifying specific mango varieties to detecting diseases, assessing "
        "ripeness, and even checking for fruit damage."
    )
    st.write("With Mangoo, you can:")
    st.markdown("""
    - **Classify Mango Varieties**: Instantly identify your mango type from a wide range of popular varieties by simply uploading an image.
    - **Detect Leaf Diseases**: Get early warnings about potential diseases like Powdery Mildew or Die Back on your mango leaves, allowing for timely intervention.
    - **Assess Ripeness**: Determine the optimal ripeness of your mangoes using image analysis combined with key physiological data, ensuring perfect taste and texture.
    - **Check for Fruit Damage**: Quickly identify and categorize damage on mango fruits, helping you sort and grade your harvest efficiently.
    """)
    st.write(
        "Our goal is to empower mango farmers and enthusiasts with smart tools, making mango cultivation "
        "more efficient, sustainable, and fruitful. Join us in revolutionizing mango farming with AI!"
    )

