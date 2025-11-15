"""
Main Streamlit application for Mango AI.
"""
import streamlit as st
from .config import APP_TITLE, APP_ICON, APP_LAYOUT
from .pages import welcome, about, variety, disease, ripeness, damage


def render_navigation_buttons():
    """Render navigation buttons at the top of the page."""
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
        .stButton>button[kind="primary"] {
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
            is_active = st.session_state['page'] == key
            button_style = "primary" if is_active else "secondary"

            if st.button(label, key=f"nav_btn_{key}", use_container_width=True, type=button_style):
                st.session_state["page"] = key
                st.rerun()  # Force rerun to switch page content and update button style

    # Add a horizontal line for visual separation below the navigation buttons
    st.markdown("---")


def main():
    """Main application entry point."""
    # Set wide layout for better visual space
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

    # Initialize session state for page navigation if not already set
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'

    # Render the navigation buttons at the top
    render_navigation_buttons()

    # Based on the current page in session state, render the corresponding content
    page_handlers = {
        'welcome': welcome.render,
        'about': about.render,
        'variety': variety.render,
        'disease': disease.render,
        'ripeness': ripeness.render,
        'damage': damage.render,
    }

    handler = page_handlers.get(st.session_state['page'], welcome.render)
    handler()


if __name__ == "__main__":
    main()

