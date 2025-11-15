"""
Welcome page for the Mango AI application.
"""
import streamlit as st
from ..config import IMAGES_DIR


def render():
    """Render the welcome page."""
    st.title("AI for your mango trees")
    st.image(str(IMAGES_DIR / "welcome.jpeg"), use_container_width=True)
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
        - Storage : Shelf life of mangoes being short (2 to 3 weeks) they are cooled as soon as possible to storage temperature of 13 degree Celcius. 
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
    st.image(str(IMAGES_DIR / "cost.png"), use_container_width=True)
    st.write("""
        ## Projected income:
        - Repayment Schedule (Mango Cultivation)
        - Total Financial Outlay(Rs) 34400
        - Margin money @ 5% of TFO((Rs.) 1720
        - Bank Loan(Rs.) 32680 (Amount in Rs.)
        - Repayment period is 14 years including 7 years grace period.
    """)

