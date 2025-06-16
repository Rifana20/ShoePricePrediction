import streamlit as st
import joblib
import pandas as pd
import base64

# Set page config
st.set_page_config(page_title="Shoe Price Predictor", layout="wide")

# Load and embed background image using base64
def set_bg(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stSidebar {{
        background-color: rgba(255, 255, 255, 0.85);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function
set_bg("back.jpg")  # Make sure the image file is in the same folder

st.markdown("<h1 style='color: black;'>👟 Shoe Price Predictor</h1>", unsafe_allow_html=True)

# Load the trained model
model = joblib.load("shoe_price_model.pkl")

# Dropdown values
brands = ['Nike', 'Adidas', 'Vans', 'Puma', 'Converse']
types = ['Sneakers', 'Running Shoes', 'Sandals', 'Boots']
genders = ['Men', 'Women', 'Unisex']
materials = ['Leather', 'Canvas', 'Synthetic']

# Sidebar inputs
st.sidebar.header("Choose Shoe Features")
brand = st.sidebar.selectbox("Brand", brands)
shoe_type = st.sidebar.selectbox("Type", types)
gender = st.sidebar.selectbox("Gender", genders)
material = st.sidebar.selectbox("Material", materials)
size = st.sidebar.slider("Size", 5.0, 15.0, 9.0, 0.5)

# Predict button
if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame([[brand, shoe_type, gender, size, material]],
                            columns=["Brand", "Type", "Gender", "Size", "Material"])
    prediction = model.predict(input_df)[0]
    st.markdown(
        f"<h2 style='color: black;'>💰 Estimated Price: ${prediction:,.2f}</h2>",
        unsafe_allow_html=True
    )
    st.write("### 🔎 Details:")
    st.table(input_df)





