# Step 1: Import the libraries we need
import streamlit as st          # Streamlit builds the web app
import numpy as np              # For number arrays
import pandas as pd             # For data tables
import pickle                   # To save/load our trained model
import os

# For training the model fresh (since we embed training here)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered"       # "centered" or "wide"
)


# ============================================================
#  SECTION 1: LOAD / TRAIN THE MODEL
#
#  @st.cache_resource  ← This decorator is IMPORTANT.
#  It tells Streamlit: "Run this function only ONCE and
#  remember the result. Don't retrain every time the user
#  clicks a button!"
# ============================================================
import pickle

@st.cache_resource
def load_model():
    model  = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ============================================================
#  SECTION 2: PAGE HEADER
#  st.title  → biggest heading  (like <h1> in HTML)
#  st.write  → text / markdown  (understands **bold**, etc.)
# ============================================================

st.title("🏠 House Price Predictor \n Made BY Fawad Ahmad")
st.write(
    "This app uses a **Linear Regression** model trained on the "
    "Boston Housing dataset. Adjust the sliders below and click "
    "**Predict** to estimate the house price."
)

# A horizontal line (just a visual divider)
st.divider()


# ============================================================
#  SECTION 3: SIDEBAR — MODEL INFO
#
#  st.sidebar  → everything inside this block appears in
#  the left sidebar panel, keeping the main page clean.
# ============================================================

with st.sidebar:
    st.header("📊 Model Info")
    st.metric("R² Score",  "0.77")
    st.metric("MSE",       "your value here")
    st.write("---")
    st.write("**Algorithm:** Linear Regression")
    st.write("**Scaler:** StandardScaler")
    st.write("**Test size:** 20%")
    st.write("**Random state:** 42")

    st.write("---")
    # st.expander → a collapsible section
    with st.expander("📖 What do the features mean?"):
        st.write("""
        | Feature | Description |
        |---------|-------------|
        | CRIM    | Crime rate per capita |
        | ZN      | % land for large homes |
        | INDUS   | % non-retail business acres |
        | CHAS    | Near Charles River? (0/1) |
        | NOX     | Nitric oxide concentration |
        | RM      | Avg rooms per dwelling |
        | AGE     | % homes built before 1940 |
        | DIS     | Distance to employment centres |
        | RAD     | Highway accessibility index |
        | TAX     | Property-tax rate |
        | PTRATIO | Pupil-teacher ratio |
        | B       | Diversity index |
        | LSTAT   | % lower-status population |
        """)

#  SECTION 4: INPUT SLIDERS
#
#  st.slider(label, min, max, default)
#  → Returns the current value the user has selected.
#
#  st.columns([1,1])  → splits the page into two equal columns
#  so sliders sit side-by-side instead of in a long list.

st.subheader("🎛️ Adjust the Features")

col1, col2 = st.columns(2)     # Two equal columns

with col1:
    crim    = st.slider("CRIM – Crime rate",        0.0,  90.0,  3.6,  step=0.1)
    zn      = st.slider("ZN – Large-lot land %",    0.0, 100.0,  0.0,  step=0.5)
    indus   = st.slider("INDUS – Industrial %",     0.0,  30.0, 11.0,  step=0.1)
    chas    = st.selectbox("CHAS – Near river?", [0, 1])   # Dropdown for binary
    nox     = st.slider("NOX – Nitric oxide",       0.3,   0.9,  0.55, step=0.01)
    rm      = st.slider("RM – Avg rooms",           3.0,   9.0,  6.3,  step=0.1)
    age     = st.slider("AGE – Old homes %",        0.0, 100.0, 68.0,  step=1.0)

with col2:
    dis     = st.slider("DIS – Distance to jobs",   1.0,  12.0,  3.8,  step=0.1)
    rad     = st.slider("RAD – Highway access",     1,    24,    9,    step=1)
    tax     = st.slider("TAX – Tax rate",         180,   720,  408,   step=1)
    ptratio = st.slider("PTRATIO – Pupil/teacher",  12.0, 22.0, 18.5, step=0.1)
    b       = st.slider("B – Diversity index",      0.0, 400.0, 356.0, step=1.0)
    lstat   = st.slider("LSTAT – Lower-status %",   1.0,  38.0, 12.6,  step=0.1)


# ============================================================
#  SECTION 5: PREDICTION
#
#  st.button("label")  → returns True when clicked.
#  We wrap prediction inside an if-block so it only runs
#  when the user actually presses the button.
# ============================================================

st.divider()

if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
    # 1. Assemble the inputs into a 2-D numpy array (1 row, 13 columns)
    input_data = np.array([[crim, zn, indus, chas, nox, rm,
                            age, dis, rad, tax, ptratio, b, lstat]])

    # 2. Scale the inputs using the SAME scaler fitted during training
    #    (Very important — never skip this step!)
    input_scaled = scaler.transform(input_data)

    # 3. Get the prediction
    prediction = model.predict(input_scaled)[0]

    # 4. Show the result
    #    st.success  → green box
    #    st.warning  → yellow box
    #    st.error    → red box
    st.success(f"### 💰 Estimated House Price: **${prediction * 1000:,.0f}**")

    # Extra context based on the prediction value
    if prediction < 15:
        st.warning("This is below the dataset average. High crime or poor location factors detected.")
    elif prediction > 35:
        st.info("This is above average — a high-value neighbourhood!")
    else:
        st.write("This falls within the typical price range of the dataset.")

    # Show the raw input as a table so the user can review their choices
    with st.expander("🔍 See your input values"):
        input_df = pd.DataFrame(
            input_data,
            columns=["CRIM","ZN","INDUS","CHAS","NOX","RM",
                     "AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
        )
        st.dataframe(input_df)          # st.dataframe → interactive table


# ============================================================
#  SECTION 6: FOOTER NOTE
# ============================================================

st.divider()
st.caption(
    "Built with Streamlit 🎈  |  Model: Linear Regression  |  "
    "Dataset: Boston Housing (sklearn)"
)