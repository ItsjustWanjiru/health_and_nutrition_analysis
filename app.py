import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. DATA & MODEL LOADING ---
@st.cache_resource
def load_assets():
    # Ensure these files are in your project folder
    model = joblib.load('final_health_model.pkl')
    features = joblib.load('model_features.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, features, scaler

@st.cache_data
def load_data():
    # Ensure this CSV was exported from your notebook
    return pd.read_csv('cleaned_health_data.csv')

# Initialize
st.set_page_config(page_title="Africa Health Intelligence", layout="wide")

try:
    model, features, scaler = load_assets()
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Initialization Error: Ensure .pkl and .csv files are present. {e}")
    st.stop()

# --- 2. DYNAMIC COLUMN MAPPING ---
cols = df.columns.tolist()
y_axis = "Life expectancy at birth, total (years)" if "Life expectancy at birth, total (years)" in cols else cols[1]
year_col = "Year" if "Year" in cols else cols[0]
country_col = "Country Name" if "Country Name" in cols else "Country"

# --- 3. HEADER & KEY PERFORMANCE INDICATORS (KPIs) ---
st.title("🌍 Eastern & Southern Africa Health Intelligence Dashboard")
st.markdown("##### *Strategic Decision Support System for Population Longevity*")

avg_val = df[y_axis].mean()
top_nation = df.groupby(country_col)[y_axis].mean().idxmax()
max_life = df[y_axis].max()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Regional Avg Longevity", f"{avg_val:.1f} Years")
kpi2.metric("Top Performing Nation", top_nation)
kpi3.metric("Historical Peak", f"{max_life:.1f} Years")

st.markdown("---")

# --- 4. SIDEBAR SIMULATION CONTROLS ---
st.sidebar.header("🕹️ Simulation Controls")
st.sidebar.info("Adjust indicators to simulate how policy changes impact the ML prediction.")

user_inputs = {}
for feat in features:
    # Based on Min-Max scaling 0-1
    user_inputs[feat] = st.sidebar.slider(f"{feat}", 0.0, 1.0, 0.5)

# --- 5. MAIN INTERFACE TABS ---
tab1, tab2 = st.tabs(["📊 Historical Trends (EDA)", "🤖 Prediction Engine (ML)"])

with tab1:
    st.header("50-Year Health Trajectories")
    
    # Historical Line Chart
    fig_line = px.line(df, x=year_col, y=y_axis, color=country_col,
                       title="Life Expectancy Trends by Country (1975–2024)",
                       template="plotly_white", height=500)
    st.plotly_chart(fig_line, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Longevity Drivers")
        mort_cols = [c for c in cols if "mortality" in c.lower()]
        x_axis = mort_cols[0] if mort_cols else cols[2]
        fig_scat = px.scatter(df, x=x_axis, y=y_axis, color=country_col,
                              title=f"Correlation: Longevity vs {x_axis}")
        st.plotly_chart(fig_scat, use_container_width=True)
    with c2:
        st.subheader("Regional Stats")
        st.dataframe(df.describe().T, height=350)

with tab2:
    st.header("Hybrid Stacking Simulation")
    st.write("Click below to pass your sidebar parameters through the Stacking Ensemble.")

    if st.button("🚀 Run Longevity Simulation"):
        # Predict
        input_df = pd.DataFrame([user_inputs])
        scaled_pred = model.predict(input_df)[0]
        
        # Rescale (Assuming 40-85 year range)
        pred_years = (scaled_pred * (85 - 40)) + 40
        
        st.markdown("---")
        res1, res2 = st.columns([1, 2])
        
        with res1:
            st.metric("Predicted Longevity", f"{pred_years:.2f} Years", 
                      delta=f"{pred_years - avg_val:.2f} vs Avg")
            
            # THE GAUGE CHART (Visual feedback for simulation)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_years,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [40, 90]},
                    'bar': {'color': "#00CC96"},
                    'steps' : [
                        {'range': [40, 60], 'color': "#FF4B4B"},
                        {'range': [60, 75], 'color': "#FFA500"},
                        {'range': [75, 90], 'color': "#00CC96"}]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res2:
            st.subheader("Model Interpretation")
            st.write("The Stacking Ensemble (XGBoost + Random Forest) has processed the non-linear interactions of your inputs.")
            st.info(f"**Insight:** A predicted value of {pred_years:.2f} years suggests that the input profile is " + 
                    ("above" if pred_years > avg_val else "below") + " the regional historical average.")
            
            st.write("Prediction Confidence (Scaled Output):")
            st.progress(float(np.clip(scaled_pred, 0, 1)))
    else:
        st.warning("👈 Adjust settings in the sidebar and hit 'Run' to start the simulation.")

# --- FOOTER ---
st.markdown("---")
st.caption("WanjiruNjogu MSc DSA Data Mining Project | Hybrid Stacking Implementation | Strathmore University")