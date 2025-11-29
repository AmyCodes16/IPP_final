import streamlit as st
import pandas as pd
import altair as alt
import pickle
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data & Model Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['features']
    except FileNotFoundError:
        return None, None

df = load_data()
model, model_features = load_model()
alt.data_transformers.enable("vegafusion")
color_scale = alt.Scale(domain=['No', 'Yes'], range=['#4c78a8', '#e45756'])

# --- 3. CLEAN SIDEBAR DESIGN ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=50) # Optional: Add a logo icon if you have one
st.sidebar.title("Churn Intelligence")
st.sidebar.markdown("Data-Driven Retention System")
st.sidebar.markdown("---")

# Main Menu Logic
# We use a radio button to select the "Module" (Phase 1 vs Phase 2)
main_selection = st.sidebar.radio(
    "Select Module:",
    ["📊 Visualization & Analysis", "🔮 Smart Prediction System"],
    label_visibility="collapsed" # Hides the label for a cleaner look
)

# Logic to handle the "Collapsible" Sub-menu
if main_selection == "📊 Visualization & Analysis":
    
    # This creates the collapsible box
    with st.sidebar.expander("📂 Analysis Modules", expanded=True):
        page = st.radio(
            "Navigate to:",
            [
                "1. Baseline & Demographics", 
                "2. Financial Analysis", 
                "3. Service Analysis", 
                "4. Executive Summary"
            ],
            label_visibility="collapsed"
        )
else:
    # If Phase 2 is selected, we force the page variable
    page = "Prediction System"

# --- 4. Main Page Content (Logic remains the same, just rendering) ---

if page == "1. Baseline & Demographics":
    st.title("Phase 1: Baseline & Demographics")
    
    # Metric Row
    total_customers = len(df)
    churn_count = len(df[df['Churn']=='Yes'])
    churn_rate = (churn_count / total_customers) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churn Rate", f"{churn_rate:.1f}%", "-Bad Trend")
    c3.metric("Retained", f"{total_customers - churn_count:,}")
    st.divider()

    # Chart 1
    source = df.groupby('Churn').agg(count=('Churn', 'count')).reset_index()
    source['percentage'] = (source['count'] / total_customers * 100).round(1).astype(str) + '%'
    
    base = alt.Chart(source).mark_bar().encode(
        x=alt.X('Churn', axis=None), y=alt.Y('count'), color=alt.Color('Churn', scale=color_scale, legend=None)
    )
    text = base.mark_text(dy=-10, color='black').encode(text='percentage', y=alt.Y('count'))
    st.altair_chart((base+text).properties(title=f"Overall Churn Split", height=300), use_container_width=True)
    
    st.subheader("Demographic Breakdown")
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    df_melt = df.melt(id_vars=['Churn'], value_vars=demo_cols, var_name='Category', value_name='Value')
    chart = alt.Chart(df_melt).mark_bar().encode(
        column=alt.Column('Category', title=None, header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
        x=alt.X('Value', axis=None),
        y=alt.Y('count()', stack='normalize', axis=alt.Axis(format='%')),
        color=alt.Color('Churn', scale=color_scale),
        tooltip=['Category', 'Value', 'Churn', 'count()']
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

elif page == "2. Financial Analysis":
    st.title("Phase 1: Financial Analysis")
    st.markdown("Analyzing how **cost** and **tenure** impact customer retention.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(alt.Chart(df).transform_density('MonthlyCharges', as_=['MonthlyCharges', 'density'], groupby=['Churn']).mark_area(opacity=0.7).encode(
            x='MonthlyCharges', y='density:Q', color=alt.Color('Churn', scale=color_scale)
        ).properties(title="Monthly Charges Distribution", height=350), use_container_width=True)
    with col2:
        st.altair_chart(alt.Chart(df).transform_density('tenure', as_=['tenure', 'density'], groupby=['Churn']).mark_area(opacity=0.7).encode(
            x='tenure', y='density:Q', color=alt.Color('Churn', scale=color_scale)
        ).properties(title="Tenure Distribution", height=350), use_container_width=True)

elif page == "3. Service Analysis":
    st.title("Phase 1: Service Risk Factors")
    
    services = ['Contract', 'InternetService', 'TechSupport', 'OnlineSecurity']
    df_serv = df.melt(id_vars=['Churn'], value_vars=services, var_name='Service', value_name='Option')
    st.altair_chart(alt.Chart(df_serv).mark_bar().encode(
        column=alt.Column('Service', title=None, header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
        x=alt.X('Option', axis=None),
        y=alt.Y('count()', stack='normalize', axis=alt.Axis(format='%')),
        color=alt.Color('Churn', scale=color_scale),
        tooltip=['Service', 'Option', 'Churn', 'count()']
    ).properties(title="Churn Risk by Service Type", height=400), use_container_width=True)

elif page == "4. Executive Summary":
    st.title("Phase 1: Executive Summary")
    
    # Correlation Chart
    df_corr = df.copy()
    for c in ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']:
        df_corr[c] = df_corr[c].map({'Yes': 1, 'No': 0})
    
    # FIX: Drop customerID *before* get_dummies to prevent it from exploding into thousands of columns
    if 'customerID' in df_corr.columns:
        df_corr = df_corr.drop('customerID', axis=1)
    
    df_corr = pd.get_dummies(df_corr)
    
    corr = df_corr.corr()[['Churn']].reset_index().rename(columns={'index':'Feature'})
    corr = corr[corr['Feature']!='Churn'].sort_values(by='Churn', ascending=False)
    
    st.altair_chart(alt.Chart(corr).mark_bar().encode(
        y=alt.Y('Feature', sort='-x'), x='Churn',
        color=alt.condition(alt.datum.Churn > 0, alt.value('#e45756'), alt.value('#4c78a8')),
        tooltip=['Feature', 'Churn']
    ).properties(title="Top Statistical Drivers of Churn", height=600), use_container_width=True)

elif page == "Prediction System":
    st.title("🔮 Smart Churn Prediction System")
    
    if model is None:
        st.error("⚠️ Model not found. Please run 'Phase2_Modeling.ipynb' to generate it.")
    else:
        st.markdown("Enter customer details to generate a real-time risk assessment.")
        
        with st.form("predict_form"):
            st.subheader("Customer Profile")
            c1, c2, c3 = st.columns(3)
            with c1:
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
                total = st.number_input("Total Charges ($)", 0.0, 9000.0, monthly*tenure)
            with c2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dep = st.selectbox("Dependents", ["No", "Yes"])
            with c3:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
                pay = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                
            submit = st.form_submit_button("Run Risk Analysis", type="primary")
            
        if submit:
            # Build Input Data
            input_df = pd.DataFrame({
                'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner], 'Dependents': [dep],
                'tenure': [tenure], 'PhoneService': ['Yes'], 'MultipleLines': ['No'], 'InternetService': [internet],
                'OnlineSecurity': ['No'], 'OnlineBackup': ['No'], 'DeviceProtection': ['No'], 'TechSupport': [tech],
                'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'Contract': [contract], 'PaperlessBilling': ['Yes'],
                'PaymentMethod': [pay], 'MonthlyCharges': [monthly], 'TotalCharges': [total]
            })
            
            # Align with Model
            input_encoded = pd.get_dummies(input_df)
            input_final = input_encoded.reindex(columns=model_features, fill_value=0)
            
            # Predict
            prob = model.predict_proba(input_final)[0][1]
            pred = model.predict(input_final)[0]
            
            st.divider()
            
            # Smart Result Display
            col_1, col_2 = st.columns([1, 2])
            
            with col_1:
                if pred == 1:
                    st.metric("Risk Level", "CRITICAL", f"{prob:.1%} Risk", delta_color="inverse")
                else:
                    st.metric("Risk Level", "SAFE", f"{prob:.1%} Risk", delta_color="normal")
            
            with col_2:
                if pred == 1:
                    st.error(f"**High Churn Probability Detected**\n\nThis customer shows behaviors strongly linked to churn (e.g., {contract}, High Charges).")
                    st.button("Generate Retention Offer") # Visual only
                else:
                    st.success("**Low Churn Probability**\n\nThis customer fits the profile of a loyal subscriber.")