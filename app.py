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

# --- 2. Helper Functions ---

# function to load the historical data (Phase 1)
@st.cache_data
def load_data():
    df = pd.read_csv('telco_churn.csv')
    # Basic cleanup for visualization
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

# function to load the ML Model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['features']
    except FileNotFoundError:
        return None, None

# function to pre-process raw input data (for Phase 2)
def preprocess_input(df, model_features):
    df = df.copy()
    
    #Clean numeric columns (just in case)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Map SeniorCitizen if it's 0/1
    if 'SeniorCitizen' in df.columns and df['SeniorCitizen'].dtype in ['int64', 'float64']:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df)
    
    #  Align with Model Features (CRITICAL STEP)
    df_final = df_encoded.reindex(columns=model_features, fill_value=0)
    
    return df_final

# load resources
df = load_data()
model, model_features = load_model()
alt.data_transformers.enable("vegafusion")
color_scale = alt.Scale(domain=['No', 'Yes'], range=['#4c78a8', '#e45756'])

# --- 3. Sidebar Navigation ---
st.sidebar.title("Churn Intelligence")
st.sidebar.markdown("Data-Driven Retention System")
st.sidebar.markdown("---")

# main Selection
main_selection = st.sidebar.radio(
    "Select Module:",
    ["Visualization & Analysis", "Smart Prediction System"],
    label_visibility="collapsed"
)

# Conditional Sub-menu
if main_selection == "Visualization & Analysis":
    with st.sidebar.expander("📂 Analysis Modules", expanded=True):
        page = st.radio(
            "Navigate to:",
            ["1. Baseline & Demographics", "2. Financial Analysis", "3. Service Analysis", "4. Executive Summary"],
            label_visibility="collapsed"
        )
else:
    page = "Prediction System"

# --- 4. Main Content Logic ---

if page == "1. Baseline & Demographics":
    st.title("Phase 1: Baseline & Demographics")
    # Metrics
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
    st.altair_chart((base+text).properties(title="Overall Churn Split", height=300), use_container_width=True)
    
    st.subheader("Demographic Breakdown")
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    df_melt = df.melt(id_vars=['Churn'], value_vars=demo_cols, var_name='Category', value_name='Value')
    st.altair_chart(alt.Chart(df_melt).mark_bar().encode(
        column=alt.Column('Category', title=None, header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
        x=alt.X('Value', axis=None), y=alt.Y('count()', stack='normalize', axis=alt.Axis(format='%')),
        color=alt.Color('Churn', scale=color_scale), tooltip=['Category', 'Value', 'Churn', 'count()']
    ).properties(height=300), use_container_width=True)

elif page == "2. Financial Analysis":
    st.title("Phase 1: Financial Analysis")
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
        x=alt.X('Option', axis=None), y=alt.Y('count()', stack='normalize', axis=alt.Axis(format='%')),
        color=alt.Color('Churn', scale=color_scale), tooltip=['Service', 'Option', 'Churn', 'count()']
    ).properties(title="Churn Risk by Service Type", height=400), use_container_width=True)

elif page == "4. Executive Summary":
    st.title("Phase 1: Executive Summary")
    df_corr = df.copy()
    for c in ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']:
        df_corr[c] = df_corr[c].map({'Yes': 1, 'No': 0})
    if 'customerID' in df_corr.columns: df_corr = df_corr.drop('customerID', axis=1)
    df_corr = pd.get_dummies(df_corr)
    corr = df_corr.corr()[['Churn']].reset_index().rename(columns={'index':'Feature'}).sort_values(by='Churn', ascending=False)
    corr = corr[corr['Feature']!='Churn']
    st.altair_chart(alt.Chart(corr).mark_bar().encode(
        y=alt.Y('Feature', sort='-x'), x='Churn',
        color=alt.condition(alt.datum.Churn > 0, alt.value('#e45756'), alt.value('#4c78a8')),
        tooltip=['Feature', 'Churn']
    ).properties(title="Top Statistical Drivers of Churn", height=600), use_container_width=True)

# --- PAGE 5: THE SMART PREDICTION SYSTEM (UPDATED) ---
elif page == "Prediction System":
    st.title("AI Churn Prediction System")
    
    if model is None:
        st.error("⚠️ Model not found. Please run 'Phase2_Modeling.ipynb' to generate it.")
    else:
        # TABS for Single vs Batch
        tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 Batch Prediction (Upload CSV)"])
        
        # --- TAB 1: Single Customer (Your existing manual form) ---
        with tab1:
            st.write("Enter details for a single customer to assess their real-time risk.")
            with st.form("predict_form"):
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
                
                # Hidden defaults for compatibility
                submit = st.form_submit_button("Run Risk Analysis", type="primary")
            
            if submit:
                # Prepare Data
                input_data = {
                    'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner], 'Dependents': [dep],
                    'tenure': [tenure], 'PhoneService': ['Yes'], 'MultipleLines': ['No'], 'InternetService': [internet],
                    'OnlineSecurity': ['No'], 'OnlineBackup': ['No'], 'DeviceProtection': ['No'], 'TechSupport': [tech],
                    'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'Contract': [contract], 'PaperlessBilling': ['Yes'],
                    'PaymentMethod': [pay], 'MonthlyCharges': [monthly], 'TotalCharges': [total]
                }
                input_df = pd.DataFrame(input_data)
                
                # Preprocess & Predict
                final_input = preprocess_input(input_df, model_features)
                prob = model.predict_proba(final_input)[0][1]
                pred = model.predict(final_input)[0]
                
                st.divider()
                col_1, col_2 = st.columns([1, 2])
                with col_1:
                    if pred == 1:
                        st.metric("Risk Level", "CRITICAL", f"{prob:.1%} Risk", delta_color="inverse")
                    else:
                        st.metric("Risk Level", "SAFE", f"{prob:.1%} Risk", delta_color="normal")
                with col_2:
                    if pred == 1:
                        st.error(f"**High Churn Probability Detected**\n\nBased on {contract} contract and charges, this customer is at risk.")
                    else:
                        st.success("**Low Churn Probability**\n\nThis customer fits the profile of a loyal subscriber.")

        # --- TAB 2: Batch Prediction (The New Feature) ---
        with tab2:
            st.header("Upload Customer Data")
            st.write("Upload a CSV file containing customer details. The system will predict churn for every row.")
            
            # File Uploader
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # 1. Load Data
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"✅ File uploaded successfully! ({len(batch_df)} rows)")
                    
                    # 2. Preview
                    with st.expander("Preview Uploaded Data"):
                        st.dataframe(batch_df.head())
                    
                    if st.button("Generate Predictions"):
                        with st.spinner("Analyzing customers..."):
                            # 3. Preprocess
                            # We create a copy so we don't mess up the original for the final display
                            final_input = preprocess_input(batch_df, model_features)
                            
                            # 4. Predict
                            predictions = model.predict(final_input)
                            probabilities = model.predict_proba(final_input)[:, 1]
                            
                            # 5. Add Results to Original DataFrame
                            batch_df['Churn Prediction'] = ["Yes" if x == 1 else "No" for x in predictions]
                            batch_df['Risk Probability'] = probabilities
                            
                            # 6. Display Results
                            st.success("Analysis Complete!")
                            
                            # Highlight High Risk
                            st.subheader("High Risk Customers (Urgent Attention)")
                            high_risk = batch_df[batch_df['Risk Probability'] > 0.7]
                            st.dataframe(high_risk.style.format({'Risk Probability': '{:.1%}'}))
                            
                            # 7. Download Button
                            csv = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Full Report",
                                csv,
                                "churn_predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    st.info("Ensure your CSV has the same column names as the training data (gender, tenure, MonthlyCharges, etc.)")
