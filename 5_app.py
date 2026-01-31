import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="Sentinel Auditor", layout="wide")
MODEL_PATH = "models/sentinel_optimized.pkl"

@st.cache_resource
def load_assets():
    """Load the pipeline and extract the model for SHAP without bs naming errors."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Grab the last step (the XGBoost model) regardless of what you named it
    raw_model = pipeline.steps[-1][1]
    explainer = shap.TreeExplainer(raw_model)
    
    return pipeline, explainer

def main():
    st.title("🛡️ Sentinel: Credit Risk Command Center")
    
    pipeline, explainer = load_assets()
    
    if pipeline is None:
        st.error("Model file not found in 'models/'. Run your training script first!")
        return

    tab1, tab2 = st.tabs(["🎯 Single Audit", "📊 Batch Portfolio"])

    with tab1:
        st.subheader("Individual Risk Assessment")
        # Creating a mockup input for the M3 to process
        c1, c2, c3 = st.columns(3)
        with c1:
            limit = st.number_input("Limit Balance", value=50000)
            pay_0 = st.selectbox("Current Status (PAY_0)", [-1, 0, 1, 2, 3, 4])
        with c2:
            age = st.slider("Age", 18, 80, 30)
            bill_1 = st.number_input("Recent Bill Amount", value=2000)
        with c3:
            pay_1 = st.number_input("Recent Payment", value=1000)

        if st.button("Analyze Account"):
            # Create input with lowercase names to match model expectations
            features = [limit, 1, 2, 1, age, pay_0, 0, 0, 0, 0, 0, 
                        bill_1, 0, 0, 0, 0, 0, pay_1, 0, 0, 0, 0, 0]
            df_input = pd.DataFrame([features], columns=pipeline.feature_names_in_)
            df_input.columns = [c.lower() for c in df_input.columns] # Force lowercase
            
            prob = pipeline.predict_proba(df_input)[0][1]
            st.metric("Default Probability", f"{prob:.2%}")
            
            # SHAP Force Plot
            st.write("### Risk Drivers (SHAP Explanation)")
            shap_values = explainer.shap_values(df_input)
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.force_plot(explainer.expected_value, shap_values[0], df_input.iloc[0], 
                            matplotlib=True, show=False)
            st.pyplot(plt.gcf())

    with tab2:
        st.subheader("Bulk Portfolio Processing")
        uploaded_file = st.file_uploader("Upload 'cleaned_credit_data.csv'", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if st.button("🚀 Run Mass Audit"):
                # Clean: Drop targets and force lowercase
                data = df.drop(columns=['ID', 'id', 'default_payment_next_month'], errors='ignore')
                data.columns = [c.lower() for c in data.columns]
                
                # Inference
                probs = pipeline.predict_proba(data)[:, 1]
                df['Risk_Score'] = probs
                df['Verdict'] = ["High Risk" if p > 0.5 else "Safe" for p in probs]
                
                st.success(f"M3 scanned {len(df)} records in record time.")
                st.dataframe(df.sort_values('Risk_Score', ascending=False))
                
                # Download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Audit", csv, "sentinel_audit_results.csv", "text/csv")

if __name__ == "__main__":
    main()