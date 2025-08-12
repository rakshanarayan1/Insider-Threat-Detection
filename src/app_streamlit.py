import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from fpdf import FPDF
import os
import tempfile

from feature_engineer import run_feature_engineering_from_files

@st.cache_data
def load_model():
    return joblib.load('model_store/iforest_model.joblib')

def color_status(val):
    return ''

def generate_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Insider Threat Detection Report", ln=True, align='C')
    pdf.ln(10)

    col_widths = [50, 35, 35, 35, 35]
    headers = ['User', 'logon_count', 'http_count', 'device_count', 'status']
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align='C')
    pdf.ln()

    for user, row in df.head(50).iterrows():
        pdf.cell(col_widths[0], 10, str(user), border=1)
        pdf.cell(col_widths[1], 10, str(row['logon_count']), border=1)
        pdf.cell(col_widths[2], 10, str(row['http_count']), border=1)
        pdf.cell(col_widths[3], 10, str(row['device_count']), border=1)
        pdf.cell(col_widths[4], 10, row['status'], border=1)
        pdf.ln()

    return pdf.output(dest='S').encode('latin1')

def main():
    st.title("Insider Threat Detection Dashboard")
    model = load_model()

    st.write("### Upload Your Data")
    st.caption("Option 1: Upload structured features CSV (logon_count, http_count, device_count)\n"
               "Option 2: Upload raw log CSVs for logon, http, and device.")

    uploaded_files = st.file_uploader(
        "Upload 1 features.csv **OR** all raw log files (logon.csv, http.csv, device.csv)",
        type=['csv'], accept_multiple_files=True
    )

    features = None

    if uploaded_files:
        # Check if any file looks like a structured features CSV
        if len(uploaded_files) == 1 and 'feature' in uploaded_files[0].name.lower():
            st.info("Detected a structured features file — using directly.")
            features = pd.read_csv(uploaded_files[0], index_col=0)
        else:
            st.info("Detected raw logs — running feature engineering.")
            # Map file names to objects
            files_dict = {f.name.lower(): f for f in uploaded_files}
            logon_file = files_dict.get('logon.csv')
            http_file = files_dict.get('http.csv')
            device_file = files_dict.get('device.csv')

            if not logon_file or not http_file or not device_file:
                st.error("Please upload all three files: logon.csv, http.csv, device.csv")
            else:
                features = run_feature_engineering_from_files(logon_file, http_file, device_file)
    else:
        st.info("No file uploaded — loading default features.csv")
        features_path = os.path.join(os.path.dirname(__file__), '..', 'features.csv')
        features = pd.read_csv(features_path, index_col=0)

    if features is not None:
        preds = model.predict(features)
        features['anomaly'] = preds
        features['status'] = features['anomaly'].map({1: 'Normal', -1: 'Suspicious'})

        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            min_logon, max_logon = col1.slider(
                "Logon Count Range", int(features['logon_count'].min()),
                int(features['logon_count'].max()),
                (int(features['logon_count'].min()), int(features['logon_count'].max()))
            )
            min_http, max_http = col2.slider(
                "HTTP Count Range", int(features['http_count'].min()),
                int(features['http_count'].max()),
                (int(features['http_count'].min()), int(features['http_count'].max()))
            )
            min_device, max_device = col3.slider(
                "Device Count Range", int(features['device_count'].min()),
                int(features['device_count'].max()),
                (int(features['device_count'].min()), int(features['device_count'].max()))
            )
            status_filter = st.multiselect(
                "Anomaly Status", options=['Normal', 'Suspicious'],
                default=['Normal', 'Suspicious']
            )

        filtered = features[
            (features['logon_count'] >= min_logon) & (features['logon_count'] <= max_logon) &
            (features['http_count'] >= min_http) & (features['http_count'] <= max_http) &
            (features['device_count'] >= min_device) & (features['device_count'] <= max_device) &
            (features['status'].isin(status_filter))
        ]

        st.write(f"### Filtered Users ({len(filtered)})")
        st.dataframe(filtered[['logon_count', 'http_count', 'device_count', 'status']])

        st.write("### Anomaly Status Distribution")
        status_counts = filtered['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(status_counts, values='Count', names='Status',
                         color='Status', color_discrete_map={'Normal': '#82E0AA', 'Suspicious': '#F1948A'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie)

        if not filtered.empty:
            pdf_bytes = generate_pdf(filtered)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="insider_threat_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
