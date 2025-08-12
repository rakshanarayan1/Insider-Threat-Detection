import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from fpdf import FPDF
import os
import tempfile

@st.cache_data
def load_model():
    return joblib.load('model_store/iforest_model.joblib')

def color_status(val):
    return ''  # No styling; you can add colors here if you prefer.

def generate_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Insider Threat Detection Report", ln=True, align='C')
    pdf.ln(10)
    # Table header
    headers = ["User", "logon_count", "http_count", "device_count", "status"]
    for header in headers:
        pdf.cell(38, 10, header, border=1)
    pdf.ln()
    # Table rows (show only first 50 users in PDF)
    for user, row in df.head(50).iterrows():
        pdf.cell(38, 10, str(user), border=1)
        pdf.cell(38, 10, str(row["logon_count"]), border=1)
        pdf.cell(38, 10, str(row["http_count"]), border=1)
        pdf.cell(38, 10, str(row["device_count"]), border=1)
        pdf.cell(38, 10, str(row["status"]), border=1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin1')

def main():
    st.title("Insider Threat Detection Dashboard")

    st.write("**Upload a structured table CSV (feature summary):**")
    st.caption("The file should have index column as user, and columns: logon_count, http_count, device_count.")

    uploaded_file = st.file_uploader("Upload features CSV", type=['csv'])

    features = None
    if uploaded_file is not None:
        try:
            features = pd.read_csv(uploaded_file, index_col=0)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            features = None
        required_cols = {'logon_count', 'http_count', 'device_count'}
        if features is not None and not required_cols.issubset(features.columns):
            st.error(f"CSV is missing columns: {', '.join(required_cols - set(features.columns))}")
            features = None
    else:
        st.info("No file uploaded. Using default features table.")
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        features_path = os.path.join(root_path, 'features.csv')
        features = pd.read_csv(features_path, index_col=0)

    if features is not None:
        model = load_model()
        preds = model.predict(features)
        features['anomaly'] = preds
        features['status'] = features['anomaly'].map({1: 'Normal', -1: 'Suspicious'})

        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            min_logon, max_logon = col1.slider(
                "Logon Count",
                int(features['logon_count'].min()),
                int(features['logon_count'].max()),
                (int(features['logon_count'].min()), int(features['logon_count'].max()))
            )
            min_http, max_http = col2.slider(
                "HTTP Count",
                int(features['http_count'].min()),
                int(features['http_count'].max()),
                (int(features['http_count'].min()), int(features['http_count'].max()))
            )
            min_device, max_device = col3.slider(
                "Device Count",
                int(features['device_count'].min()),
                int(features['device_count'].max()),
                (int(features['device_count'].min()), int(features['device_count'].max()))
            )
            status_filter = st.multiselect(
                "Anomaly Status",
                options=['Normal', 'Suspicious'],
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

        st.write("### Activity Counts Summary")
        summary = filtered[['logon_count', 'http_count', 'device_count']].sum()
        st.bar_chart(summary)

        st.write("### Anomaly Status Distribution")
        status_counts = filtered['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(
            status_counts,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map={'Normal': '#82E0AA', 'Suspicious': '#F1948A'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent')
        fig_pie.update_layout(width=450, height=450, title_text=None, showlegend=True)
        st.plotly_chart(fig_pie, key="pie_chart")

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
