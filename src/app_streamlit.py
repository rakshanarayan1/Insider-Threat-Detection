import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from fpdf import FPDF
import os

# Feature Engineering Helper
def run_feature_engineering_from_files(logon_file, http_file, device_file):
    # logon and device: default headers
    logon_counts = pd.read_csv(logon_file)['user'].value_counts().rename('logon_count') if logon_file else pd.Series(dtype=int)
    device_counts = pd.read_csv(device_file)['user'].value_counts().rename('device_count') if device_file else pd.Series(dtype=int)
    # http file: manually set columns
    http_cols = ['id', 'date', 'user', 'pc', 'url']
    http_counts = pd.read_csv(http_file, header=None, names=http_cols)['user'].value_counts().rename('http_count') if http_file else pd.Series(dtype=int)
    # Merge counts
    features = pd.concat([logon_counts, http_counts, device_counts], axis=1).fillna(0).astype(int)
    features.index.name = "user"
    return features

@st.cache_data
def load_model():
    return joblib.load('model_store/iforest_model.joblib')

def color_status(val):
    return ''  # No styling for now

def generate_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Insider Threat Detection Report", ln=True, align='C')
    pdf.ln(10)
    # Table header
    headers = ["User", "logon_count", "http_count", "device_count", "status"]
    for h in headers:
        pdf.cell(38, 10, h, border=1)
    pdf.ln()
    # Table rows (first 50 rows)
    for user, row in df.head(50).iterrows():
        pdf.cell(38, 10, str(user), border=1)
        pdf.cell(38, 10, str(row["logon_count"]), border=1)
        pdf.cell(38, 10, str(row["http_count"]), border=1)
        pdf.cell(38, 10, str(row["device_count"]), border=1)
        pdf.cell(38, 10, row["status"], border=1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin1')

def main():
    st.title("Insider Threat Detection Dashboard")
    # st.write("### **How to use:**")
    # st.markdown(
    #     "- **Option 1:** Upload a single structured features CSV\n"
    #     "  - Columns: `logon_count`, `http_count`, `device_count`\n"
    #     "  - Index column should be the user name/ID\n"
    #     "- **Option 2:** Upload three raw log files at once:\n"
    #     "  - `logon.csv` (must have a `user` column)\n"
    #     "  - `http.csv` (parsed as ['id','date','user','pc','url'])\n"
    #     "  - `device.csv` (must have a `user` column)\n"
    #     "- **If you upload nothing, demo data from `features.csv` in your repo will be used."
    # )

    uploaded_files = st.file_uploader(
        "Upload features.csv OR all three raw logs (logon.csv, http.csv, device.csv):",
        type=['csv'],
        accept_multiple_files=True
    )

    features = None

    # Determine which upload path to follow
    if uploaded_files:
        if len(uploaded_files) == 1 and 'feature' in uploaded_files[0].name.lower():
            st.success("Structured features file detected.")
            features = pd.read_csv(uploaded_files[0], index_col=0)
            required_cols = {'logon_count', 'http_count', 'device_count'}
            if set(features.columns) < required_cols:
                st.error(f"CSV missing required columns: {', '.join(required_cols - set(features.columns))}")
                features = None
        else:
            files_dict = {f.name.lower(): f for f in uploaded_files}
            logon_file = files_dict.get('logon.csv')
            http_file = files_dict.get('http.csv')
            device_file = files_dict.get('device.csv')
            if not (logon_file and http_file and device_file):
                st.warning("If uploading raw data, you must provide all three: logon.csv, http.csv, device.csv.")
            else:
                st.success("Raw log files detected: running feature engineering...")
                features = run_feature_engineering_from_files(logon_file, http_file, device_file)
    else:
        st.info("No upload detected, using default features.csv from repo.")
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        features_path = os.path.join(root_path, 'features.csv')
        features = pd.read_csv(features_path, index_col=0)

    if features is not None:
        model = load_model()
        preds = model.predict(features)
        features['anomaly'] = preds
        features['status'] = features['anomaly'].map({1:'Normal', -1:'Suspicious'})

        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            min_logon, max_logon = col1.slider(
                "Logon Count",
                int(features['logon_count'].min()), int(features['logon_count'].max()),
                (int(features['logon_count'].min()), int(features['logon_count'].max()))
            )
            min_http, max_http = col2.slider(
                "HTTP Count",
                int(features['http_count'].min()), int(features['http_count'].max()),
                (int(features['http_count'].min()), int(features['http_count'].max()))
            )
            min_device, max_device = col3.slider(
                "Device Count",
                int(features['device_count'].min()), int(features['device_count'].max()),
                (int(features['device_count'].min()), int(features['device_count'].max()))
            )
            status_filter = st.multiselect(
                "Anomaly Status", options=['Normal', 'Suspicious'],
                default=['Normal', 'Suspicious']
            )
        # Filtered Data
        filtered = features[
            (features['logon_count'] >= min_logon) & (features['logon_count'] <= max_logon) &
            (features['http_count'] >= min_http) & (features['http_count'] <= max_http) &
            (features['device_count'] >= min_device) & (features['device_count'] <= max_device) &
            (features['status'].isin(status_filter))
        ]

        st.write(f"### Filtered Users ({len(filtered)})")
        st.dataframe(filtered[['logon_count','http_count','device_count','status']])

        # Bar Chart
        st.write("### Activity Counts Summary")
        summary = filtered[['logon_count', 'http_count', 'device_count']].sum()
        st.bar_chart(summary)

        # Pie Chart
        st.write("### Anomaly Status Distribution")
        status_counts = filtered['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(
            status_counts,
            values='Count', names='Status',
            color='Status', color_discrete_map={'Normal': '#82E0AA', 'Suspicious': '#F1948A'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent')
        fig_pie.update_layout(width=450, height=450, title_text=None, showlegend=True)
        st.plotly_chart(fig_pie, key="pie_chart")

        # PDF Download
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
