import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
import tempfile
import os

@st.cache_data
def load_model():
    return joblib.load('model_store/iforest_model.joblib')

def color_status(val):
    return ''  # No background color or styling

def main():
    st.title("Insider Threat Detection Dashboard")
    model = load_model()

    uploaded_file = st.file_uploader("Upload features CSV", type=['csv'])
    st.caption("Upload a CSV file with these columns: user (index), logon_count, http_count, device_count.")

    features = None

    if uploaded_file is not None:
        try:
            features = pd.read_csv(uploaded_file, index_col=0)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
        required_cols = {'logon_count', 'http_count', 'device_count'}
        if features is not None:
            missing_cols = required_cols - set(features.columns)
            if missing_cols:
                st.error(f"Uploaded CSV is missing these required columns: {', '.join(missing_cols)}")
                features = None
    else:
        st.info("No file uploaded. Loading default features.")
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        features_path = os.path.join(root_path, 'features.csv')
        features = pd.read_csv(features_path, index_col=0)

    if features is not None:
        preds = model.predict(features)
        features['anomaly'] = preds
        features['status'] = features['anomaly'].map({1: 'Normal', -1: 'Suspicious'})

        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_logon, max_logon = st.slider(
                    "Logon Count Range",
                    int(features['logon_count'].min()),
                    int(features['logon_count'].max()),
                    (int(features['logon_count'].min()), int(features['logon_count'].max()))
                )
            with col2:
                min_http, max_http = st.slider(
                    "HTTP Count Range",
                    int(features['http_count'].min()),
                    int(features['http_count'].max()),
                    (int(features['http_count'].min()), int(features['http_count'].max()))
                )
            with col3:
                min_device, max_device = st.slider(
                    "Device Count Range",
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
        styled_df = filtered[['logon_count', 'http_count', 'device_count', 'status']].style.applymap(color_status, subset=['status'])
        st.dataframe(styled_df)

        # Activity Counts Summary - native Streamlit bar chart
        st.write("### Activity Counts Summary")
        summary = filtered[['logon_count', 'http_count', 'device_count']].sum()
        st.bar_chart(summary)

        # Anomaly Status Distribution - pie chart with Plotly
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
            st.download_button(
                label="Download PDF",
                data=b'',  # Replace with PDF output if needed
                file_name="insider_threat_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
