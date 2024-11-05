import streamlit as st
import pandas as pd
import heartpy as hp
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def hrv():
    # Streamlit app setup
    st.title("HRV Analysis with pyhrv")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with heart rate (bpm) data", type="csv")

    if uploaded_file:
        # Read CSV file, skipping the first two rows
        data = pd.read_csv(uploaded_file, skiprows=2)

        # Display the raw data
        st.write("### Raw Data")
        st.write(data.head())


        # Assuming a column named 'HR (bpm)' in the uploaded CSV file
        if 'HR (bpm)' in data.columns:
            heart_rate_bpm = data['HR (bpm)'].values

            st.write("### BPM")
            # Create a Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=heart_rate_bpm, mode='lines+markers', name="Values"))
            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

            # Convert heart rate from bpm to R-R intervals in milliseconds
            rr_intervals = 60000 / heart_rate_bpm  # in ms

            try:
                # Time-domain HRV metrics
                time_domain_results = td.time_domain(rr_intervals)
                #st.write("Time-domain HRV metrics:")
                #st.json(time_domain_results, expanded=True)

                # Debugging: Check keys in frequency_domain_results
                #st.write("### Time-domain Results Keys")
                #st.write(time_domain_results.keys())  # Show the keys present in the results
                #st.write(time_domain_results)

                if st.toggle("Resultados temporales"):
                    st.write(time_domain_results[20])
                    for i in time_domain_results.keys():
                        st.markdown(f"""
                            **{i}**: {time_domain_results[i]}\n
                            """)




                # Frequency-domain HRV metrics
                frequency_domain_results = fd.welch_psd(rr_intervals, show=False)
                #st.write("Frequency-domain HRV metrics:")
                #st.json(frequency_domain_results, expanded=True)  # Display all results for inspection

                # Debugging: Check keys in frequency_domain_results
                #st.write("### Frequency-domain Results Keys")
                #st.write(frequency_domain_results.keys())  # Show the keys present in the results

                if st.toggle("Resultados en Frecuencia"):
                    st.write(frequency_domain_results[8])
                    for i in frequency_domain_results.keys():
                        st.markdown(f"""
                            **{i}**: {frequency_domain_results[i]}\n
                            """)




            except Exception as e:
                st.error(f"Error processing data: {e}")

        else:
            st.error("CSV file must contain a 'HR (bpm)' column.")
    else:
        st.info("Please upload a CSV file to begin analysis.")
