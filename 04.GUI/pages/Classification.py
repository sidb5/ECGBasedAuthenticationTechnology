import shutil
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import wfdb

from feature_extraction import (
    non_fid_for_plot,
    non_fiducial_features_bonus_plots,
    non_fiducial_features_bonus_plots2,
    points_for_plot,
    processing,
)
from final_project import (
    Fiducial_Features,
    non_fiducial_features,
    non_fiducial_features_bonus_preprocessing,
)

st.set_page_config(
    page_title="ECG Based Authentication Interface",
    page_icon=":star:",
    layout="wide",
)
st.write("## Classification")

BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_DIR = BASE_DIR / "temp"


def load_model(filename):
    with open(BASE_DIR / filename, "rb") as model_file:
        return pickle.load(model_file)


def read_data(path):
    fs = 1000
    patient = wfdb.rdrecord(path, channels=[1])
    signal = patient.p_signal[:, 0]
    time = len(signal) / fs
    return signal, time, patient


def get_features(signal, index):
    if index == 0:
        return Fiducial_Features([signal])
    if index == 1:
        return non_fiducial_features([signal])
    return non_fiducial_features_bonus_preprocessing([signal])


non_fid_model = load_model("random_forest_classifier_nonFid.pkl")
non_fid_bonus_model = load_model("random_forest_classifier_nonFidBonus.pkl")
fid_model = load_model("random_forest_classifier_Fid.pkl")

uploaded_files = st.file_uploader("Choose Signal: ", accept_multiple_files=True)

signal_name = ""
signal = None

if uploaded_files:
    TEMP_DIR.mkdir(exist_ok=True)
    for uploaded_file in uploaded_files:
        signal_name = uploaded_file.name.split(".")[0]
        with open(TEMP_DIR / uploaded_file.name, "wb") as handle:
            handle.write(uploaded_file.getbuffer())

    if signal_name:
        signal, _, record = read_data(str(TEMP_DIR / signal_name))
        fig = wfdb.plot_wfdb(record=record, figsize=(200, 20), title="Record")
        st.pyplot(fig)
        st.write("Done Reading files!")

methods = {"Fiducial": 0, "Non Fiducial": 1, "Non Fiducial Bonus": 2}
selected_feature_extraction_method = st.radio(
    "Select Feature Extraction Method: ",
    ["Fiducial", "Non Fiducial", "Non Fiducial Bonus"],
)
index = methods[selected_feature_extraction_method]
st.write("Selected Feature Extraction Method: ", selected_feature_extraction_method)

processed_signal = processing(signal) if signal is not None else None

if st.button("Predict"):
    if processed_signal is None:
        st.warning("Upload a complete WFDB record before running prediction.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        if index == 0:
            _, values = get_features(processed_signal, index)
            values = np.array(values).reshape(-1, 23)
            prediction = fid_model.predict_proba(values[:, :22])
            threshold_percentage = 0.8
            flag = 0
            for row in prediction:
                for subject_id, percentage in enumerate(row):
                    if percentage >= threshold_percentage:
                        st.write(
                            f"Identified as subject {subject_id + 1} with {percentage * 100}% certainty."
                        )
                        flag = 1
                if flag == 1:
                    break
            if flag == 0:
                st.write("subject is undefind")

        elif index == 1:
            _, values = get_features(processed_signal, index)
            values = np.array(values).reshape(1, 81)
            prediction = non_fid_model.predict_proba(values[:, :80])
            threshold_percentage = 0.5
            flag = 0
            for row in prediction:
                for subject_id, percentage in enumerate(row):
                    if percentage >= threshold_percentage:
                        st.write(
                            f"Identified as subject {subject_id + 1} with {percentage * 100}% certainty."
                        )
                        flag = 1
            if flag == 0:
                st.write("subject is undefind")

        else:
            _, values = get_features(processed_signal, index)
            values = np.array(values).reshape(-1, 41)
            prediction = non_fid_bonus_model.predict_proba(values[:, :40])
            threshold_percentage = 0.95
            flag = 0
            for row in prediction:
                for subject_id, percentage in enumerate(row):
                    if percentage >= threshold_percentage:
                        st.write(
                            f"Identified as subject {subject_id + 1} with {percentage * 100}% certainty."
                        )
                        flag = 1
                if flag == 1:
                    break
            if flag == 0:
                st.write("subject is undefind")

    with col2:
        if index == 0:
            res = points_for_plot(processed_signal, start=500, end=2000)
            fs = 1000
            time = len(res["denoised_signal"]) / fs
            ts = np.arange(0, time, 1.0 / fs)
            fig, ax = plt.subplots()
            ax.plot(ts, res["denoised_signal"], alpha=0.6, lw=1, label="Raw signal")
            ax.scatter(res["qx"] / fs, res["qy"], alpha=0.5, color="red", label="Q point")
            ax.scatter(
                res["sx"] / fs, res["sy"], alpha=0.5, color="green", label="S point"
            )
            ax.scatter(res["Rx"] / fs, res["Ry"], alpha=0.5, color="blue", label="R point")
            ax.scatter(
                res["qrs_on_x"] / fs,
                res["qrs_on_y"],
                alpha=0.5,
                color="black",
                label="QRS onset",
            )
            ax.scatter(
                res["qrs_off_x"] / fs,
                res["qrs_off_y"],
                alpha=0.5,
                color="yellow",
                label="QRS offset",
            )
            ax.scatter(
                res["Px"] / fs, res["Py"], alpha=0.5, color="orange", label="P point"
            )
            ax.scatter(
                res["p_on_x"] / fs,
                res["p_on_y"],
                alpha=0.5,
                color="pink",
                label="P onset",
            )
            ax.scatter(
                res["p_off_x"] / fs,
                res["p_off_y"],
                alpha=0.5,
                color="cyan",
                label="P offset",
            )
            ax.scatter(
                res["Tx"] / fs, res["Ty"], alpha=0.5, color="purple", label="T point"
            )
            ax.scatter(
                res["t_on_x"] / fs,
                res["t_on_y"],
                alpha=0.5,
                color="brown",
                label="T onset",
            )
            ax.scatter(
                res["t_off_x"] / fs,
                res["t_off_y"],
                alpha=0.5,
                color="gray",
                label="T offset",
            )
            ax.set_title("Fiducial Points")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (mV)")
            ax.legend(prop={"size": 7}, loc="upper right")
            st.pyplot(fig)

        elif index == 1:
            components = non_fid_for_plot(processed_signal)
            fig, ax = plt.subplots(5)
            for row in range(5):
                ax[row].plot(components[row])
                ax[row].set_title("")
            for axis in fig.get_axes():
                axis.label_outer()
            fig.tight_layout(h_pad=0)
            st.pyplot(fig)

        else:
            wavelet_features = non_fiducial_features_bonus_plots(processed_signal)
            beat_segments = non_fiducial_features_bonus_plots2(processed_signal)
            fig, ax = plt.subplots(2)
            ax[0].plot(beat_segments[0])
            ax[0].set_title("")
            ax[1].plot(wavelet_features[0])
            ax[1].set_title("")
            for axis in fig.get_axes():
                axis.label_outer()
            fig.tight_layout(h_pad=0)
            st.pyplot(fig)

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
