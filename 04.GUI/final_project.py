import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from feature_extraction import (
    Fiducial_Points_Detection,
    non_fiducial_features_bonus,
    nonFiducial,
    processing,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

GUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUI_DIR.parent


SUBJECT_RECORDS = [
    "01.Dataset/104/s0306lre",
    "01.Dataset/117/s0291lre",
    "01.Dataset/122/s0312lre",
    "01.Dataset/252/s0487_re",
    "01.Dataset/173/s0305lre",
    "01.Dataset/182/s0308lre",
    "01.Dataset/234/s0460_re",
    "01.Dataset/238/s0466_re",
    "01.Dataset/255/s0491_re",
    "01.Dataset/166/s0275lre",
]


def load_subjects():
    subjects = []
    for record_path in SUBJECT_RECORDS:
        patient = wfdb.rdrecord(str(REPO_ROOT / record_path), channels=[1])
        subjects.append(processing(patient.p_signal[:, 0]))
    return subjects


def _concatenate_feature_sets(feature_sets):
    if not feature_sets:
        return pd.DataFrame({})
    return np.concatenate(feature_sets, axis=0)


def Fiducial_Features(signals):
    fid_features = []
    fid_features_values = []
    for index, signal in enumerate(signals, start=1):
        fid_feature = Fiducial_Points_Detection(signal)
        fid_feature["class"] = index
        fid_features_values.append(fid_feature.values)
        fid_features.append(fid_feature)
    return fid_features, fid_features_values


def create_data_frame_fid(fid_features_values):
    df = _concatenate_feature_sets(fid_features_values)
    np.save("fiducial_feature.npy", df)
    return df


def random_forest_classifier(X_train, y_train, filename):
    random_forest_classifier_object = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    random_forest_classifier_object.fit(X_train, y_train)
    pickle.dump(random_forest_classifier_object, open(filename, "wb"))


def SVM_classifier(X_train, y_train, filename):
    svm_classifier_object = SVC(kernel="linear")
    svm_classifier_object.fit(X_train, y_train)
    pickle.dump(svm_classifier_object, open(filename, "wb"))


def LR_classifier(X_train, y_train, filename):
    lr_classifier_object = LogisticRegression(random_state=0)
    lr_classifier_object.fit(X_train, y_train)
    pickle.dump(lr_classifier_object, open(filename, "wb"))


def data_frame_non(non_fid, label):
    df = pd.DataFrame({})
    for i in range(len(non_fid)):
        df[i] = non_fid[i]
    df["class"] = label
    return df


def non_fiducial_features(signals):
    non_fid_features = []
    non_fid_features_values = []
    for index, signal in enumerate(signals, start=1):
        non_fid_feature = nonFiducial(signal).reshape(80, 1)
        non_fid_feature = data_frame_non(non_fid_feature, index)
        non_fid_features_values.append(non_fid_feature.values)
        non_fid_features.append(non_fid_feature)
    return non_fid_features, non_fid_features_values


def create_data_frame_non_fid(non_fid_features_values):
    df = _concatenate_feature_sets(non_fid_features_values)
    np.save("non_fiducial_feature.npy", df)
    df.to_csv("Non_feaducial_feature.csv", index=False)
    return df


def data_frame(non_fid, label):
    df = pd.DataFrame({})
    for i in range(len(non_fid)):
        df[i] = non_fid[i]
    df = df.T
    df["class"] = label
    return df


def non_fiducial_features_bonus_preprocessing(signals):
    non_fid_features = []
    non_fid_features_values = []
    for index, signal in enumerate(signals, start=1):
        non_fid_feature = non_fiducial_features_bonus(signal)
        non_fid_feature = data_frame(non_fid_feature, index)
        non_fid_features_values.append(non_fid_feature.values)
        non_fid_features.append(non_fid_feature)
    return non_fid_features, non_fid_features_values


def create_data_frame_non_fid_bonus(non_fid_features_values):
    df = _concatenate_feature_sets(non_fid_features_values)
    np.save("bouns_feature.npy", df)
    df.to_csv("bouns_feature.csv", index=False)
    return df
