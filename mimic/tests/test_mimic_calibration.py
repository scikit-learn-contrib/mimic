from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
# import sys
# sys.path.append('../')
import unittest
from mimic import _MimicCalibration
import pytest


def test_mimic_example():
    n_samples = 1000
    X, y = make_classification(n_samples=3 * n_samples, n_features=6,
                               random_state=42)
    X -= X.min()  # MultinomialNB only allows positive X
    # split train and test
    X_train, y_train = X[:n_samples], y[:n_samples]
    X, y = X[n_samples:2 * n_samples], y[n_samples:2 * n_samples]
    clf = MultinomialNB().fit(X_train, y_train)
    y_prob = clf.predict_proba(X)
    y_pre_calib_prob = np.array([y[1] for y in y_prob])
    mimic_obj = _MimicCalibration(threshold_pos=5, record_history=False)

    # X: probability prediction from MultinomialNB
    # y: binary target, [0, 1]
    X = y_pre_calib_prob
    # use mimi calibration to calibrate X
    mimic_obj.fit(X, y)
    # y_calib_prob: the mimic-calibrated probaility
    y_calib_prob = mimic_obj.predict(X)
    assert(y_calib_prob.shape[0] == X.shape[0]), \
        "The length of calibrated prob must be the same as \
        pre-calibrated prob."


def test_mimic_history_plots():
    import numpy as np
    n_samples = 1000
    X, y = make_classification(n_samples=3 * n_samples, n_features=6,random_state=42)
    X -= X.min()

    # Train data: train binary model.
    X_train, y_train = X[:n_samples], y[:n_samples]
    # calibrate data.
    X_calib, y_calib = X[n_samples:2 * n_samples], y[n_samples:2 * n_samples]
    # test data.
    X_test, y_test = X[2 * n_samples:], y[2 * n_samples:]
    clf = MultinomialNB().fit(X_train, y_train)

    # y_calib_score: training in the calibration model.
    y_calib_score = clf.predict_proba(X_calib)
    y_calib_score = np.array([score[1] for score in y_calib_score])

    # y_test_score: evaluation in the calibration model.
    y_test_score = clf.predict_proba(X_test)
    y_test_score = np.array([score[1] for score in y_test_score])
    
    mimicObject = _MimicCalibration(threshold_pos=5, record_history=True)
    mimicObject.fit(y_calib_score, y_calib)
    y_mimic_score = mimicObject.predict(y_test_score)
    history = mimicObject.history_record_table
    mimicObject.output_history_result([0, 5, 19])

def test_mimic_output():
    import pandas as pd
    tolerance = 1e-6
    df = pd.read_csv("mimic/tests/data_set.csv")
    X = df["probability"].values
    y = df["y"].values
    y_mimic = df["mimic_probability"].values
    mimicObject = _MimicCalibration(threshold_pos=5, record_history=True)
    mimicObject.fit(X, y)
    pred = mimicObject.predict(X)
    error = abs(y_mimic - pred)/y_mimic
    pass_flag = (error < tolerance).all()
    assert(pass_flag), "The numerical error is greater than {x}".format(x = tolerance)
