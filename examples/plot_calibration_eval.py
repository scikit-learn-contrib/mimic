"""
===========================
Plotting Calibration curve
===========================
Compare mimic and isotonic calibration.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss,
                             precision_score,
                             recall_score,
                             f1_score)
from sklearn.svm import LinearSVC
from copy import copy
import sys
# sys.path.append('../')
from mimic.mimic_calibration import _MimicCalibration
# plt.rcParams["figure.figsize"] = (20,10)


def calibration_comparison(base_estimator,
                           n_samples,
                           weights=None,
                           n_bins=10,
                           detail=False):

    X, y = make_classification(n_samples=3*n_samples,
                               n_features=6,
                               random_state=42,
                               weights=weights)
    base_estimator_dict = {
        "MultinomialNB": MultinomialNB(),
        "GaussianNB": GaussianNB(),
        "SVC": LinearSVC()
    }

    if (base_estimator == "MultinomialNB"):
        X -= X.min()
    # Train data: train binary model.
    X_train, y_train = X[:n_samples], y[:n_samples]
    print("Positive Rate: {x}".format(x=y_train.mean()))
    # calibrate data.
    X_calib, y_calib = X[n_samples:2 * n_samples], y[n_samples:2 * n_samples]
    # test data.
    X_test, y_test = X[2 * n_samples:], y[2 * n_samples:]
    # train the base estimator
    clf = base_estimator_dict[base_estimator].fit(X_train, y_train)

    if (base_estimator == "SVC"):
        # y_calib_score: training in the calibration model.
        y_calib_score = clf.decision_function(X_calib)
        y_calib_score = (y_calib_score - y_calib_score.min()) /\
                        (y_calib_score.max() - y_calib_score.min())
        # y_test_score: evaluation in the calibration model.
        y_test_score = clf.decision_function(X_test)
        y_test_score = (y_test_score - y_test_score.min()) /\
                       (y_test_score.max() - y_test_score.min())
    else:
        # y_calib_score: training in the calibration model.
        y_calib_score = clf.predict_proba(X_calib)
        y_calib_score = np.array([score[1] for score in y_calib_score])

        # y_test_score: evaluation in the calibration model.
        y_test_score = clf.predict_proba(X_test)
        y_test_score = np.array([score[1] for score in y_test_score])

    calibrate_model_dict = {
        "mimic": _MimicCalibration(threshold_pos=5, record_history=False),
        "isotonic": IsotonicRegression(y_min=0.0,
                                       y_max=1.0,
                                       out_of_bounds='clip'),
        # "platt": LogisticRegression()
    }

    result = {}
    result[base_estimator] = {}
    for cal_name, cal_object in calibrate_model_dict.items():
        # import pdb; pdb.set_trace()
        print(cal_name)
        cal_object.fit(copy(y_calib_score), copy(y_calib))
        if cal_name in ["mimic", "isotonic"]:
            y_output_score = cal_object.predict(copy(y_test_score))
        else:
            raise "Please specify probability prediction function."

        frac_pos, predicted_value = calibration_curve(
            y_test,
            y_output_score,
            n_bins=n_bins)
        b_score = brier_score_loss(y_test, y_output_score, pos_label=1)
        # precsion = precision_score(y_test, y_output_score)
        # recall = recall_score(y_test, y_output_score)
        # f1 = f1_score(y_test, y_output_score)

        result[base_estimator][cal_name] = {
            "calibration_curve": [frac_pos, predicted_value],
            # "eval_score" : [b_score, precsion, recall, f1]
            "eval_score": [b_score]
        }

        if (detail):
            result[base_estimator][cal_name]["detail"] = {
                "y_test": y_test,
                "y_test_calibrate_score": y_output_score
            }

    return result


def show_comparison_plots(base_estimator,
                          n_samples,
                          weights=None,
                          n_bins=10,
                          detail=False):
    res = calibration_comparison(base_estimator,
                                 n_samples,
                                 weights,
                                 n_bins,
                                 detail)
    all_calibration_methods = res[base_estimator]
    all_calibration_methods_names = list(all_calibration_methods.keys())
    color_map = {
        "isotonic": 'orangered',
        "mimic": 'limegreen'}

    eval_df = []
    fig = plt.figure()
    for i, calib_name in enumerate(all_calibration_methods_names):
        calib_model = all_calibration_methods[calib_name]
        frac_pos, predicted_value = calib_model["calibration_curve"]
        b_score = all_calibration_methods[calib_name]["eval_score"][0]
        if (i == 0):
            plt.plot(frac_pos,
                     frac_pos,
                     color='grey',
                     label="perfect-calibration",
                     alpha=0.3,
                     linewidth=5)

        plt.plot(predicted_value,
                 frac_pos,
                 color=color_map[calib_name],
                 label="%s: %1.4f" % (calib_name, b_score),
                 alpha=0.7, linewidth=5)

    plt.legend(fontsize=20)
    plt.xlabel("calibrated probability", fontsize=18)
    plt.ylabel("fraction_of_positives", fontsize=18)
    plt.show()


show_comparison_plots("GaussianNB", 10000, None, 10)
