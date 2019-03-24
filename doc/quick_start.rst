#####################################
Quick Start with the sklearn-mimic
#####################################


1. Quick start
-------------------------------------

To use mimic calibration, please git clone the following repository::

    $ git clone https://github.com/pinjutien/mimic.git

This calibration method is for binary classification model.
It requires the probabity prediction of binary model (`X`) and the binary target (`y`).
Both `X` and `y` are 1-d array.
    >>> from mimic import _MimicCalibration
    >>> mimicObject = _MimicCalibration(threshold_pos=5, record_history=False)
    >>> mimicObject.fit(X, y)
    >>> y_pred = mimicObject.predict(X)

