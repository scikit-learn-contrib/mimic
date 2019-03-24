.. title:: User guide : contents

.. _user_guide:

==================================================
User guide: mimic calibration
==================================================

Introduction
------------
mimic calibration is a calibration method for binary classification model.
This method was presented at NYC ML Meetup talk given by Sam Steingold, see [*]_.

Here is how it is implemented.

1. Sort the probabitliy in the ascending. Merge neighbor data points into
   one bin until the number of positive equal to threshold positive at each bin.
   In this initial binning, only the last bin may have number of positive less than threshold positive.
2. Calculate the number of positive rate at each bin. Merge neighbor two bins
   if nPos rate in the left bin is greater than right bin.
3. Keep step 2. until the nPos rate in the increasing order.
4. In this step, we have information at each bin, such as nPos rate, the avg/min/max probability.
   we record those informations in two places. One is `boundary_table`. The other is `calibrated_model`.                       `boundary_table`: it records probability at each bin. The default is recording the avg prob of bin.                         `calibrated_model`: it records all the information of bin, such nPos rate, the avg/min/max probability.
5. The final step is linear interpolation.


Estimator
---------
:class:`_MimicCalibration` is the main class of mimic calibration.

    >>> from mimic import _MimicCalibration

Once imported, you can specify two parameters, `threshold_pos` and `record_history`.
`threshold_pos` is the number of positive in the initial binning.
`record_history`: a boolean flag to record merging-bin history.

    >>> from mimic import _MimicCalibration
    >>> mimicObject = _MimicCalibration(threshold_pos=5, record_history=False)
    >>> mimicObject.fit(X, y)
    >>> y_pred = mimicObject.predict(X)

It also provides ``fit`` and ``predict`` methods.

- ``fit``: it requires two inputs `X` and `y`.
  `X`: 1-d array, the probability from the binary model.
  `y`: 1-d array, binary target, its element is 0 or 1.

- ``predict``: it requires one inputs `pre_calib_prob`.
  `pre_calib_prob`: 1-d array, the probability prediction from the binary model.
  It returns 1-d array, the mimic-calibrated probability.

Reference
----------
.. [*] https://www.youtube.com/watch?v=Cg--SC76I1I
