# mimic cailbration
It is a calibration on the probability prediction from binary classification model. 
This method was mentioned at [NYC ML Meetup talk given by Sam Steingold](https://www.youtube.com/watch?v=Cg--SC76I1I).

## Introduction:

### Implementation:
It requires two inputs, the probability prediction from binary classification model and the binary target (0 and 1).                                                                                                  
 Here is how it is implemented.
1. Sort the probabitliy in the ascending. Merge neighbor data points into one bin until the number of positive equal to threshold positive at each bin. In this initial binning, only the last bin may have number of positive less than threshold positive.
2. Calculate the number of positive rate at each bin. Merge neighbor two bins if nPos rate in the left bin is greater than right bin.                                                                                                 
3. Keep step 2. until the nPos rate in the increasing order.                                                                  
4. In this step, we have information at each bin, such as nPos rate, the avg/min/max probability.                             
we record those informations in two places. One is `boundary_table`. The other is `calibrated_model`.                          
`boundary_table`: it records probability at each bin. The default is recording the avg prob of bin.                          
`calibrated_model`: it records all the information of bin, such nPos rate, the avg/min/max probability.                      
The final step is linear interpolation on the prediction. Given probability prediction from binary classification model, we find out which bin it belongs to. Then, we perform linear interpolation in the corresponding bin.

### Parameters:
```python
_MimicCalibration(threshold_pos, record_history)
```
* threshold_pos: the number of positive in the initial binning. default = 5.
* record_history: boolean parameter, decide if record all the mergeing of bin history. default = False.

## How to use it

```python
from mimic_calibration import _MimicCalibration
mimicObject = _MimicCalibration(threshold_pos=5, record_history=True)
# y_calib_score: probability prediction from binary classification model
# y_calib: the binary target, 0 or 1.
mimicObject.fit(y_calib_score, y_calib)
y_mimic_calib_score = mimicObject.predict(y_calib_score)
```

## Results:
### mimic-calibrated-probability
![Image of mimic calibrated prob](https://github.com/pinjutien/mimic/blob/master/data/mimic_calib_prob.png)

### History of merging bins.
![Image of binning history](https://github.com/pinjutien/mimic/blob/master/data/merging_bins_history.png)
