import numpy as np
import pandas as pd
from time import time
import sys

class mimic(object):

    def __init__(self, score_target_df, number_positive_within_bin = 5):
        t0 = time()
        self.df = score_target_df[['score', 'target']].sort_values(['score'])
        print("Sorted time: {x} s".format(x =(time() - t0)))
        self.sorted_score = self.df['score'].values
        self.sorted_target = self.df['target'].values
        assert(set(self.sorted_target) == set([0,1]))
        self.threshold_pos = number_positive_within_bin
        self.calibrated_model = None
        self.boundary_table = []

    def construct_initial_bin(self, sorted_score, sorted_target, threshold_pos):

        # 1st step: make each bin having 5 positive.
        bin_right_index_array = []
        count = 0
        # 0 or 1 in sorted_target
        for i in range(len(sorted_target)):
            y = sorted_target[i]
            
            if y > 0:
                count += 1
            
            if (count == threshold_pos):
                bin_right_index_array += [i]
                count = 0

        if (len(sorted_target)-1 not in bin_right_index_array):
            bin_right_index_array += [len(sorted_target)-1]
        
        bl_index = 0
        bin_info = []
        test_pos = 0
        
        for br_index in bin_right_index_array:
            # score stats
            score_temp = sorted_score[bl_index: br_index+ 1]
            score_min = min(score_temp)
            score_max = max(score_temp)
            score_mean = np.mean(score_temp)
            # target
            target_temp = sorted_target[bl_index: br_index+ 1]
            nPos_temp = np.sum(target_temp)
            total_temp = len(target_temp)
            ctr_temp = 1.0*nPos_temp/total_temp

            bin_info += [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]
            test_pos += nPos_temp
            bl_index = br_index+ 1

        print("Test Pos: {x}".format(x = test_pos))
        return bin_info

    def merge_bins(self, binning_input, increasing_flag):
        # binning_input
        # [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]
        nbins = len(binning_input)
        result = []
        
        for i in range(1, nbins):
            # current_bin: latest new bin in the result
            if (i == 1):
                result += [binning_input[0]]
            
            current_bin = result[-1]
            current_bin_ctr = current_bin[-1]
            
            next_bin = binning_input[i]
            next_bin_ctr = next_bin[-1]
            
            if(current_bin_ctr > next_bin_ctr):
                increasing_flag = False
                # merge two bins:
                # [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]
                new_bin_index_temp = min(current_bin[0], next_bin[0])
                new_score_min_temp = min(current_bin[1], next_bin[1])
                new_score_max_temp = max(current_bin[2], next_bin[2])
                new_score_mean_temp = (current_bin[3] + next_bin[3])/2.0
                new_pos_temp = current_bin[4] + next_bin[4]
                new_total_temp = current_bin[5] + next_bin[5]
                new_ctr_temp = 1.0*new_pos_temp/new_total_temp

                # update the latest bin info in the latest result
                result[-1] = [new_bin_index_temp, new_score_min_temp, new_score_max_temp,
                              new_score_mean_temp, new_pos_temp, new_total_temp, new_ctr_temp]
            else:
                result += [next_bin]
            
        return result, increasing_flag

    def run_merge_function(self, current_binning, record_history = False):
        # current_binning
        # [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]

        history_binning = []
        if (record_history):
            history_binning += [current_binning]

        keep_merge = True
        while(keep_merge):
            new_bin_temp, increasing_flag = self.merge_bins(current_binning, True)
            if (record_history):
                history_binning += [new_bin_temp]

            # update the current_binning
            current_binning = new_bin_temp
            # if it increasing monotonically, we stop merge
            keep_merge = not increasing_flag

        if (record_history):
            return history_binning

        return [new_bin_temp]


    def to_dataFrame(self,binning_input, detail= False):
        # [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]
        df = pd.DataFrame(data = binning_input, columns= ["bin_index", "raw_score_l", "raw_score_r", "score_mean", "nPos", "bin_size", "predict_ctr"])
        if (detail):
            return df
        return df[["raw_score_l", "predict_ctr"]]

    def get_bin_boundary(self, current_binning, boundary_choice = 2):
        """
        current_binning:

        [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]

        boundary_choice:
        0: choose socre_min, ie left boundary of bin
        1: choose socre_max, ie right boundary of bin
        2: choose socre_mean, ie mean score of bin

        """
        num_rows = len(current_binning)
        
        boundary_table_temp = []
        
        k = 2
        
        if (boundary_choice == 0):
            k = 1
            
        if (boundary_choice == 2):
            k = 3
            
        for i in range(num_rows):
            boundary_table_temp += [current_binning[i][k]]
            
        return boundary_table_temp

    
    def calibrate(self):
        t0 = time()
        initial_binning = self.construct_initial_bin(self.sorted_score,
                                                     self.sorted_target,
                                                     self.threshold_pos)
        print("Initialize binning time: {x} s".format(x =(time() - t0)))
        print("Number of bins at Initial step: {x}".format(x = len(initial_binning)))

        t0 = time()
        final_binning = self.run_merge_function(initial_binning, record_history = False)
        print("Merge binning time: {x} s".format(x =(time() - t0)))
        latest_bin_temp = final_binning[-1]
        print("Number of bins in the end: {x}".format(x = len(latest_bin_temp)))
        self.calibrated_model = latest_bin_temp
        self.boundary_table = self.get_bin_boundary(latest_bin_temp, boundary_choice = 2)
        df = self.to_dataFrame(latest_bin_temp, detail= False)
        return df


    def predict(self, x, debug= False):
        
        """
        x: a raw score, ie pre-calibrated score.

        calibrated_model:
        [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]
        """
        
        if((self.calibrated_model is None) & (debug is False)):
            sys.exit("Please calibrate model first by calling calibrate function.")
        else:
            # linear interpolation
            which_bin = np.digitize([x], self.boundary_table, right = True)[0]

            if ((which_bin == 0) or (which_bin == len(self.boundary_table)-1)):
                y = self.calibrated_model[which_bin][6]
            else:
                delta_y = self.calibrated_model[which_bin][6] - self.calibrated_model[which_bin-1][6]
                delta_x = self.boundary_table[which_bin] - self.boundary_table[which_bin-1]

                y = self.calibrated_model[which_bin-1][6] + \
                    (1.0*delta_y/delta_x) * (x - self.boundary_table[which_bin-1])
                
            return y
                

def test_construct_initial_bin():
    df = pd.read_csv("test-input-1.csv")
    sorted_score = df.score.values
    sorted_target = df.target.values
    threshold_pos = 5
    bin_info = mimic(df).construct_initial_bin(sorted_score, sorted_target, threshold_pos)
    test_df = pd.DataFrame(data = bin_info, columns = ["left_index", "score_min", "score_max", "score_mean", "target", "total", "ctr"])
    benchmark_df = pd.read_csv("benchmark-1.csv")
    ep = 1e-5
    test_1 = abs((benchmark_df.ctr.sum() - test_df.ctr.sum())/benchmark_df.ctr.sum()) < ep
    assert(test_1), "Test Fail."

    
if __name__ == '__main__':

    # test_construct_initial_bin()
    # testing example
    import random
    random.seed(10)
    num_rows = 10000
    score = [random.random() for i in range(num_rows)]

    random.seed(20)
    threshold = 0.90
    target = [1 if random.random() >= threshold else 0 for i in range(num_rows)]
    print("Number of positive: {x}".format(x = sum(target)))
    score_target_df = pd.DataFrame(data = list(zip(score, target)), columns =["score", "target"])

    # mimic function
    
    mimic_model = mimic(score_target_df)
    mimic_model.calibrate()
    raw_score = 0.2
    calibrated_score = mimic_model.predict(raw_score)
    print(calibrated_score)

