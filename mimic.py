import numpy as np
import pandas as pd
from time import time


class mimic(object):

    def __init__(self, score_target_df, number_positive_within_bin = 5):
        t0 = time()
        self.df = score_target_df[['score', 'target']].sort_values(['score'])
        print("Sorted time: {x} s".format(x =(time() - t0)))
        self.sorted_score = self.df['score'].values
        self.sorted_target = self.df['target'].values
        assert(set(self.sorted_target) == set([0,1]))
        self.threshold_pos = number_positive_within_bin

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
        # from copy import deepcopy
        # initial_binning
        # [[bl_index, score_min, score_max, score_mean, nPos_temp, total_temp, ctr_temp]]
        # current_binning = deepcopy(initial_binning)
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
            
    def calibrate(self):
        t0 = time()
        initial_binning = self.construct_initial_bin(self.sorted_score,
                                                     self.sorted_target,
                                                     self.threshold_pos)
        print("Initialize binning time: {x} s".format(x =(time() - t0)))

        t0 = time()
        final_binning = self.run_merge_function(initial_binning, record_history = False)
        print("Merge binning time: {x} s".format(x =(time() - t0)))
        
        return final_binning

    
if __name__ == '__main__':

    # testing example
    import random
    random.seed(10)
    num_rows = 10000
    score = [random.random() for i in range(num_rows)]

    random.seed(20)
    threshold = 0.90
    target = [1 if random.random() >= threshold else 0 for i in range(num_rows)]
    print("Number of positive: {x}".format(x = sum(target)))
    score_target_df = pd.DataFrame(data = zip(score, target), columns =["score", "target"])

    # mimic function
    res = mimic(score_target_df).calibrate()
