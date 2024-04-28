"""
This file houses functions useful for our analyses in the notebooks we have made.
These functions are written for use on spike trains housed in Pandas Dataframes. These Dataframes should include identifiers about the
taste, neuron, and trial IDs of the spike trains, and the type of data in the recording (neuron or lick)

NOTE: All functions assume that the data starts in the column after the one housing Trial IDs.
"""


import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


"""
Data Accessing
"""

# This function returns one single spike train based on some identifying information. It requires a taste, neuron, and trial ID, and
# assumes you want the neuron data unless otherwise specified.

def get_spike_train(dataFrame, taste, neuron, trial, recording_type = 'Neuron'):
    
    start_index = dataFrame.columns.get_loc('Trial') + 1

    return dataFrame[(dataFrame['Recording Type'] == recording_type) & (dataFrame['Taste'] == taste)
                       & (dataFrame['Neuron'] == neuron) & (dataFrame['Trial'] == trial)].iloc[0,start_index:]


"""
Data Editing
"""

# This function will take in the full-length spike trains and return subsections of them in another dataframe.
# This is written to accomodate common analyses we do. Below is a list of the 'result' parameter options and what they do.
# 'pre-taste': Returns recordings before and including taste administration. Used as a control. 
# 'post-taste': Returns recordings including and after taste administration. This is where the animal experiences the stimulus.
# 'one second': Returns recordings from the first second including and after taste administration. This is where the bulk of the processing #               is thought to happen.

def truncate(dataFrame, result = 'post-taste'):
    
    copy_data = dataFrame.copy()
    
    min_index = copy_data.columns.get_loc('Trial') + 1
    min_time = copy_data.columns[min_index]
    max_time = copy_data.columns[-1]
    max_index = copy_data.columns.get_loc(max_time)
    
    taste_index = int(np.floor((max_index + min_index) / 2))
       
    if result == 'pre-taste':
        copy_data.drop(copy_data.iloc[:, taste_index+1:], inplace=True, axis=1)
        return copy_data
    if result == 'post-taste':
        copy_data.drop(copy_data.iloc[:, min_index:taste_index], inplace=True, axis=1)
        return copy_data
    if result == 'one second':
        post_taste = truncate(dataFrame,'post-taste')
        return truncate(post_taste, 'pre-taste')
            

"""
Data smoothing
"""

# Smoothing a single spike train, likely taking in np.array or pd.series data.
# Code is based off of code found here:
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

def smooth_spike_train(x,window_len=100,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    edit: These additions will then be deleted

    input:
        x: the input signal as np array of size (4000,)
        window_len: the dimension of the smoothing window; should be an integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if window_len<3:
        return x


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    truncate_interval_length = int(np.floor(window_len/2))

    y = y[truncate_interval_length:-truncate_interval_length+1]
    
    if len(y) > len(x):
        y = y[:len(x)]

    return y


# Smoothing a collection of spike trains.

def smooth_all_spike_trains(dataFrame, window_len=100, window='hanning'):

    copy_data = dataFrame.copy()
    
    nSpikeTrains = copy_data.shape[0]
    start_index = copy_data.columns.get_loc('Trial') + 1
    start = copy_data.columns[start_index]
    end = copy_data.columns[-1] + 1
    
    spike_trains = np.array(copy_data.iloc[:,start_index:])
    
    for i in range(nSpikeTrains):
        spike_trains[i,:] = smooth_spike_train(spike_trains[i,:], window_len, window)
        
    smoothed_df = pd.DataFrame(spike_trains, columns = [j for j in range(start,end)])
    
    smoothed_df.insert(0,'Recording Type',np.array(copy_data['Recording Type']))
    smoothed_df.insert(1,'Taste',np.array(copy_data['Taste'].astype(int)))
    smoothed_df.insert(2,'Neuron',np.array(copy_data['Neuron'].astype(int)))
    smoothed_df.insert(3,'Trial',np.array(copy_data['Trial'].astype(int)))

    return smoothed_df


"""
Data Subsampling
"""

def subsample_spike_train(spike_train, subsampling_rate=.1):

    """
    spike_train is a spike train of size (length,), with length = 2000 or 4000
    subsampling_rate is a percentage, the subsampled spike train will have size
    subsampling_rate*length, rounded
    """

    original_length = spike_train.shape[0]
    new_length = int(np.round(original_length*subsampling_rate))

    samples = np.round(np.linspace(1,original_length,new_length)).astype(int)-1

    subsampled_spike_train = spike_train[samples]

    return subsampled_spike_train


# Subsampling a smoothed spike train.

def subsample_all_spike_trains(dataFrame, subsampling_rate=.1):

    """
    subsampling_rate is a percentage, the subsampled spike trains will have size subsampling_rate*length, rounded
    """
    
    copy_data = dataFrame.copy()

    # The index of the column containing the first timestamp
    start_index = copy_data.columns.get_loc('Trial') + 1
    
    start_time = copy_data.columns[start_index]
    end_time = copy_data.columns[-1]
    original_length = end_time - start_time + 1
    
    new_length = int(np.round(original_length*subsampling_rate))
    
    samples = np.round(np.linspace(start_time+1,end_time,new_length)).astype(int)-1

    subsampled_df = copy_data[['Recording Type', 'Taste', 'Neuron', 'Trial'] + [j for j in samples]]

    return subsampled_df


"""
Data Binning
"""

# Binning is a common tool in neuroscience where the number of spikes that occur over a given timespan are summed. 
# This function returns a binned representation of one spike train.

def bin_spike_train(spike_train, bin_width=100):
    
    original_length = spike_train.shape[0]
    
    # What happens if the length of the data doesn't evenly divide by the bin width?
    # The last bin will potentially be cut off.
    new_Dim = int(np.ceil(original_length/bin_width))
    
    binned = np.zeros(shape=new_Dim)
    for interval in range(new_Dim):
        int_start = interval * bin_width

        # If we've run out of room in the original spike train, set the end of the bin to be the end of the spike train
        if (interval+1) * bin_width > original_length:
            int_end = original_length
        else:
            int_end = (interval+1) * bin_width

        binned[interval] = np.sum(spike_train[int_start:int_end])
                
    return binned



def bin_all_spike_trains(dataFrame, bin_width=100):

    copy_data = dataFrame.copy()
    
    nSpikeTrains = copy_data.shape[0]
    start_index = copy_data.columns.get_loc('Trial') + 1
    start = copy_data.columns[start_index]
    end = copy_data.columns[-1] + 1
       
    spike_trains = np.array(copy_data.iloc[:,start_index:])
    binned_spike_trains = np.zeros(shape=(nSpikeTrains, int(np.ceil(spike_trains.shape[1]/bin_width))))
    
    for i in range(nSpikeTrains):
        binned_spike_trains[i,:] = bin_spike_train(spike_trains[i,:], bin_width)
        
    binned_df = pd.DataFrame(binned_spike_trains, columns = [j for j in range(start,end,bin_width)])
    
    binned_df.insert(0,'Recording Type',np.array(copy_data['Recording Type']))
    binned_df.insert(1,'Taste',np.array(copy_data['Taste'].astype(int)))
    binned_df.insert(2,'Neuron',np.array(copy_data['Neuron'].astype(int)))
    binned_df.insert(3,'Trial',np.array(copy_data['Trial'].astype(int)))

    return binned_df


"""
Data Analysis
"""

# Support Vector Machines. 

# This function returns one classifiction rate. X is a numpy array (with each row as a spike train) and y is the taste labels.
def SVM_one_neuron(X, y, test_size = 1/3, num_splits=20):
    
    # Which SVM Optimization problem do we solve?
    n_samples = X.shape[0] * (1-test_size) # Number of spike trains in the training set
    n_features = X.shape[1]  # Number of time points in the spike trains
    dual_param = (n_samples < n_features)

    # Define the SVM model
    model_SVM = LinearSVC(dual=dual_param, max_iter=10000, tol = 0.0001, random_state=651)
    #add a tolerance here - what happens if tol is 0.001 instead of 0.0001?


    split_crs = [] 

    for j in range(num_splits):                          # Use several splits of training and testing sets for robustness

        X_train,X_test,y_train,y_test = train_test_split(X,y,                 # This function is from sklearn
                                                         test_size = test_size, # Default: 2/3 of data to train and 1/3 to test
                                                         shuffle = True,
                                                         stratify = y)        # Sample from each taste

        model_SVM.fit(X_train,y_train)                   # Re-fit the classifier with the training set
        split_crs.append(model_SVM.score(X_test,y_test))  # Fit the testing set and record score

    svm_rate = np.mean(split_crs)  # After scores from each split have been obtained, 
                                                             # record the average

    return svm_rate


# This function returns one classification rate per neuron in the dataframe it is given.

def SVM_all_neurons(dataFrame, test_size = 1/3, num_splits=20):
    
    if 'Trial' in dataFrame.columns:
        start_index = dataFrame.columns.get_loc('Trial') + 1
    else:
        start_index = 0
        
    # This will be the returned array, consisting of one classification rate per neuron
    all_SVM_rates = []

    # Iterate through all neurons
    for neuron in dataFrame['Neuron'].unique():
        
        neuron_df = dataFrame[dataFrame['Neuron']==neuron] # Select all spike trains from this neuron
        X = neuron_df.iloc[:,start_index:]  # X is the data. It has the shape (n_observations, n_times)
        y = np.array(neuron_df['Taste'])                           # y is the labels. We're classifying based on taste.
        
        
        
        """
        # Which SVM Optimization problem do we solve?
        n_samples = X.shape[0] * (1-test_size) # Number of spike trains in the training set
        n_features = X.shape[1]  # Number of time points in the spike trains
        dual_param = (n_samples < n_features)
        
        # Define the SVM model
        model_SVM = LinearSVC(dual=dual_param)
        
        
        raw_full_SVM_class_rates = []                        # Reset this container for each neuron

        for j in range(num_splits):                          # Use several splits of training and testing sets for robustness

            X_train,X_test,y_train,y_test = train_test_split(X,y,                 # This function is from sklearn
                                                             test_size = test_size, # Default: 2/3 of data to train and 1/3 to test
                                                             shuffle = True,
                                                             stratify = y)        # Sample from each taste
            
            model_SVM.fit(X_train,y_train)                   # Re-fit the classifier with the training set
            raw_full_SVM_class_rates.append(model_SVM.score(X_test,y_test))  # Fit the testing set and record score

        all_SVM_rates.append(np.mean(raw_full_SVM_class_rates))  # After scores from each split have been obtained, 
                                                                 # record the average
        """
        all_SVM_rates.append(SVM_one_neuron(X, y, test_size=test_size, num_splits=num_splits))

    return all_SVM_rates


# This function will extract our rate metric given a neuron spike train, a lick spike train, and the number of lick intervals to use:
def extract_rate(neuron_st, lick_st, n_lick_intervals):
    
    # Find when the licks happen
    lickTimes = [i for i in lick_st.index if lick_st[i] == 1]
            
    # Check for sufficient number of lick intervals in the data:
    if len(lickTimes) >= n_lick_intervals + 1:

        # Rate is easy; just sum up the number of fires and divide by the number of lick intervals:
        R = np.sum(neuron_st.loc[lickTimes[0] : lickTimes[n_lick_intervals]-1])/n_lick_intervals
        
    else:
        # There weren't enough lick intervals, so we're setting Rate to -1.
        R = -1
        
    return R

    
# This function will extract our phase metric.
def extract_phase(neuron_st, lick_st, n_lick_intervals):
    
    # Find when the licks happen
    lickTimes = [i for i in lick_st.index if lick_st[i] == 1]
            
    # Check for a sufficent number of lick intervals in the data:
    if len(lickTimes) >= n_lick_intervals + 1:

        # Find out if there are any fires over this stretch:
        n_fires = np.sum(neuron_st.loc[lickTimes[0] : lickTimes[n_lick_intervals]-1])

        if n_fires > 0:
            # Here we will put all the phases of the fires in these intervals
            P_list = []            
            for interval in range(n_lick_intervals):
                first_lick = lickTimes[interval]
                second_lick = lickTimes[interval+1]
                P_lick_int = [((fire - first_lick)/(second_lick - first_lick)) for fire in range(first_lick,second_lick) if neuron_st.loc[fire] == 1]
                for i in P_lick_int:
                    P_list.append(i)

            # P_list now contains the phase of each spike. We will average over all spikes, weighting each spike evenly.
            P = np.mean(P_list)
        else:
            # If there were no fires
            P = 0
    else:
        # If there weren't enough lick intervals
        P = -1
        
    return P

"""
Rate-Only, Phase-Only, and Combination Analyses

First, we define functions to calculate Separation Scores, an observational analysis. Then we define functions for predictive analyses.
"""

# This function will return a separation score using only one metric as the basis. This function is intended to do Rate-Only and 
# Phase-Only separation on one neuron and taste pair at a time.

# Pass in:
    # neuron_tp_df: A Pandas dataframe having at least columns for 'Taste' and whatever the metric of interest is
    # metric: A string corresponding to the metric of interest. Usually 'R_Norm' or 'P' for our purposes.
    
    # optional
    # min_trials_per_taste: The lowest number of trials in each taste you want to consider. Any taste pair failing to meet this threshold
            # for both tastes will automatically be assigned a score of 0.
    # nLines: The number of coordinates you want to try. The higher this number is, the more likely you are to find the absolute best line.

def one_metric_separation_score(neuron_tp_df, metric, min_trials_per_taste = 3, nLines = 100):
    
    tp = neuron_tp_df['Taste'].unique()
    if len(tp) == 2:
        t1 = tp[0]
        t2 = tp[1]
    
        # Find if there are a sufficient number of trials from each taste to perform the analysis
        t1_trials = neuron_tp_df[neuron_tp_df['Taste']==t1].shape[0]
        t2_trials = neuron_tp_df[neuron_tp_df['Taste']==t2].shape[0]

        if t1_trials >= min_trials_per_taste and t2_trials >= min_trials_per_taste:

            points = neuron_tp_df[['Taste', metric]]
            
            # Create lines to test
            lines = np.linspace(0,1,nLines)  # Each axis goes from 0 to 1

            points_label1 = np.array(neuron_tp_df[neuron_tp_df['Taste'] == t1][metric])
            points_label2 = np.array(neuron_tp_df[neuron_tp_df['Taste'] == t2][metric])

            # This will contain one score per line we test.
            line_scores = []
            for line in lines:

                # There are two scenarios.

                # Scenario 1: Points labeled 0 are mostly to the 'left' of the line, and points labeled 1 are to the 'right'
                xleft0 = [x for x in points_label1 if x < line]
                xright0 = [x for x in points_label2 if x > line]
                x_score0 = (len(xleft0) + len(xright0))/points.shape[0]

                # Scenario 2: Points labeled 0 are mostly to the 'right' of the line, and points labeled 1 are to the 'left'
                xleft1 = [x for x in points_label1 if x > line]
                xright1 = [x for x in points_label2 if x < line]
                x_score1 = (len(xleft1) + len(xright1))/points.shape[0]

                # Find out which of the scenarios performed better
                x_score = max([x_score0,x_score1])
                line_scores.append(x_score)

                # Now we can figure out which line was the best in the training set and punch it into the testing set

            # Sort the scores to find which line was best
            metricScore = max(line_scores)
            bestXLine_index = np.argsort(line_scores)[::-1][0]
            bestXLine = lines[bestXLine_index]

        else:
            # Scenario where there weren't enough trials
            metricScore = 0
            bestXLine = 0
    else:
        # Scenario where at least one taste had no trials
        metricScore = 0
        bestXLine = 0
        
    return metricScore, bestXLine


# This function determines how many radii are necessary to test based on a given angle. This is used in the combination score functions and
# speeds up computation time.
def max_R(theta):
    if -np.pi/2 <= theta and theta <= 0:
        R = np.cos(theta)
    elif 0 <= theta and theta <= np.pi/2:
        R = np.cos(theta) + np.sin(theta)
    elif np.pi/2 <= theta and theta <= np.pi:
        R = np.sin(theta)
    return R




def combo_separation_score(neuron_tp_df, metric1, metric2, min_trials_per_taste = 3, n_Thetas = 135, n_Radii = 150):

    tp = neuron_tp_df['Taste'].unique()
    
    if len(tp) == 2:
        t1 = tp[0]
        t2 = tp[1]

        # Find if there are a sufficient number of trials from each taste to perform the analysis
        t1_trials = neuron_tp_df[neuron_tp_df['Taste']==t1].shape[0]
        t2_trials = neuron_tp_df[neuron_tp_df['Taste']==t2].shape[0]

        if t1_trials >= min_trials_per_taste and t2_trials >= min_trials_per_taste:


            thetas = np.linspace(-np.pi/2, np.pi, n_Thetas) # Lines corresponding to normal vectors in Quadrant 4 are not relevant

            # Pull off only relevant data for this analysis
            points = neuron_tp_df[['Taste', metric1, metric2]]

            # Reset arrays to put all test scores and normal vectors into
            all_scores = []
            all_normal_vectors = np.zeros(shape=(n_Thetas*n_Radii, 2))

            # Now our points have been separated into a training set and testing set. Let's split up the training set by taste.
            points_label1 = np.array(points[points['Taste'] == t1][[metric1, metric2]])
            points_label2 = np.array(points[points['Taste'] == t2][[metric1, metric2]])

            # Iterate over all normal vectors (combinations of theta and R)
            i = 0
            for theta in thetas:
                mR = max_R(theta)
                rs = np.linspace(0,mR,int(round(mR/.01)))
                for r in rs:

                    # Here we test a line.

                    # Calculate the coordinates of the normal vector
                    n = np.array([r*np.cos(theta),r*np.sin(theta)])
                    # Store the normal vector information
                    all_normal_vectors[i,:] = [r,theta]
                    i = i+1

                    # There are two scenarios.

                    # Scenario 1: Points labeled 0 are mostly to the 'left' of the line, and points labeled 1 are to the 'right'
                    left0 = [p for p in points_label1 if np.matmul((p - n), n.T) < 0]
                    right0 = [p for p in points_label2 if np.matmul((p - n), n.T) > 0]
                    score0 = (len(left0) + len(right0))/points.shape[0]

                    # Scenario 2: Points labeled 0 are mostly to the 'right' of the line, and points labeled 1 are to the 'left'
                    left1 = [p for p in points_label1 if np.matmul((p - n), n.T) > 0]
                    right1 = [p for p in points_label2 if np.matmul((p - n), n.T) < 0]
                    score1 = (len(left1) + len(right1))/points.shape[0]

                    # Find out which scenario was correct
                    score = max([score0,score1])
                    scenario = np.argsort([score0, score1])[::-1][0]
                    all_scores.append(score)

            # End of Testing All Lines

            # Sort the scores and figure out which line was the best
            comboScore = max(all_scores)
            bestLine = np.argsort(all_scores)[::-1][0]
            # Record the r and theta information for the best line
            best_r = all_normal_vectors[bestLine,0]
            best_theta = all_normal_vectors[bestLine,1]

        else:
            # There weren't enough trials
            comboScore = 0
            best_r = 0
            best_theta = 0
    else:
        # There were no trials of at least one taste
        comboScore = 0
        best_r = 0
        best_theta = 0

    return comboScore, best_r, best_theta





    # n_splits: The number of train-test splits to average over.
    # test_size: The relative size of the testing set.
def one_metric_predict_score(neuron_tp_df, metric, min_trials_per_taste = 3, nLines = 100, n_splits = 5, test_size = 1/3):
    
    tp = neuron_tp_df['Taste'].unique()
    if len(tp) == 2:
        t1 = tp[0]
        t2 = tp[1]
    
        # Find if there are a sufficient number of trials from each taste to perform the analysis
        t1_trials = neuron_tp_df[neuron_tp_df['Taste']==t1].shape[0]
        t2_trials = neuron_tp_df[neuron_tp_df['Taste']==t2].shape[0]

        if t1_trials >= min_trials_per_taste and t2_trials >= min_trials_per_taste:

            # Pull off only relevant data for this analysis
            points = neuron_tp_df[['Taste', metric]]

            # Create lines to test
            lines = np.linspace(0,1,nLines)  # Each axis goes from 0 to 1

            # Record one score for each split. This will house those scores.
            splits_x_scores = []
            for j in range(n_splits):

                # Separate the data into training and testing sets.
                X_train,X_test = train_test_split(points,
                                                  test_size = test_size,     # Use 2/3 of data to train and 1/3 to test as default
                                                  shuffle = True,
                                                  stratify = points['Taste'])        # Sample proportionally from each taste

                # Now our points have been separated into a training set and testing set. Split the training set up by taste.
                x_points_label1 = np.array(X_train[X_train['Taste'] == t1][metric])
                x_points_label2 = np.array(X_train[X_train['Taste'] == t2][metric])

                # This will contain one score per line we test.
                line_scores = []
                for line in lines:

                    # There are two scenarios.

                    # Scenario 1: Points labeled 0 are mostly to the 'left' of the line, and points labeled 1 are to the 'right'
                    xleft0 = [x for x in x_points_label1 if x < line]
                    xright0 = [x for x in x_points_label2 if x > line]
                    x_score0 = (len(xleft0) + len(xright0))/X_train.shape[0]

                    # Scenario 2: Points labeled 0 are mostly to the 'right' of the line, and points labeled 1 are to the 'left'
                    xleft1 = [x for x in x_points_label1 if x > line]
                    xright1 = [x for x in x_points_label2 if x < line]
                    x_score1 = (len(xleft1) + len(xright1))/X_train.shape[0]

                    # Find out which of the scenarios performed better
                    x_score = max([x_score0,x_score1])
                    x_scenario = np.argsort([x_score0, x_score1])[::-1][0]
                    line_scores.append(x_score)

                # Now we can figure out which line was the best in the training set and punch it into the testing set

                # Sort the scores to find which line was best
                bestXLine = np.argsort(line_scores)[::-1][0]
                test_x_line = lines[bestXLine]

                x_points_label1 = np.array(X_test[X_test['Taste'] == t1][metric])
                x_points_label2 = np.array(X_test[X_test['Taste'] == t2][metric])

                if x_scenario == 0:
                    # Scenario 1: Points labeled 0 are mostly to the 'left' of the line, and points labeled 1 are to the 'right'
                    xleft0 = [x for x in x_points_label1 if x < test_x_line]
                    xright0 = [x for x in x_points_label2 if x > test_x_line]
                    x_score = (len(xleft0) + len(xright0))/X_test.shape[0]
                elif x_scenario == 1:
                    # Scenario 2: Points labeled 0 are mostly to the 'right' of the line, and points labeled 1 are to the 'left'
                    xleft1 = [x for x in x_points_label1 if x > test_x_line]
                    xright1 = [x for x in x_points_label2 if x < test_x_line]
                    x_score = (len(xleft1) + len(xright1))/X_test.shape[0]

                # Record how the testing set performed
                splits_x_scores.append(x_score)   

            # Record the mean scores from the testing sets
            metricScore = np.mean(splits_x_scores)

        else:
            # Scenario where there weren't enough trials
            metricScore = 0
    else:
        # Scenario where at least one taste had no trials
        metricScore = 0
        
    return metricScore
    


    
    
def combo_predict_score(neuron_tp_df, metric1, metric2, min_trials_per_taste = 3, n_Thetas = 135, n_Radii = 150, n_splits = 5, test_size = 1/3):

    tp = neuron_tp_df['Taste'].unique()
    
    if len(tp) == 2:
        t1 = tp[0]
        t2 = tp[1]

        # Find if there are a sufficient number of trials from each taste to perform the analysis
        t1_trials = neuron_tp_df[neuron_tp_df['Taste']==t1].shape[0]
        t2_trials = neuron_tp_df[neuron_tp_df['Taste']==t2].shape[0]

        if t1_trials >= min_trials_per_taste and t2_trials >= min_trials_per_taste:


            thetas = np.linspace(-np.pi/2, np.pi, n_Thetas) # Lines corresponding to normal vectors in Quadrant 4 are not relevant

            # Pull off only relevant data for this analysis
            points = neuron_tp_df[['Taste', metric1, metric2]]

            splits_scores = []
            for j in range(n_splits):
                # Reset arrays to put all test scores and normal vectors into
                all_scores = []
                all_normal_vectors = np.zeros(shape=(n_Thetas*n_Radii, 2))

                # Separate the data into training and testing sets.
                X_train,X_test = train_test_split(points,
                                                  test_size = test_size,     # Use 2/3 of data to train and 1/3 to test as default
                                                  shuffle = True,
                                                  stratify = points['Taste'])        # Sample proportionally from each taste

                # Now our points have been separated into a training set and testing set. Let's split up the training set by taste.
                points_label1 = np.array(X_train[X_train['Taste'] == t1][[metric1, metric2]])
                points_label2 = np.array(X_train[X_train['Taste'] == t2][[metric1, metric2]])

                # Iterate over all normal vectors (combinations of theta and R)
                i = 0
                for theta in thetas:
                    mR = max_R(theta)
                    rs = np.linspace(0,mR,int(round(mR/.01)))
                    for r in rs:

                        # Here we test a line.

                        # Calculate the coordinates of the normal vector
                        n = np.array([r*np.cos(theta),r*np.sin(theta)])
                        # Store the normal vector information
                        all_normal_vectors[i,:] = [r,theta]
                        i = i+1

                        # There are two scenarios.

                        # Scenario 1: Points labeled 0 are mostly to the 'left' of the line, and points labeled 1 are to the 'right'
                        left0 = [p for p in points_label1 if np.matmul((p - n), n.T) < 0]
                        right0 = [p for p in points_label2 if np.matmul((p - n), n.T) > 0]
                        score0 = (len(left0) + len(right0))/X_train.shape[0]

                        # Scenario 2: Points labeled 0 are mostly to the 'right' of the line, and points labeled 1 are to the 'left'
                        left1 = [p for p in points_label1 if np.matmul((p - n), n.T) > 0]
                        right1 = [p for p in points_label2 if np.matmul((p - n), n.T) < 0]
                        score1 = (len(left1) + len(right1))/X_train.shape[0]

                        # Find out which scenario was correct
                        score = max([score0,score1])
                        scenario = np.argsort([score0, score1])[::-1][0]
                        all_scores.append(score)

                # End of Testing All Lines

                # Sort the scores and figure out which line was the best
                bestLine = np.argsort(all_scores)[::-1][0]
                # Record the r and theta information for the best line
                split_best_r = all_normal_vectors[bestLine,0]
                split_best_theta = all_normal_vectors[bestLine,1]

                # Here's the winning normal vector:
                n = np.array([split_best_r*np.cos(split_best_theta),split_best_r*np.sin(split_best_theta)])

                # Try this r and theta on the testing set
                # Separate the testing set out based on their taste
                points_label1 = np.array(X_test[X_test['Taste'] == t1][[metric1, metric2]])
                points_label2 = np.array(X_test[X_test['Taste'] == t2][[metric1, metric2]])

                # There are two scenarios.

                if scenario == 0:
                    # Scenario 1: Points labeled 0 are mostly to the 'left' of the line, and points labeled 1 are to the 'right'
                    left0 = [p for p in points_label1 if np.matmul((p - n), n.T) < 0]
                    right0 = [p for p in points_label2 if np.matmul((p - n), n.T) > 0]
                    score = (len(left0) + len(right0))/X_test.shape[0]
                elif scenario == 1:
                    # Scenario 2: Points labeled 0 are mostly to the 'right' of the line, and points labeled 1 are to the 'left'
                    left1 = [p for p in points_label1 if np.matmul((p - n), n.T) > 0]
                    right1 = [p for p in points_label2 if np.matmul((p - n), n.T) < 0]
                    score = (len(left1) + len(right1))/X_test.shape[0]

                splits_scores.append(score)

                # End of One Split

            # End Of All Splits

            # Record the average classification rate across all splits
            comboScore = np.mean(splits_scores)    

        else:
            # There weren't enough trials
            comboScore = 0
    else:
        # There were no trials of at least one taste
        comboScore = 0

    return comboScore












"""
Visualizing Data
"""

# This function is useful for plotting two columns on a dataframe, colored by taste. This is particularly useful for plotting rate and 
# phase metrics.

def scatter_df(dataFrame, x_metric, y_metric, tastes):
    if 0 in tastes:
        taste_df = dataFrame[(dataFrame['Taste'] == 0)]
        plt.scatter(taste_df[x_metric], taste_df[y_metric], color='red', alpha=0.5,label='0')
    if 1 in tastes:
        taste_df = dataFrame[(dataFrame['Taste'] == 1)]
        plt.scatter(taste_df[x_metric], taste_df[y_metric], color='blue', alpha=0.5,label='1')
    if 2 in tastes:
        taste_df = dataFrame[(dataFrame['Taste'] == 2)]
        plt.scatter(taste_df[x_metric], taste_df[y_metric], color='green', alpha=0.5,label='2')
    if 3 in tastes:
        taste_df = dataFrame[(dataFrame['Taste'] == 3)]
        plt.scatter(taste_df[x_metric], taste_df[y_metric], color='black', alpha=0.5,label='3')
    if 4 in tastes:
        taste_df = dataFrame[(dataFrame['Taste'] == 4)]
        plt.scatter(taste_df[x_metric], taste_df[y_metric], color='turquoise', alpha=1,label='4')

        
def raster_plot(data_frame, neuron_ID, lick_ints=5, tastes=range(5)):
    
    # Pull off only observations from the neuron in question
    spike_trains = data_frame[data_frame['Neuron'] == neuron_ID]
    
    timestamps = []
    taste_labels = []
    
    taste_colors = ['red', 'blue', 'green', 'black', 'turquoise']
    
    for taste in tastes:
        # Pull off neuron spikes and lick spikes
        nt_df = spike_trains[(spike_trains['Taste'] == taste) & (spike_trains['Recording Type'] == 'Neuron')]
        lt_df = spike_trains[(spike_trains['Taste'] == taste) & (spike_trains['Recording Type'] == 'Lick')]
        
        for trial in nt_df['Trial']:
            neuron_st = get_spike_train(nt_df, taste, neuron_ID, trial)
            lick_st = get_spike_train(lt_df, taste, neuron_ID, trial, 'Lick')
            if np.sum(lick_st) > lick_ints:
                lick_times = [time for time in lick_st.index if lick_st[time] == 1]
                adj_fire_times = []
                for lick_int in range(lick_ints):
                    start = lick_times[lick_int]
                    end = lick_times[lick_int+1]
                    fire_times = [time for time in range(start,end) if neuron_st[time] == 1]
                    for i in fire_times:
                        adj_fire_times.append(lick_int + ((i-start)/(end-start)))
                        
                timestamps.append(adj_fire_times)
                taste_labels.append(taste)
                        
    timestamps = timestamps
    y_coords = np.linspace(start=.15, stop=1, num=len(timestamps))
    for i, s in enumerate(timestamps):
        x_c = s
        y_c = np.repeat(y_coords[i], len(s))
        
        plt.scatter(x_c, y_c, color=taste_colors[taste_labels[i]], marker='|')
        
    y_coords = np.linspace(start=0,stop=.13, num=15)
    for i in y_coords:
        y_c = np.repeat(i, lick_ints+1)
        plt.scatter(range(lick_ints + 1), y_c, color='darkorange', marker='|')
        
        
        

