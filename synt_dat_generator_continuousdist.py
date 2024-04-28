
##### Synthetic Data Generator #####
# Written by Cam Neese and Audrey Nash


# The project focuses on generating synthetic data to evaluate the performance of our analysis techniques. Synthetic data can be created with certain goals in mind that will help us evaluate specific hypotheses about the role of rate and phase, as well as ESA and different classification algorithms.



# We need a function that will return a dataframe of synthetic data. These are spike trains with 1) Taste IDs, 2) Neuron IDs, 3) Trial IDs, and 4) the actual data itself.


# The parameters:
# 1) Number of Tastes
# 2) Number of Neurons
# 3) Number of Trials per taste per neuron
# 4) Rate - Constant for T1
# 5) alpha - array of scalars for each subsequent taste (i.e. if we have two tastes, R1 = Rate, R2 = alpha[0])
# 6) Phase - 2D array of scalars for each taste. The first column is centers of the Gaussian curves, the second column is the sd's of the curves.
#            Default is random uniform distributed

# The constants:
# 1) Length of each lick interval (100 ms)
length_li = 100
# 2) The number of lick intervals (5 intervals; therefore the overall ms as well)
nLickInts = 5

import pandas as pd
import numpy as np


# The following is an array of beta distribution parameters. These correspond to the different Phase Templates.

file_loc = 'C:/Users/nasha/OneDrive - Florida State University/BertramNeuroProj/Synthetic_Project/Beta_Distributions/beta_parameters_moving.npy'
params = np.load(file_loc)
new_row = [1,1]
beta_parameters = np.insert(params, 0, new_row, axis=0)


def phase_template_matrix_generator(phase, nTastes):
    
    if isinstance(phase, int):
        if phase == 0:
            raise NameError('You cannot select for no spikes to occur. If you are only giving one number for the Phase Code, it must be 1-9. See the Phase Templates.')
        elif phase not in range(len(beta_parameters)):
            raise NameError('If you are only giving one number for the Phase Code, it must be in our distribution set. See the Phase Templates.')
            
        phase_template_matrix = np.full(shape=(nTastes, nLickInts), fill_value=phase)
        
    if isinstance(phase, list):
        
        if len(phase) != nTastes or 0 in phase:
            raise NameError('List of Phase Codes needs to have one entry per taste requested, and cannot have any zeros.')
            
        phase_template_matrix = np.zeros(shape=(nTastes, nLickInts))
        for i in range(len(phase)):
            if phase[i] not in range(len(beta_parameters)):
                raise NameError(f'Acceptable Phase Codes are between 0 and 9. See the Phase Templates. betaparam = {beta_parameters}')
            phase_template_matrix[i,:] = np.repeat(phase[i], nLickInts)
            
    if type(phase) == np.ndarray:
        if phase.shape != (nTastes, nLickInts):
            raise NameError('NumPy Array must be of shape nTastes x nLickIntervals. Usually, the number of lick intervals is 5.')
        for taste in range(nTastes):
            if np.sum(phase[taste,:]) == 0:
                raise NameError('Must have at least one non-zero Phase Code within a taste (row).')
            for i in range(nLickInts):
                if phase[taste, i] not in range(len(beta_parameters)):
                    raise NameError('Acceptable Phase Codes are in beta_parameters. See the Phase Templates.')
        phase_template_matrix = phase
        
    return phase_template_matrix   
    
    
    
# alpha_t is the alpha for this taste
def spike_placer(nTrials, rate, alpha_t, phase_template_matrix, taste):
    
    # Get alphaR
    alphaR = rate*alpha_t
    # Find the total number of spikes to allocate over all spike trains
    nTotalSpikes = alphaR*nLickInts*nTrials
    
    # step 1: Find out how many spikes are in each trial
    nSpikesPerTrial = int(np.floor(nTotalSpikes/nTrials))
    extraSpikes = nTotalSpikes%nTrials
    
    extraSpikesTrials = np.random.choice(a=range(nTrials), size=int(extraSpikes), replace=False)
    
    # Number of spikes in each trial
    trialSpikes = np.repeat(nSpikesPerTrial, nTrials)
    for trial in extraSpikesTrials:
        trialSpikes[trial] += 1
    
    # step 2: Given the number of spikes in a trial, allocate them appropriately
    
    # Pull off phase template for this taste
    phase_codes = phase_template_matrix[taste,:]
    # Find where non-zero phase codes are
    nonzero_li = [i for i in range(nLickInts) if phase_codes[i] != 0]
    n_nonzero = len(nonzero_li)   
    # Calculate the minimum number of spikes in a non-zero phase lick int
    # ASSUMPTION: If there is a 0 phase, the spikes get re-allocated within that trial
    min_spikes_per_nonzero_li = np.floor(nSpikesPerTrial/n_nonzero)
    
    trial_li_spikes = np.zeros(shape=(nTrials, nLickInts))
    
    # Give the minimum number of spikes to all non-zero phase intervals
    for li in nonzero_li:
        trial_li_spikes[:,li] = min_spikes_per_nonzero_li
    
    # Check for extra spikes and select lick intervals for the extra spikes
    for trial in range(nTrials):
        diff = int(trialSpikes[trial] - np.sum(trial_li_spikes[trial,:]))
        if diff > 0:
            trial_li_spikes[trial, np.random.choice(nonzero_li, size=diff, replace=False)] += 1
    
    return trial_li_spikes # 2d array of length nTrials x nLickInts with each entry representing the number of spikes that go in that li





def make_synthetic_data(nTastes, nNeurons, nTrials, rate, alpha, phase=1):
    
    # Check for acceptable rates
    for a in alpha:
        if rate*a < .2:
            raise NameError('An alphaR does not guarantee one spike per spike train. Increase the alpha or the base rate.')
        elif rate*a > 10:
            print('Warning: A large alphaR was given. The spike placement algorithm may have a difficult time. If data generation takes more than a couple of minutes, consider lowering the value of your alphaR.')
    
    # Generate Phase Template Matrix
    phase_template_matrix = phase_template_matrix_generator(phase, nTastes)
    
    
    # Initialize a blank dataframe. We need one row for each combination of taste, neuron, and trial. One column for each of the identifiers, and 500 columns for the spike train.
    synthetic_df = pd.DataFrame(index=range(nTastes*nNeurons*nTrials), columns=['Recording Type', 'Taste', 'Neuron', 'Trial'] + [i for i in range(int(nLickInts*length_li))])

    indexer = 0
    for neuron in range(nNeurons):
        for taste in range(nTastes):
            nSpikes = spike_placer(nTrials, rate, alpha[taste], phase_template_matrix, taste)
            for trial in range(nTrials):
                trial_st = np.zeros(shape=(int(length_li*nLickInts)))
                for lick_interval in range(nLickInts):
                    li_spikes = nSpikes[trial, lick_interval]
                    phase_code_li = int(phase_template_matrix[taste, lick_interval])
                    # Decide the timestamps of the spikes in this lick int
                    refractory_satisfied = False            
                    while not refractory_satisfied:
                        refractory_satisfied = True
                        timestamps = np.sort(np.random.beta(beta_parameters[phase_code_li, 0], beta_parameters[phase_code_li, 1], size=int(li_spikes)))
                        spike_times = [int(np.round(i*length_li)) + (length_li*lick_interval) for i in timestamps]
                        for j in range(len(spike_times)-1):
                            if spike_times[j] == spike_times[j+1] or spike_times[j]+1 == spike_times[j+1] or length_li*nLickInts in spike_times:
                                refractory_satisfied=False
                                    
                        # Note: Check if refractory period is satisfied.
                    # else: pick spike times    
                    
                    for time in spike_times:
                        trial_st[time] += 1
                
                synthetic_df.iloc[indexer, 0] = 'Neuron'
                synthetic_df.iloc[indexer, 1] = taste
                synthetic_df.iloc[indexer, 2] = neuron
                synthetic_df.iloc[indexer, 3] = trial
                synthetic_df.iloc[indexer, 4:] = trial_st
                indexer += 1
            
    return synthetic_df

def MakeList(r1, r2):
    return [item for item in range(r1, r2+1)]

def data_treatment(df, smoothing_window, nTastes, nNeurons, nTrials):
      
    copy_data = df.copy()
    #step one needs to be to remove the first four columns. 
    start_index = copy_data.columns.get_loc('Trial') + 1
    info=pd.DataFrame(copy_data.iloc[:,0:start_index])
    
    #make a df full of zeros to add into the places we need
    zeros=np.zeros(shape=((nTastes*nNeurons*nTrials), (2*smoothing_window)))
    zeros_df = pd.DataFrame(zeros)
    
    #seperate the data from each LI into a seperate dataframe  
    ST_df = copy_data.iloc[:, start_index: ]
    pieces = []
    index_for_splitting = len(ST_df.columns)//nLickInts
    start = 0 
    end = index_for_splitting
    for split in range(nLickInts):
        temp_df = ST_df.iloc[:, start:end]
        part_df = pd.concat([zeros_df, temp_df, zeros_df], axis = 1)
        pieces.append(part_df)
        #pieces.append(temp_df)
        start += index_for_splitting
        end += index_for_splitting
    
    #now put all of our df pieces together and rename columns for consistency
   
    zerodata = pd.concat([pieces[i] for i in range(nLickInts)], axis = 1)
    zerodata.columns = [j for j in range(len(zerodata.columns))]
    padded_data = pd.concat([info, zerodata], axis=1)
    mapping = {padded_data.columns[0]:'Recording Type', padded_data.columns[1]: 'Taste', padded_data.columns[2]:'Neuron', padded_data.columns[3]: 'Trial'}
    padded_data = padded_data.rename(columns=mapping)

    return padded_data
        
        
def empty_nullspace(df, smoothing_window, nTastes, nNeurons, nTrials):
    copy_data = df.copy()
    #step one needs to be to remove the first four columns and save them. 
    start_index = copy_data.columns.get_loc('Trial') + 1
    info=pd.DataFrame(copy_data.iloc[:,0:start_index])

    #now remove the last empty interval and first four col from the spike train df
    end_index = -(2*smoothing_window)    
    ST_df = copy_data.iloc[:, start_index: end_index]
    
    #iterate through to seperate each LI/zeros combo 
    pieces = []
    index_for_splitting = len(ST_df.columns)//nLickInts
    #here fctn is changed from above just slightly to remove the zero-intervals
    #will end up with smoothed spike train at same length as it was before zero padding
    start = (2*smoothing_window)
    end = index_for_splitting
    for split in range(nLickInts):
        temp_df = ST_df.iloc[:, start:end]
        pieces.append(temp_df)
        start += index_for_splitting
        end += index_for_splitting  
        
    #this line adds all lick intervals back together into one df
    plain_st = pd.concat([pieces[i] for i in range(nLickInts)], axis = 1)
    
    #now lets fix our column indexing
    plain_st.columns = [j for j in range(len(plain_st.columns))]
    ST = pd.concat([info, plain_st], axis = 1)
    mapping = {ST.columns[0]:'Recording Type', ST.columns[1]: 'Taste', ST.columns[2]:'Neuron', ST.columns[3]: 'Trial'}
    ST = ST.rename(columns=mapping)
    
    return ST
    
    
    
    
    
    
    
    

        