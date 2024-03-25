#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:38:28 2023

This scritpt is used to clean up the results and focus on long-term dynamics (last 10 time steps before end of simulation)

@author: jbaggio
"""

import os

import pickle
import numpy as np
import pandas as pd


#define function to calculate Gini coefficient
def gini(array):
    """
    Parameters
    ----------
    data : array of node extraction for the whole system
    Returns
    -------
    the gini coefficient of extractions

    """
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))



"""
Result Cleaning for long-term dynamics

"""

#if model is already finalized
os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/ModelOutcome/SobolModelResults')
#load results per pop size
Popmax = [5, 10, 25, 50]     #population size

#load results of the model, note that for 25 and 50 system sizes we simulated the model in batches of X repetitions each, that is why the if-else loop
modres = {}
modrestime = {}
for pn in Popmax:
    if pn > 15:
        for rep_i in range (1,6):
            ending = str(pn) + '-' + str(rep_i)
            name = 'Sobolmodel_results' + ending + '.p'
            popn = 'pop_' + ending
            print(popn)
            try:
                modres[popn] = pickle.load(open(name, 'rb'))
                modres[popn] ['maxtime'] = modres[popn].groupby(['run', 'rep','groupid', 'PoP'])['time'].transform('max')   # calculate maxtime
                modres[popn]['propext'] = modres[popn]['extract'] / modres[popn]['avail_res']      # use extraction as ratio of resources available when extracting
                modres[popn]['propres'] = modres[popn]['Resource'] / (1000 * modres[popn]['PoP'])  # use resource as ratio of initial resources 
                #modres[popn]['net'] = modres[popn]['Network'].str.replace(r"[^a-z]", " ", regex=True)     # eliminate part of the network name related to the size of the networ
                modres[popn].sort_values(by=['run', 'rep'])
                modrestime[popn] = modres[popn]
                modres[popn] = modres[popn][modres[popn]['time'] > (modres[popn]['maxtime'] - 10)]  # filter for the last 10 t-steps before maxtime is reached
                

            except:
                pass
    else:
        ending = str(pn)
        name = 'Sobolmodel_results' + ending + '.p'
        popn = 'pop_' + ending
        print(popn)
        modres[popn] = pickle.load(open(name, 'rb'))
        modres[popn]['maxtime'] = modres[popn].groupby(['run', 'rep','groupid'])['time'].transform('max')   # calculate maxtime
        modres[popn]['propext'] = modres[popn]['extract'] / modres[popn]['avail_res']                       # use extraction as ratio of resources available when extracting
        modres[popn]['propres'] = modres[popn]['Resource'] / (1000 * modres[popn]['PoP'])                   # use resource as ratio of initial resources 
        #modres[popn]['net'] = modres[popn]['Network'].str.replace(r"[^a-z]", " ", regex=True)              # eliminate part of the network name related to the size of the networ
        modres[popn].sort_values(by=['run', 'rep'])
        modrestime[popn] = modres[popn]
        modres[popn] = modres[popn][modres[popn]['time'] > (modres[popn]['maxtime'] - 10)]                  # filter for the last 10 t-steps before maxtime is reached





#now make sure that the number of reps is correct, given that we had to chunk them we have 4 cuncks for pop 25 (25 reps each) and 5 for pop 50 (20 reps each)
modres['pop_25-2']['rep'] = modres['pop_25-2']['rep'] + max(modres['pop_25-1']['rep'])
modres['pop_25-3']['rep'] = modres['pop_25-3']['rep'] + max(modres['pop_25-2']['rep'])
modres['pop_25-4']['rep'] = modres['pop_25-4']['rep'] + max(modres['pop_25-3']['rep'])

modres['pop_50-2']['rep'] = modres['pop_50-2']['rep'] + max(modres['pop_50-1']['rep'])
modres['pop_50-3']['rep'] = modres['pop_50-3']['rep'] + max(modres['pop_50-2']['rep'])
modres['pop_50-4']['rep'] = modres['pop_50-4']['rep'] + max(modres['pop_50-3']['rep'])
modres['pop_50-5']['rep'] = modres['pop_50-5']['rep'] + max(modres['pop_50-4']['rep'])
   
#now make sure that the number of reps is correct, given that we had to chunk them we have 4 cuncks for pop 25 (25 reps each) and 5 for pop 50 (20 reps each)
modrestime['pop_25-2']['rep'] = modrestime['pop_25-2']['rep'] + max(modrestime['pop_25-1']['rep'])
modrestime['pop_25-3']['rep'] = modrestime['pop_25-3']['rep'] + max(modres['pop_25-2']['rep'])
modrestime['pop_25-4']['rep'] = modrestime['pop_25-4']['rep'] + max(modres['pop_25-3']['rep'])

modrestime['pop_50-2']['rep'] = modrestime['pop_50-2']['rep'] + max(modrestime['pop_50-1']['rep'])
modrestime['pop_50-3']['rep'] = modrestime['pop_50-3']['rep'] + max(modrestime['pop_50-2']['rep'])
modrestime['pop_50-4']['rep'] = modrestime['pop_50-4']['rep'] + max(modrestime['pop_50-3']['rep'])
modrestime['pop_50-5']['rep'] = modrestime['pop_50-5']['rep'] + max(modrestime['pop_50-4']['rep'])

#group variables depending on groupid, time-step and parameter combination (run) 
groupres = {}
#grouping columns
group_cols = ['run', 'rep', 'groupid'] #groupid # without group id the aggregation is done at the system level (with 1, 2, 5 and 10 groups per system)
#categorical variables for which we calculate the number in each category for the last 10 time-steps
cate_cols = ['strategy', 'net']
#numeric variables for which we calculate the mean of the last 10 timesteps
num_cols = ['PoP', 'MinRes', 'ResGrowth', 'Needs', 'Var_mag', 'extract', 'Resource', 'propext', 'propres', 'maxtime',
            'tom','sys',  #tom and sys are the actual values
            'sense','arouse', 'ec']
#numeric variable for which the min per group (social cognition) for the last 10 time-steps (done for theoretical reasons and how sc affects sensemaking)
min_cols =['sc']

#now average results  for numeric columns, count for categorical columns and and use the min sc. These are all the long-term dynamics
for key in modres:
    print(key)
    modres_n = modres[key]
    modgroup = modres_n.groupby(group_cols).agg({**{col: 'mean' for col in num_cols}, **{col: lambda x: x.value_counts().to_dict() for col in cate_cols},**{col:'min' for col in min_cols}})
    # Reset the index to transform multi-index levels into columns
    modgroup = modgroup.reset_index()
    groupres[key] = modgroup

    #now recreate the net, time and groupid as columns, and generate the number of individuals within a group with a specific extraction strategy per time-step. 
    #These are already mean values over parameter combinations and groupid.  
for key in groupres:
    print(key)
    # groupres[key]['time'] = groupres[key].index.get_level_values(2)
    # groupres[key]['groupid'] = groupres[key].index.get_level_values(2)
    groupres[key]['net'] = groupres[key]['net'].apply(lambda x: list(x.keys())[0])
    groupres[key]['n_sust'] = groupres[key]['strategy'].apply(lambda x: x.get('sustain', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))
    groupres[key]['n_profit'] = groupres[key]['strategy'].apply(lambda x: x.get('profit', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))
    groupres[key]['n_distr'] = groupres[key]['strategy'].apply(lambda x: x.get('distr', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))
    groupres[key]['n_fair'] = groupres[key]['strategy'].apply(lambda x: x.get('fair', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))

#overall figures comparing pop and network on extraction and arousal, and on strategies chosen
modres_longterm = pd.concat(groupres.values(), ignore_index=True)
modres_longterm['net'] = modres_longterm['net'].str.strip()

#group variables depending on groupid, time-step and parameter combination (run) 
grouprestime = {}
#grouping columns
group_cols = ['run', 'rep'] #groupid # without group id the aggregation is done at the system level (with 1, 2, 5 and 10 groups per system)
#categorical variables for which we calculate the number in each category for the last 10 time-steps
cate_cols = ['strategy', 'net']
#numeric variables for which we calculate the mean of the last 10 timesteps
num_cols = ['PoP', 'MinRes', 'ResGrowth', 'Needs', 'Var_mag', 'extract', 'Resource', 'propext', 'propres', 'maxtime',
            'tom','sys',  #tom and sys are the actual values
            'sense','arouse', 'ec']
#numeric variable for which the min per group (social cognition) for the last 10 time-steps (done for theoretical reasons and how sc affects sensemaking)
min_cols =['sc']

#now average results  for numeric columns, count for categorical columns and and use the min sc. These are all the long-term dynamics
for key in modrestime:
    print(key)
    modres_n = modrestime[key]
    modgroup = modres_n.groupby(group_cols).agg({**{col: 'mean' for col in num_cols}, **{col: lambda x: x.value_counts().to_dict() for col in cate_cols},**{col:'min' for col in min_cols}})
    # Reset the index to transform multi-index levels into columns
    modgroup = modgroup.reset_index()
    grouprestime[key] = modgroup

    #now recreate the net, time and groupid as columns, and generate the number of individuals within a group with a specific extraction strategy per time-step. 
    #These are already mean values over parameter combinations and groupid.  
for key in grouprestime:
    print(key)
    # groupres[key]['time'] = groupres[key].index.get_level_values(2)
    # groupres[key]['groupid'] = groupres[key].index.get_level_values(2)
    groupres[key]['net'] = groupres[key]['net'].apply(lambda x: list(x.keys())[0])
    groupres[key]['n_sust'] = groupres[key]['strategy'].apply(lambda x: x.get('sustain', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))
    groupres[key]['n_profit'] = groupres[key]['strategy'].apply(lambda x: x.get('profit', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))
    groupres[key]['n_distr'] = groupres[key]['strategy'].apply(lambda x: x.get('distr', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))
    groupres[key]['n_fair'] = groupres[key]['strategy'].apply(lambda x: x.get('fair', 0) / (x.get('sustain', 0) + x.get('profit', 0) + x.get('distr', 0) + x.get('fair', 0)))

modres_time = pd.concat(grouprestime.values(), ignore_index=True)
modres_time['net'] = modres_time['net'].str.strip()

#create a net/pop value
modres_longterm['uniqueid'] = modres_longterm['run'].astype(str) +'_' +  modres_longterm['PoP'].astype(str) + '_' +  modres_longterm['rep'].astype(str) + '_' + modres_longterm['groupid'].astype(str)  
modres_longterm['uniqueidSys'] = modres_longterm['run'].astype(str) + '_' + modres_longterm['PoP'].astype(str) +'_' + modres_longterm['rep'].astype(str)

modres_time['uid'] = modres_time['run'].astype(str) + modres_time['PoP'].astype(str) + modres_time['rep']

#save the resulting dataframe for analysis
os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/ModelResults')
pickle.dump(modres_longterm, open('modres_longterm.p', 'wb'))
pickle.dump(modres_time, open('modres_time.p', 'wb'))


    /Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/ModelResults/modrestotround.p
    


