#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:34:31 2023

This scritp analyzes the results from Stress_Cog_Mem_Sobol_CleanDataResults

@author: jbaggio
"""

import os

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


"""

Figures for the combining arousal, cognitive metrics, population, network type 
and outcomes (resources, extraction, time to collapse)

"""
os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/ModelResults')

modlt = pickle.load(open('modres_longterm.p', 'rb'))
modlt['maxtime'] = modlt['maxtime'] / 100 # so instead of having it in interval 0,100 we have it in interval 0,1
modlt_orig = modlt.copy()

modlt = modlt.drop(columns = ['strategy', 'uniqueid'])

group_cols = ['uniqueidSys'] #, #groupid # without group id the aggregation is done at the system level (with 1, 2, 5 and 10 groups per system)
#categorical variables for which we calculate the number in each category for the last 10 time-steps
cate_cols = ['net']
#numeric variables for which we calculate the mean of the last 10 timesteps
num_cols = ['PoP', 'MinRes', 'ResGrowth', 'Needs', 'Var_mag', 
            'extract', 'Resource', 'propext', 'propres', 'maxtime',
            'tom', 'sys', 'sense', 'arouse', 'ec', 'net', 'sc', 
            'n_sust','n_profit', 'n_distr', 'n_fair', 'run']

modlt = modlt.groupby(group_cols).agg({**{col: 'mean' for col in num_cols}, **{col: lambda x: x.value_counts().to_dict() for col in cate_cols}})
modlt['net'] = modlt['net'].apply(lambda x: list(x.keys())[0])
modlt['uid'] = modlt.index #get column index
#system level as maxtime is calculated at the system level)

#choose the specific metrics
aggmet = 'mean' #or 'mean', 'count', 'median' etc.. gemoetric mean is more robust to skewed distributions.

#bin variables of interest as we are interested in overall patterns here.
cutlist = np.arange(0, 1.05, 0.1)
modlt['arousecat'] = pd.cut(modlt.arouse, bins = cutlist)
modlt['tomcat'] = pd.cut(modlt.tom, bins = cutlist)
modlt['syscat'] = pd.cut(modlt.sys, bins = cutlist)
modlt['sensecat'] = pd.cut(modlt.sense, bins = cutlist)
modlt['sccat'] = pd.cut(modlt.sc, bins = cutlist)
modlt['eccat'] = pd.cut(modlt.ec, bins = cutlist)

modlt['sustcat'] = pd.cut(modlt.n_sust, bins = cutlist)
modlt['profitcat'] = pd.cut(modlt.n_profit, bins = cutlist)
modlt['distrcat'] = pd.cut(modlt.n_distr, bins = cutlist)
modlt['faircat'] = pd.cut(modlt.n_fair, bins = cutlist)
modlt['extcat'] = pd.cut(modlt.propext, bins = cutlist)

# change directory to where figures will be stored
os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/Figures/Together')
sns.set(context='paper', style='white', palette='colorblind', font='Helvetica', font_scale=2)


dftcar = pd.pivot_table(modlt.reset_index(), index=['tomcat'], columns =['arousecat'], values='sc',aggfunc = aggmet)
dftcae = pd.pivot_table(modlt.reset_index(), index=['syscat'], columns =['arousecat'], values='ec',aggfunc = aggmet)
dftcat = pd.pivot_table(modlt.reset_index(), index=['sccat'], columns =['eccat'], values='sense',aggfunc = aggmet)


f10, (ax1, ax2, ax3) = plt.subplots(1, 3,  sharey=False, sharex=False, figsize=(20,20))
sns.heatmap(dftcar, cmap='coolwarm', vmin = 0, vmax = 1,  ax = ax1)
ax1.set(xlabel = 'Arousal', 
        ylabel = 'ToM')
ax1.invert_yaxis()
ax1.set(title = 'Social Cognition')
sns.heatmap(dftcae, cmap='coolwarm', vmin = 0, vmax = 1, ax = ax2)
ax2.set(xlabel = 'Arousal', 
        ylabel = 'Systematizing')
ax2.invert_yaxis()
ax2.set(title = 'Environmental Cognition')
sns.heatmap(dftcat, cmap='coolwarm', vmin = 0, vmax = 1, ax = ax3)
ax3.set(xlabel = 'System Cognition', 
        ylabel = 'Social Cognition')
ax3.invert_yaxis()
ax3.set(title = 'Sensemaking')
f10.tight_layout()
f10.savefig('Cog_Sense-Arouse2.pdf', bbox_inches='tight') 
plt.close()


dftsar = pd.pivot_table(modlt.reset_index(), index=['extcat'], columns =['sustcat'], values='arouse',aggfunc = aggmet)
dftsae = pd.pivot_table(modlt.reset_index(), index=['extcat'], columns =['faircat'], values='arouse',aggfunc = aggmet)
dftsat = pd.pivot_table(modlt.reset_index(), index=['extcat'], columns =['distrcat'], values='arouse',aggfunc = aggmet)
dftsas = pd.pivot_table(modlt.reset_index(), index=['extcat'], columns =['profitcat'], values='arouse',aggfunc = aggmet)


f101, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(20,20))
sns.heatmap(dftsar, cmap='coolwarm', ax = ax1, vmin = 0.4, vmax = 0.9)
ax1.set(xlabel = 'Sustainability Strategy', 
        ylabel = 'Extraction')
ax1.invert_yaxis()
ax1.set(title = 'Arousal')
sns.heatmap(dftsae, cmap='coolwarm', ax = ax2, vmin = 0.4, vmax = 0.9)
ax2.set(xlabel = 'Fairness Strategy', 
        ylabel = 'Extraction')
ax2.invert_yaxis()
ax2.set(title = 'Arousal')
sns.heatmap(dftsat, cmap='coolwarm', ax = ax3, vmin = 0.4, vmax = 0.9)
ax3.set(xlabel = 'Equality Strategy', 
        ylabel = 'Extraction')
ax3.invert_yaxis()
ax3.set(title = 'Arousal')
sns.heatmap(dftsas, cmap='coolwarm', ax = ax4, vmin = 0.4, vmax = 0.9)
ax4.set(xlabel = 'Profit Strategy', 
        ylabel = 'Extraction')
ax4.invert_yaxis()
ax4.set(title = 'Arousal')
f101.tight_layout()
f101.savefig('Ar_Strat-Ext.pdf', bbox_inches='tight') 
plt.close()

dftanr = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['sensecat'], values='n_sust',aggfunc = aggmet)
dftane = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['sensecat'], values='n_fair',aggfunc = aggmet)
dftant = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['sensecat'], values='n_distr',aggfunc = aggmet)
dftanz = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['sensecat'], values='n_profit',aggfunc = aggmet)

f11, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2,  sharey=True, sharex=False, figsize=(20,20))
sns.heatmap(dftanr, cmap='coolwarm', ax = ax1, vmin = 0, vmax = 1)
ax1.set(xlabel = 'Sense making', 
       ylabel = 'Network Architecture')
ax1.invert_yaxis()
ax1.set(title = 'Sustainability Strategy')
sns.heatmap(dftane, cmap='coolwarm', ax = ax2, vmin = 0, vmax = 1)
ax2.set(xlabel = 'Sense making', 
       ylabel = 'Network Architecture')
ax2.invert_yaxis()
ax2.set(title = 'Fairness Strategy')
sns.heatmap(dftant, cmap='coolwarm', ax = ax3, vmin = 0, vmax = 1)
ax3.set(xlabel = 'Sense making', 
       ylabel = 'Network Architecture')
ax3.invert_yaxis()
ax3.set(title = 'Equality Strategy')
sns.heatmap(dftanz, cmap='coolwarm', ax = ax4, vmin = 0, vmax = 1)
ax4.set(xlabel = 'Sense making', 
       ylabel = 'Network Architecture')
ax4.invert_yaxis()
ax4.set(title = 'Profit Strategy')
f11.tight_layout()
f11.savefig('Sens-Net-Strat.pdf', bbox_inches='tight') 
plt.close()


dfttse = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['sustcat'], values='maxtime',aggfunc = 'mean')
dfttss = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['faircat'], values='maxtime',aggfunc = 'mean')
dftsea = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['distrcat'], values='maxtime',aggfunc = 'mean')
dftsez = pd.pivot_table(modlt.reset_index(), index=['net'], columns =['profitcat'], values='maxtime',aggfunc = 'mean')

f13, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(20,20))
sns.heatmap(dfttse, cmap='coolwarm', ax = ax1, vmin = 0, vmax = 0.5)
ax1.set(xlabel = 'Sustainabilty Strategy', 
       ylabel = 'Network Architecture ')
ax1.invert_yaxis()
ax1.set(title = 'Resource Sustainability')
sns.heatmap(dfttss, cmap='coolwarm', ax = ax2, vmin = 0, vmax = 0.5)
ax2.set(xlabel = 'Fairness Strategy',
       ylabel = 'Network Architecture')
ax2.set(title = 'Resource Sustainability')
ax2.invert_yaxis()
sns.heatmap(dftsea, cmap='coolwarm', ax = ax3, vmin = 0, vmax = 0.5)
ax3.set(xlabel = 'Equality Strategy', 
       ylabel = 'Network Architecture')
ax3.set(title = 'Resource Sustainability')
ax3.invert_yaxis()
sns.heatmap(dftsez, cmap='coolwarm', ax = ax4, vmin = 0, vmax = 0.5)
ax4.set(xlabel = 'Profit Strategy', 
       ylabel = 'Network Architecture')
ax4.set(title = 'Resource Sustainability')
ax4.invert_yaxis()
f13.tight_layout()
f13.savefig('Net_Strat_Time_mean.pdf', bbox_inches='tight') 
plt.close()



dfnetsense = pd.pivot_table(modlt.reset_index(), index=['arousecat'], columns =['profitcat'], values='maxtime',aggfunc = 'mean')

f14, (ax) = plt.subplots(1, 1,  sharey=False, sharex=False, figsize=(20,20))
sns.heatmap(dfnetsense, cmap='coolwarm', ax = ax, vmin = 0, vmax = 1)
ax.set(xlabel = 'Sensemaking', 
       ylabel = 'Network Architecture ')
ax.invert_yaxis()
f14.tight_layout()
f14.savefig('Ar_prof_Time_mean.pdf', bbox_inches='tight') 
plt.close()


os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/Figures/ByPop')

#by pop
popmax = pd.unique(modlt['PoP'])
for pop in popmax:
    # Filter the DataFrame for the current 'PoP' value
    modpop  =  modlt[(modlt['PoP'] == pop)]
    
    dftcar = pd.pivot_table(modpop.reset_index(), index=['tomcat'], columns =['arousecat'], values='sc',aggfunc = aggmet)
    dftcae = pd.pivot_table(modpop.reset_index(), index=['syscat'], columns =['arousecat'], values='ec',aggfunc = aggmet)
    dftcat = pd.pivot_table(modpop.reset_index(), index=['sccat'], columns =['eccat'], values='sense',aggfunc = aggmet)

    f10, (ax1, ax2, ax3) = plt.subplots(1, 3,  sharey=False, sharex=False, figsize=(20,20))
    sns.heatmap(dftcar, cmap='coolwarm', vmin = 0, vmax = 1, ax = ax1)
    ax1.set(xlabel = 'Arousal', 
            ylabel = 'ToM')
    ax1.invert_yaxis()
    ax1.set(title = 'Social Cognition')
    sns.heatmap(dftcae, cmap='coolwarm', vmin = 0, vmax = 1, ax = ax2)
    ax2.set(xlabel = 'Arousal', 
            ylabel = 'Systematizing')
    ax2.invert_yaxis()
    ax2.set(title = 'Environmental Cognition')
    sns.heatmap(dftcat, cmap='coolwarm', vmin = 0, vmax = 1, ax = ax3)
    ax3.set(xlabel = 'System Cognition', 
            ylabel = 'Social Cognition')
    ax3.invert_yaxis()
    ax3.set(title = 'Sensemaking')
    f10.tight_layout()
    f10.savefig('Cog_Sense-Arouse' +'_' + str(pop) +'.pdf', bbox_inches='tight') 
    plt.close()

    dftsar = pd.pivot_table(modpop.reset_index(), index=['extcat'], columns =['sustcat'], values='arouse',aggfunc = aggmet)
    dftsae = pd.pivot_table(modpop.reset_index(), index=['extcat'], columns =['faircat'], values='arouse',aggfunc = aggmet)
    dftsat = pd.pivot_table(modpop.reset_index(), index=['extcat'], columns =['distrcat'], values='arouse',aggfunc = aggmet)
    dftsas = pd.pivot_table(modpop.reset_index(), index=['extcat'], columns =['profitcat'], values='arouse',aggfunc = aggmet)

    f101, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(20,20))
    sns.heatmap(dftsar, cmap='coolwarm', ax = ax1, vmin = 0.4, vmax = 0.9)
    ax1.set(xlabel = 'Sustainability Strategy', 
            ylabel = 'Extraction')
    ax1.invert_yaxis()
    ax1.set(title = 'Arousal')
    sns.heatmap(dftsae, cmap='coolwarm', ax = ax2, vmin = 0.4, vmax = 0.9)
    ax2.set(xlabel = 'Fairness Strategy', 
            ylabel = 'Extraction')
    ax2.invert_yaxis()
    ax2.set(title = 'Arousal')
    sns.heatmap(dftsat, cmap='coolwarm', ax = ax3, vmin = 0.4, vmax = 0.9)
    ax3.set(xlabel = 'Equality Strategy', 
            ylabel = 'Extraction')
    ax3.invert_yaxis()
    ax3.set(title = 'Arousal')
    sns.heatmap(dftsas, cmap='coolwarm', ax = ax4, vmin = 0.4, vmax = 0.9)
    ax4.set(xlabel = 'Profit Strategy', 
            ylabel = 'Extraction')
    ax4.invert_yaxis()
    ax4.set(title = 'Arousal')
    f101.tight_layout()
    f101.savefig('Ar_Strat-Ext' +'_' + str(pop) + '.pdf', bbox_inches='tight') 
    plt.close()

    dftanr = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['sensecat'], values='n_sust',aggfunc = aggmet)
    dftane = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['sensecat'], values='n_fair',aggfunc = aggmet)
    dftant = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['sensecat'], values='n_distr',aggfunc = aggmet)
    dftanz = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['sensecat'], values='n_profit',aggfunc = aggmet)

    f11, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2,  sharey=True, sharex=False, figsize=(20,20))
    sns.heatmap(dftanr, cmap='coolwarm', ax = ax1, vmin = 0, vmax = 1)
    ax1.set(xlabel = 'Sense making', 
           ylabel = 'Network Architecture')
    ax1.invert_yaxis()
    ax1.set(title = 'Sustainability Strategy')
    sns.heatmap(dftane, cmap='coolwarm', ax = ax2, vmin = 0, vmax = 1)
    ax2.set(xlabel = 'Sense making', 
           ylabel = 'Network Architecture')
    ax2.invert_yaxis()
    ax2.set(title = 'Fairness Strategy')
    sns.heatmap(dftant, cmap='coolwarm', ax = ax3, vmin = 0, vmax = 1)
    ax3.set(xlabel = 'Sense making', 
           ylabel = 'Network Architecture')
    ax3.invert_yaxis()
    ax3.set(title = 'Equality Strategy')
    sns.heatmap(dftanz, cmap='coolwarm', ax = ax4, vmin = 0, vmax = 1)
    ax4.set(xlabel = 'Sense making', 
           ylabel = 'Network Architecture')
    ax4.invert_yaxis()
    ax4.set(title = 'Profit Strategy')
    f11.tight_layout()
    f11.savefig('Sens-Net-Strat' +'_' + str(pop) + '.pdf', bbox_inches='tight') 
    plt.close()

    dfttse = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['sustcat'], values='maxtime',aggfunc = 'median')
    dfttss = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['faircat'], values='maxtime',aggfunc = 'median')
    dftsea = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['distrcat'], values='maxtime',aggfunc = 'median')
    dftsez = pd.pivot_table(modpop.reset_index(), index=['net'], columns =['profitcat'], values='maxtime',aggfunc = 'median')

    f13, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(20,20))
    sns.heatmap(dfttse, cmap='coolwarm', ax = ax1, vmin = 0, vmax = 0.5)
    ax1.set(xlabel = 'Sustainabilty Strategy', 
           ylabel = 'Extraction ')
    ax1.invert_yaxis()
    ax1.set(title = 'Sustainability')
    sns.heatmap(dfttss, cmap='coolwarm', ax = ax2, vmin = 0, vmax = 0.5)
    ax2.set(xlabel = 'Fairness Strategy',
           ylabel = 'Extraction')
    ax2.set(title = 'Sustainability')
    ax2.invert_yaxis()
    sns.heatmap(dftsea, cmap='coolwarm', ax = ax3, vmin = 0, vmax = 0.5)
    ax3.set(xlabel = 'Equality Strategy', 
           ylabel = 'Extraction')
    ax3.set(title = 'Sustainability')
    ax3.invert_yaxis()
    sns.heatmap(dftsez, cmap='coolwarm', ax = ax4, vmin = 0, vmax = 0.5)
    ax4.set(xlabel = 'Profit Strategy', 
           ylabel = 'Extraction')
    ax4.set(title = 'Sustainability')
    ax4.invert_yaxis()
    f13.tight_layout()
    f13.savefig('Net_Strat_Time' +'_' + str(pop) + '.pdf', bbox_inches='tight') 
    plt.close()

"""

Other figures: Violinplots

"""
os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/Figures/BoxPlots')
sns.set(context='paper', style='whitegrid', palette='colorblind', font='Helvetica', font_scale=1.5)


#Bin resource related parameters
modlt['varcat'] = pd.cut(modlt.Var_mag, 10)
modlt['needcat'] = pd.cut(modlt.Needs, 10)
modlt['growthcat'] = pd.cut(modlt.ResGrowth, 10)
modlt['minrescat'] = pd.cut(modlt.MinRes, 10)

plt.figure()
ax = sns.violinplot(x=modlt["varcat"], y=modlt["maxtime"], hue = modlt['PoP'], palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Resource Disturbance', ylabel='Sustainability')
plt.savefig('Dist-time.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["needcat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Min Resource Needs', ylabel='Sustainability')
plt.savefig('Needs-Time.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["growthcat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Resource Growth', ylabel='Sustainability')
plt.savefig('Growth-Time.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["minrescat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Min Resources Allowed', ylabel='Sustainability')
plt.savefig('MinRes-Time.pdf', bbox_inches='tight')
plt.close()


plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["eccat"], y=modlt["sense"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Environmental Cognition', ylabel='Sensemaking')
plt.savefig('Ec-Sense.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["sccat"], y=modlt["sense"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Social Cognition', ylabel='Sensemaking')
plt.savefig('Sc-Sense.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["net"], y=modlt["sense"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Network Architecture', ylabel='Sensemaking')
plt.savefig('Net-Sense.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["extcat"], y=modlt["arouse"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Extraction', ylabel='Arousal')
plt.savefig('Ext-Arouse.pdf', bbox_inches='tight')
plt.close()


plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["arousecat"], y=modlt["ec"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Arousal', ylabel='Environmental Cognition')
plt.savefig('Arouse-EnvCognition.pdf', bbox_inches='tight')
plt.close()


plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["syscat"], y=modlt["ec"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Systematizing', ylabel='Environmental Cognition')
plt.savefig('Sys-EnvCognition.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["arousecat"], y=modlt["sc"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='Arousal', ylabel='Social Cognition')
plt.savefig('Arouse-SocialCognition.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 10))
ax = sns.violinplot(x=modlt["tomcat"], y=modlt["sc"],hue = modlt['PoP'],  palette="tab10", cut=0)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel='ToM', ylabel='Social Cognition')
plt.savefig('Tom-SocialCognition.pdf', bbox_inches='tight')
plt.close()


fig1,((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(50,50))

sns.violinplot(x=modlt["sustcat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
#ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax1.set(xlabel='Sustainability Strategy', ylabel='Sustainability')

sns.violinplot(x=modlt["faircat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax2.set(xlabel='Fairness Strategy', ylabel='Sustainability')

sns.violinplot(x=modlt["distrcat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
#ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax3.set(xlabel='Equality Strategy', ylabel='Sustainability')

sns.violinplot(x=modlt["profitcat"], y=modlt["maxtime"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(),rotation=90)
#ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax4.set(xlabel='Profit Strategy', ylabel='Sustainability')

ax1.get_legend().remove()
ax3.get_legend().remove()
ax4.get_legend().remove()
fig1.tight_layout()
fig1.savefig('Strategies-Time.pdf', bbox_inches='tight') 
plt.close()


fig1,((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(25,25))

sns.violinplot(x=modlt["sustcat"], y=modlt["maxtime"], cut=0, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
#ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax1.set(xlabel='Sustainability Strategy', ylabel='Resource Sustainability')

sns.violinplot(x=modlt["faircat"], y=modlt["maxtime"], cut=0, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
#ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax2.set(xlabel='Fairness Strategy', ylabel='Resource Sustainability')

sns.violinplot(x=modlt["distrcat"], y=modlt["maxtime"], cut=0, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
#ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax3.set(xlabel='Equality Strategy', ylabel='Resource Sustainability')

sns.violinplot(x=modlt["profitcat"], y=modlt["maxtime"], cut=0, ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(),rotation=90)
#ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax4.set(xlabel='Profit Strategy', ylabel='Resource Sustainability')


fig1.tight_layout()
fig1.savefig('Strategies-TimeA.pdf', bbox_inches='tight') 
plt.close()



# Create subplots
fig2,((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(50,50))

sns.violinplot(x=modlt["sensecat"], y=modlt["n_sust"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax1)
ax1.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax1.set(xlabel='Sensemaking', ylabel='Sustainability Strategy')

sns.violinplot(x=modlt["sensecat"], y=modlt["n_fair"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax2)
ax2.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax2.set(xlabel='Sensemaking', ylabel='Fairness Strategy')

sns.violinplot(x=modlt["sensecat"], y=modlt["n_distr"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax3)
ax3.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax3.set(xlabel='Sensemaking', ylabel='Equality Strategy')

sns.violinplot(x=modlt["sensecat"], y=modlt["n_profit"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax4)
ax4.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax4.set(xlabel='Sensemaking', ylabel='Profit Strategy')

ax1.get_legend().remove()
ax3.get_legend().remove()
ax4.get_legend().remove()
fig2.tight_layout()
fig2.savefig('Sensemaking-Strategies.pdf', bbox_inches='tight') 
plt.close()


fig3,((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2,  sharey=False, sharex=False, figsize=(50,50))

sns.violinplot(x=modlt["sustcat"], y=modlt["arouse"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax1)
ax1.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax1.set(xlabel='Sustainability Strategy', ylabel='Arousal')

sns.violinplot(x=modlt["faircat"], y=modlt["arouse"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax2)
ax2.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax2.set(xlabel='Fairness Strategy', ylabel='Arousal')

sns.violinplot(x=modlt["distrcat"], y=modlt["arouse"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax3)
ax3.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax3.set(xlabel='Equality Strategy', ylabel='Arousal')

sns.violinplot(x=modlt["profitcat"], y=modlt["arouse"],hue = modlt['PoP'],  palette="tab10", cut=0, ax=ax4)
ax4.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax4.set(xlabel='Profit Strategy', ylabel='Arousal')

ax1.get_legend().remove()
ax3.get_legend().remove()
ax4.get_legend().remove()
fig3.tight_layout()
fig3.savefig('Strategies-Arousal.pdf', bbox_inches='tight') 
plt.close()

