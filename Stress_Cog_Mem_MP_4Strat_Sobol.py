#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:10:07 2023

Model code for Stress, Cognition, Memory and group adaptability to changing conditions.

Simple  model to assess how stress arousal, cognition and memory affect ability of groups to solve complex problems and adapt
to changing conditions .

The model is comprised by two, interdependent feedback loops: The SRL (sensemaking-resource loop), and the SAL (sensemaking-arousal loop)


"""

import os
from joblib import Parallel, delayed
import multiprocessing

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import networkx as nx
from itertools import combinations, groupby
#for parameeter exploration via sobol sequences
from scipy.stats.qmc import Sobol


#E-R random graph with no isolates and one connected component (probability of connection is less meaningful for low proabilities)
def er_connected_graph(n, p):
    """
    Parameters
    ----------
    n : number of nodes
    p : probability of two nodes to be connected

    Returns
    -------
    G : Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted, 
    starting by adding one edge to each node and then adding the rest of edges with probability p
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = rnd.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if rnd.random() < p:
                G.add_edge(*e)
    return G

#function representing D (disturbance)
def resource_fluctuations(t, mag):
    """
    Parameters
    ----------
    t : time 

    Returns
    -------
    this function creates disturbances (fluctuations) in the resource. This will be added to the resource intrinsic growth.
    
    """

    # Set parameters to control fluctuations
    amplitude = 0.5
    frequency = 0.1
    magnitude = mag
    # Generate fluctuations using sine and random noise
    fluctuation = amplitude * np.sin(2 * np.pi * frequency * t)
    noise = magnitude * np.random.randn()
    disturb = fluctuation + noise

    return disturb

#group sensemaking based on ec and sc values
def get_sensemaking(net, node):
    """
    Parameters
    ----------
    net     : network 
    node    : node 

    Returns
    -------
    sensemaking : this function calculates a sensemaking threshold by taking the average of the "ec" values and the minimum of the "sc" 
    values within the subgraph of a node and its neighbors. The sensemaking threshold is a measure used to determine 
    the level of similarity or proximity required for sensemaking between nodes in the network.

    """
    # Get the subgraph of the node and its neighbors
    subgraph = net.subgraph(list(net.neighbors(node)) + [node])
    # Get the ec and sc values of the nodes in the subgraph
    ecvals = [subgraph.nodes[n]['ec'] for n in subgraph.nodes()]
    scvals = [subgraph.nodes[n]['sc'] for n in subgraph.nodes()]
    # Calculate the sensemaking threshold as the average of ec values and the minimum of sc values
    sensemaking = (np.mean(ecvals) + min(scvals)) / 2
    return sensemaking



def update_objectives(net, node):
    """
    Parameters
    ----------
    net     : network
    node    : node

    Returns
    -------
    This function updates the objective attributes of the given node and its neighbors based on a sensemaking threshold 
    and ensures that the attribute preferences are properly scaled.
    
    """
    # Get the sensemaking threshold for the node
    sensemaking = get_sensemaking(net, node)
    nx.set_node_attributes(net, {node : sensemaking}, name = 'sense')
    
    # Get the subgraph of the node and its neighbors
    subgraph = net.subgraph(list(net.neighbors(node)) + [node])
    # Get the nodes within the sensemaking threshold
    thdistr = [n for n in subgraph.nodes() if abs(subgraph.nodes[n]['distr'] - subgraph.nodes[node]['distr']) < sensemaking]
    thfair = [n for n in subgraph.nodes() if abs(subgraph.nodes[n]['fair'] - subgraph.nodes[node]['fair']) < sensemaking] 
    thsustain = [n for n in subgraph.nodes() if abs(subgraph.nodes[n]['sustain'] - subgraph.nodes[node]['sustain']) < sensemaking] 
    thprofit = [n for n in subgraph.nodes() if abs(subgraph.nodes[n]['profit'] - subgraph.nodes[node]['profit']) < sensemaking] 

    # Calculate the average value of each attribute for the nodes within the sensemaking threshold and update attributes of node and neighbors
    #calculate only if there are neighbors within the sensemaking threshold.
    avg_distr = np.mean([subgraph.nodes[n]['distr'] for n in thdistr]) if thdistr else nx.get_node_attributes(net, 'distr')[node]
    nx.set_node_attributes(net, {n: avg_distr for n in [node] + thdistr}, name='distr')
    
    avg_profit = np.mean([subgraph.nodes[n]['profit'] for n in thprofit]) if thprofit else nx.get_node_attributes(net, 'profit')[node]
    nx.set_node_attributes(net, {n: avg_profit for n in [node] + thprofit}, name='profit')

    avg_fair = np.mean([subgraph.nodes[n]['fair'] for n in thfair]) if thfair else nx.get_node_attributes(net, 'fair')[node]
    nx.set_node_attributes(net, {n: avg_fair for n in [node] + thfair}, name='fair')
    
    avg_sustain = np.mean([subgraph.nodes[n]['sustain'] for n in thsustain]) if thsustain else nx.get_node_attributes(net, 'sustain')[node]
    nx.set_node_attributes(net, {n: avg_sustain for n in [node] + thsustain}, name='sustain')
   
    #now rescale as the sum of the objective preferences can not be higher than one.
    # first check whether attributes sum is > 1
    attr_sum = net.nodes[node]['distr'] + net.nodes[node]['fair'] + net.nodes[node]['profit'] + net.nodes[node]['sustain']
    #if it is the case, then rescale accordingly.
    if attr_sum != 1:
        for n in net.nodes():
            attrs = net.nodes[n]
            total = attrs['distr'] + attrs['fair'] + attrs['profit'] + attrs['sustain']
            if total > 0:
                attrs['distr'] /= total
                attrs['fair'] /= total
                attrs['profit'] /= total
                attrs['sustain'] /= total

    
def select_objective(net, node, maxonly):
    """
    Parameters
    ----------
    net     : network
    node    : node
    maxonly : if strategy are based on cumulative probability of if select the highest preferred strategy

    Returns
    -------
    This function aims to select an objective based on the attribute values of a node, 
    taking into account the relative strengths of preferences indicated by the differences between attribute values.
    using cumulative probabilities and comparing them with random numbers, the function selects an objective 
    based on the attribute values and their relative strengths of preferences. 
    
    The selection is based on the cumulative probabilities.
    """
    selected_attributes = ['distr', 'fair', 'profit', 'sustain']
    node_attributes = net.nodes[node]
    node_objs = {attr: node_attributes[attr] for attr in selected_attributes}
    sorted_objs = sorted(node_objs.items(), key=lambda x: x[1], reverse=True)
        
    if maxonly == 1:
        selected_strategy = sorted_objs[0][0]
    else:
        diff_values = [sorted_objs[i][1] - sorted_objs[i+1][1] for i in range(len(sorted_objs)-1)]
        # Calculate total sum and probabilities for each attribute
        total = sum([sorted_objs[i][1] for i in range(len(sorted_objs))])
        probabilities = [sorted_objs[i][1] / total for i in range(len(sorted_objs))]
    
        # Choose strategy with probability proportional to attribute values
        rand = np.random.rand()
        cumulative_prob = 0
        for i in range(len(sorted_objs)):
            cumulative_prob += probabilities[i]
            if i < len(diff_values) and diff_values[i] > 0.25:
                if rand <= cumulative_prob:
                    selected_strategy = sorted_objs[i][0]
                    break
            else:
                if rand < cumulative_prob:
                    selected_strategy = sorted_objs[i][0]
                    break
    return selected_strategy
            

def determine_extraction(objmain, pop_size, needed, min_resource, resource, fair_pref, profit_pref, resorig):
    """
    Parameters
    ----------
    objmain     : main objective (that determines resource extraction of a node)
    pop_size    : network size (number of individuals in the system)
    needed      : minimum amount of resources needed for a node to function
    min_resource: proportion of resources that are required for the system not to collapse
    resource    : current resources available
   
    fair_pref   : preference for the fair extraction strategy (equals to how much effort i put into that)
    profit_pref : preference for the profit extraction strategy (equals the amount of effort i put into extracting for profit)
    resorig     : the original state of the resources (at the beginning of the simulation run)

    Returns
    -------
    This unction determines how much resources are extracted by each node/individual.
    If resources fall below minimum, extraction will be 0 as no extraction can be possible
    """
    
    minsust = resorig * min_resource #minum resources needed for the system to keep functioning.
    ngroups = pop_size / grsize     #assumption is that resources are divided equally to groups of 5 individuals, if one changes grsize, this should change as well
    if minsust > resource:
        return 0
    else:
        #here they take a share that is proportional with how many people have access to the resource
        if objmain == 'distr':
            extr_temp = (resource - minsust) / (pop_size)  
        #here fair means that their preference is the effor they put into extracting resources,
        #and takes into account minimum resources needed for the system to function as well as 
        #the fact that system resources should be allocated to a group based on individual effort made. Max they take 1/2  of all resources.
        if objmain == 'fair':
            share = pop_size * (1 - fair_pref)
            if share < 2:
                share = 2
            extr_temp = ((resource - minsust) / (share * ngroups))    
        #here profit means that they maximize profit not worrying about min resources for sustainability. Profit max takes max 1/2 + 1 of all resources
        if objmain == 'profit':
            share = pop_size * (1 - profit_pref)
            if share < 2:
                share = 2
            extr_temp =  (resource / (share * ngroups)) + 1                 
        #here sustain means that individuals will take the amount of resources above the original level of resources, 
        #if not, they take only what is needed
        if objmain == 'sustain':                           
            if resource > resorig:
                extr_temp = resource - resorig 
            else:
                extr_temp =  needed
        #this implies that individuals, if the strategy they choose gives them less than what is needed, 
        #they take what is needed.
        if extr_temp >= needed:                           
            return extr_temp
        else:
            return needed
        
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
        
def evaluate_objectives(obext, resource, resorig, resource_min, needed, distr, fair, profit, sustain, node, net):
    """
    Parameters
    ----------
    obext        : extraction of a specific node/individual 
    resource     : resources available at time t, before extraction starts
    resorig      : resources available at the beginning of the simulation run
    resource_min : proportion of resources needed for the system not to collapse
    needed       : minimum amount of resources needed by a node to function
    distr        : preference for distributional equity strategy
    fair         : preference for fair strategy that implies extraction is proportionate to effort in fair
    profit       : preference for profit strategy that implies that extraction is proportionate to how much i value profit (i.e. profit preference)
    sustain      : preference for the sustain strategy that implies taking resources without reducing them beyond the original resources available, notwithstanding what is needed.
    node         : node calling the function
    subgraphs_id : id of the subgraph the node is part (node + 1st degree neighbors)
    
    Returns
    -------
    evaluation : This function evaluates the objectives based on the provided input parameters and calculates their respective objective values. 
    The objective values are then weighted and combined to obtain an overall evaluation score.
    
    For equality of distribution, one evaluate the objective by calculating gini coefficient of extraction, this implies information on what
    others are extracting, an error could be introduced. Here we introduce an error of max 10% of evaluation of the gini coefficient and bound
    all the evaluation results between between 0 and 1 so that the weighted evaluation will also be between 0 and 1.
    """
    ngroups = len(net.nodes) / grsize #5 is grsize, if grsize changes this needs to change as well.
    if obext < needed:
        distr_obj   = 0
        fair_obj    = 0
        profit_obj  = 0
        sustain_obj = 0
    else:
        #get the extraction of neighbors to calculate max and gini
        subgrapheval =  nx.ego_graph(net, node, radius=1, center=True)
        subgraph_obextall = np.array(list(nx.get_node_attributes(subgrapheval, 'extract').values()))
        #evaluate distance to objective for all objectives (evaluateion will then be the weighted sum of the distance between extraction and objective)
        #this takes into account that although we go for one extraction strategy, we may have also some preferences for others and they influence our evaluation of the overall
        #extraction and resources left.
        distr_obj  = np.clip(1 - ((gini(subgraph_obextall))), 0,1)
        fair_obj   = np.clip(1 - (fair - (obext / (resource - ((resource* resource_min))/ngroups))), 0,1)
        profit_obj = np.clip(1 - ((max(subgraph_obextall) - obext) / max(subgraph_obextall)), 0,1)
        sustain_obj = np.clip(((resource - resorig) / resorig), 0,1)
    
    evaluation = distr_obj * distr + fair_obj * fair + profit_obj * profit + sustain_obj * sustain  
    return evaluation
      

#main model loop with parameters, all the action happens in t. Note that the sequence of parameters must match the sequence from the Sobol sampler.
#Given that we are interested only in the long-term dynamics we will store results only for the last 10 time-steps before collapse (resource or social) or max time of the sim reached
def run_simulation(gr, mr, nd, mg, ta, ts, sa, ss, netkey, net, parcomb, nr):
    # Extract the parameters
    for t in range (0, Time, 1):
        if t == 0:
            notmet = 0 #this is to take into account how many times individuals do not meet their resource need, 5 times in a row 51% of individuals does not meet their needs the system collapses 
            popsize = len(net.nodes)
            res_t = R * popsize
            res_t0 = res_t #initial resources available
            minres = res_t0 * mr #minimum amount of resources needed for the system to work
            nx.set_node_attributes(net, {node_id: node_id for node_id in net.nodes}, name='node_id')
            nx.set_node_attributes(net, parcomb, name='run')
            nx.set_node_attributes(net, nr, name='rep')
            nx.set_node_attributes(net, t, name='time')
            nx.set_node_attributes(net, popsize, name='PoP')
            nx.set_node_attributes(net, ta, name='ToM')
            nx.set_node_attributes(net, ts, name='ToM_sd')
            nx.set_node_attributes(net, sa, name='Sys')
            nx.set_node_attributes(net, ss, name='Sys_sd')
            nx.set_node_attributes(net, netkey, name='Network')
            nx.set_node_attributes(net, res_t, name='Resource')
            nx.set_node_attributes(net, mr, name = 'MinRes')
            nx.set_node_attributes(net, gr, name = 'ResGrowth')
            for ind in net.nodes:
                #starting conditions, tom and sys do not change, during the simulation run.
                tomattr = np.clip(np.random.normal(ta, ts),0,1)
                sysattr = np.clip(np.random.normal(sa, ss), 0,1)
                nx.set_node_attributes(net, {ind: tomattr}, name="tom")
                nx.set_node_attributes(net, {ind: sysattr}, name="sys")
                #arousal at time 0
                arattr = np.clip(np.random.normal(0.5,0.1),0,1)
                nx.set_node_attributes(net, {ind: arattr}, name="arouse")
                #objective preferences at time 0
                objvect = ['sustain', 'profit', 'distr', 'fair']
                #if we want to assess the model with less strategies, we just put the ones we do not want at 0 to start.
                ob1 = np.random.uniform(0, 1)
                ob2 = np.random.uniform(0, 1)
                ob3 = np.random.uniform(0, 1)
                ob4 = np.random.uniform(0, 1)
                #rescael objective preferences so they sum to 1
                tot = ob1 + ob2 + ob3 + ob4
                ob1 = ob1 / tot
                ob2 = ob2 / tot
                ob3 = ob3 / tot
                ob4 = ob4 / tot
                obtemp = {objvect[0]: ob1, objvect[1]: ob2, objvect[2]:ob3, objvect[3]: ob4}
                #now sort the keys alphabetically, so all individiual have the same sequence 
                obtemp = {key: value for key, value in sorted(obtemp.items())}
                nx.set_node_attributes(net, {ind : ob1}, name = 'profit')
                nx.set_node_attributes(net, {ind : ob2}, name = 'distr')   
                nx.set_node_attributes(net, {ind : ob3}, name = 'sustain')   
                nx.set_node_attributes(net, {ind : ob4}, name = 'fair')
                #select the initial main objective that determines effort/extraction - only recordkeeping as this will change in the next round. Here no interaction between nodes.
                objsel = select_objective(net, ind, maxonly = 0)
                nx.set_node_attributes(net, {ind: objsel}, name = 'strategy')
            result = pd.DataFrame.from_dict(dict(net.nodes(data=True)), orient='index')
        else:
            #update resource via growth and disturbance
            dist = resource_fluctuations(t, mg)
            res_t = res_t + (res_t * gr) + dist
            #if the resource, with growth are below the threshold for system functionality, stop the loop and return results.
            if res_t < minres:
                return result
            #first assess env and social cognition of each individual and arousal
            syattr = nx.get_node_attributes(net, 'sys')
            tomattr = nx.get_node_attributes(net, 'tom')
            arousattr = nx.get_node_attributes(net, 'arouse')
            for ind in net.nodes:
                #we start at time T with 
                ec1 = (2 * syattr[ind] * (- 2 * arousattr[ind]**2 + 2 * arousattr[ind]))
                sc1 = (2 * tomattr[ind] * (- 2 * arousattr[ind]**2 + 2 * arousattr[ind]))
                nx.set_node_attributes(net, {ind : ec1}, name = 'ec')
                nx.set_node_attributes(net, {ind : sc1}, name = 'sc')
            for ind in net.nodes:
                #update objectives based on sensemaking, that depends on arousal as well as social and environmental cognition
                #define function to update objectives based on sensemaking threshold
                update_objectives(net, ind)                                
            #now with the updated objectives, we check the effort in resource extraction that will depend on preferences and probabilities
            #but shuffle the nodes so that the order is not the same 
            nodelist = list(net.nodes)
            rnd.shuffle(nodelist)
            res_start = res_t #resources available before effort in extraction
            for ind in nodelist:
                #select the main objective that determines effort/extraction
                objsel = select_objective(net, ind, maxonly=1)
                nx.set_node_attributes(net, {ind: objsel}, name = 'strategy')
                #this is important only if the objective selected is fair or profit.
                fair_i  = nx.get_node_attributes(net, 'fair')[ind]
                prof_i  = nx.get_node_attributes(net, 'profit')[ind]
                #determine the effort and the extraction, not that given the function if res_t < minimum resources, than extraction will be 0
                extract = determine_extraction(objmain = objsel, pop_size = nm, resource = res_t, needed = nd, 
                                       min_resource = mr, fair_pref = fair_i, 
                                       profit_pref = prof_i, resorig = res_t0) 
                nx.set_node_attributes(net, {ind : extract}, name = 'extract')
                res_group = res_t * grsize / (popsize)
                nx.set_node_attributes(net,  {ind : res_group}, name='avail_res')
                res_t = res_t - extract

            #now evaluate the objectives and update arousal accordingly, than restart loop
            for ind in net.nodes:
                #get the values we eed, fairness preference, profit preference, extraction of all individuals, resource at time t
                #and personal extraction
                distr_i = nx.get_node_attributes(net, 'distr')[ind]
                fair_i  = nx.get_node_attributes(net, 'fair')[ind]
                prof_i  = nx.get_node_attributes(net, 'profit')[ind]
                sust_i  = nx.get_node_attributes(net, 'sustain')[ind]
                extr_i  = nx.get_node_attributes(net, 'extract')[ind]
                
                evalext = evaluate_objectives(obext=extr_i, resource = res_start, resorig = res_t0,
                                          resource_min = mr, needed = nd, distr = distr_i, 
                                          fair = fair_i, profit = prof_i, sustain = sust_i, node = ind, net = net)
    
                if evalext == 0:
                    arousal = 1
                elif evalext == 1:
                    arousal = 0
                else:
                    evalar = (evalext / (1-evalext))**(-2)
                    arousal = 1 / (1 + evalar) #(np.tanh(evalext/(1-evalext))**5) #arousal = 0 when evaluation is maximum, = maximum satisfaction, and 1 when one has minimum satisfaction
                nx.set_node_attributes(net, {ind : arousal}, name = 'arouse')
            
            nx.set_node_attributes(net, {node_id: node_id for node_id in net.nodes}, name='node_id')
            nx.set_node_attributes(net, parcomb, name='run')
            nx.set_node_attributes(net, nr, name='rep')
            nx.set_node_attributes(net, t, name='time')
            nx.set_node_attributes(net, popsize, name='PoP')
            nx.set_node_attributes(net, nd, name = 'Needs')
            nx.set_node_attributes(net, mg, name = 'Var_mag')
            nx.set_node_attributes(net, ta, name='ToM')
            nx.set_node_attributes(net, ts, name='ToM_sd')
            nx.set_node_attributes(net, sa, name='Sys')
            nx.set_node_attributes(net, ss, name='Sys_sd')
            nx.set_node_attributes(net, res_t, name = 'Resource')
            nx.set_node_attributes(net, mr, name = 'MinRes')
            nx.set_node_attributes(net, gr, name = 'ResGrowth')
            nx.set_node_attributes(net, netkey, name='Network')
            result_temp = pd.DataFrame.from_dict(dict(net.nodes(data=True)), orient='index')
        if t !=0:
            result = pd.concat([result, result_temp], ignore_index=True)

        #if more than 75% of individuals are not able to extract enough resources for 5 timesteps in a row, stop the loop
        arrayextract = np.array(list(nx.get_node_attributes(net, 'extract').values()))
        if t > 1:
            if np.percentile(arrayextract, 75) < nd:
                notmet +=1
            else:
                notmet = 0
            if notmet > 4:
                return result
    # Update the counter for each parameter combination and if using salib for sobol indices results are only the time.
    return result


#parameters used for having individual dataframes based on population size (makes it faster as pd.concat is slow)
grsize = 5 #individuals in a group
Ngr = [10]   #number of groups used: 1, 2, 5, 10
Nmax = [ng1 * grsize for ng1 in Ngr]  #number of individuals is equal to the groupsize * the number of groups

netdict = {}
pr = 0
for nm in Nmax:
    print(nm)
   #generate a cycle based graph where 5 nodes are in cycle and one node of each cycle is connected to another node in another cycle
    gcycle = nx.Graph()
    for i in range(0, nm, grsize):
        cycle = nx.cycle_graph(grsize)
        mapping = dict(zip(cycle.nodes(), range(i, i+grsize)))
        cycle = nx.relabel_nodes(cycle, mapping)
        gcycle.add_edges_from(cycle.edges())
        # Assign unique group IDs to nodes in each cycle
        group_id = i // grsize  # Calculate the group ID based on the current index
        nx.set_node_attributes(gcycle, {node: group_id for node in cycle.nodes()}, name='groupid')
    # Connect one node from each cycle to another cycle
    for i in range(0, nm, grsize):
        gcycle.add_edge(i, (i+grsize)% nm)
    gcycle.remove_edges_from(nx.selfloop_edges(gcycle))
    netname = 'cycle_' + str(nm)
    netdict[netname] = gcycle
   
   #generate a star based graph where grsize nodes are in star and either the most or the least connected node of each star is connected to another node in another star
   #highest degrees connected
    gstarmax = nx.Graph()
    for i in range(0, nm, grsize):
        star = nx.star_graph(grsize - 1)
        mapping = dict(zip(star.nodes(), range(i, i+ grsize)))
        star = nx.relabel_nodes(star, mapping)
        gstarmax.add_edges_from(star.edges())
        # Assign unique group IDs to nodes in each cycle
        group_id = i // grsize  # Calculate the group ID based on the current index
        nx.set_node_attributes(gstarmax, {node: group_id for node in star.nodes()}, name='groupid')
        if i > 0:
            # choose the node with the highest degree in the previous star graph to connect to this star
            prev_center_node = max(list(gstarmax.nodes())[i-grsize:i], key=lambda n: gstarmax.degree[n])
            center_node = max(star.nodes(), key=lambda n: star.degree[n])
            gstarmax.add_edge(prev_center_node, center_node)
    # Connect the last center node to the center node with i=0
    last_center_node = i
    first_center_node = 0
    gstarmax.add_edge(last_center_node, first_center_node)
    gstarmax.remove_edges_from(nx.selfloop_edges(gstarmax))
    netname = 'starmax_' + str(nm)
    netdict[netname] = gstarmax
   
    if nm > grsize:
        #one of the nodes with lowest degree connected
        gstarmin = nx.Graph()
        for i in range(0, nm, grsize):
            star = nx.star_graph(grsize-1)
            mapping = dict(zip(star.nodes(), range(i, i+grsize)))
            star = nx.relabel_nodes(star, mapping)
            gstarmin.add_edges_from(star.edges())
            # Assign unique group IDs to nodes in each cycle
            group_id = i // grsize  # Calculate the group ID based on the current index
            nx.set_node_attributes(gstarmin, {node: group_id for node in star.nodes()}, name='groupid')
            # find the lowest degree node in the current star
            degrees = nx.degree(star)
            sorted_degrees = sorted(degrees, key=lambda x: x[1])
            lowest_degree_node = sorted_degrees[0][0]
            if i > 0:
            # find the lowest degree node in the previous star
                previous_star = nx.subgraph(gstarmin, range(i-grsize, i))
                degrees = nx.degree(previous_star)
                sorted_degrees = sorted(degrees, key=lambda x: x[1])
                lowest_degree_node_previous_star = sorted_degrees[0][0]
                # connect the lowest degree nodes in the current and previous stars
                gstarmin.add_edge(lowest_degree_node, lowest_degree_node_previous_star)
                # Connect the last center node to the center node with i=0
        last_center_node = i + 1
        first_center_node = 1
        gstarmin.add_edge(last_center_node, first_center_node)
        gstarmin.remove_edges_from(nx.selfloop_edges(gstarmin))
        netname = 'starmin_' + str(nm)
        netdict[netname] = (gstarmin)
  
   #generate a wheel based graph where grsize nodes are in wheel and the center node of each wheel is connected to another center node in another wheel
    gwheelmax = nx.Graph()
    for i in range(0, nm, grsize):
        wheel = nx.wheel_graph(grsize)
        mapping = dict(zip(wheel.nodes(), range(i, i+grsize)))
        wheel = nx.relabel_nodes(wheel, mapping)
        gwheelmax.add_edges_from(wheel.edges())
        # Assign unique group IDs to nodes in each cycle
        group_id = i // grsize  # Calculate the group ID based on the current index
        nx.set_node_attributes(gwheelmax, {node: group_id for node in wheel.nodes()}, name='groupid')
        # connect one node from each wheel to one other wheel
        if i > 0:
           # choose the node with the highest degree in the previous star graph to connect to this star
            prev_center_node = max(list(gwheelmax.nodes())[i-grsize:i], key=lambda n: gwheelmax.degree[n])
            center_node = max(wheel.nodes(), key=lambda n: wheel.degree[n])
            gwheelmax.add_edge(prev_center_node, center_node)
    # Connect the last center node to the center node with i=0
    last_center_node = i
    first_center_node = 0
    gwheelmax.add_edge(last_center_node, first_center_node)
    gwheelmax.remove_edges_from(nx.selfloop_edges(gwheelmax))
    netname = 'wheelmax_' + str(nm)
    netdict[netname] = gwheelmax
    
    if nm > grsize:
        #generate a wheel based graph where 5 nodes are in wheel and one of the peripheral nodes of each wheel is connected to another peripheral node in another wheel
        gwheelmin = nx.Graph()
        for i in range(0, nm, grsize):
            wheel = nx.wheel_graph(grsize)
            mapping = dict(zip(wheel.nodes(), range(i, i+grsize)))
            wheel = nx.relabel_nodes(wheel, mapping)
            gwheelmin.add_edges_from(wheel.edges())
            # Assign unique group IDs to nodes in each cycle
            group_id = i // grsize  # Calculate the group ID based on the current index
            nx.set_node_attributes(gwheelmin, {node: group_id for node in wheel.nodes()}, name='groupid')
            # connect one node from each wheel to one other wheel
            degrees = nx.degree(wheel)
            sorted_degrees = sorted(degrees, key=lambda x: x[1])
            lowest_degree_node = sorted_degrees[0][0]
            if i > 0:
                # find the lowest degree node in the previous star
                previous_wheel = nx.subgraph(gwheelmin, range(i-grsize, i))
                degrees = nx.degree(previous_wheel)
                sorted_degrees = sorted(degrees, key=lambda x: x[1])
                lowest_degree_node_previous_wheel = sorted_degrees[0][0]
                # connect the lowest degree nodes in the current and previous wheel
                gwheelmin.add_edge(lowest_degree_node, lowest_degree_node_previous_wheel)
        # Connect the last center node to the center node with i=0
        last_center_node = i + 1
        first_center_node = 1
        gwheelmin.add_edge(last_center_node, first_center_node)
        gwheelmin.remove_edges_from(nx.selfloop_edges(gwheelmin))
        netname = 'wheelmin_' + str(nm)
        netdict[netname] = gwheelmin
   
    #generate a complete graph where grsize nodes are fully connected and one node of each full graph is connected to another node in another full graph
    gfull = nx.Graph()
    for i in range(0, nm, grsize):
        full = nx.complete_graph(grsize)
        mapping = dict(zip(full.nodes(), range(i, i+grsize)))
        full = nx.relabel_nodes(full, mapping)
        gfull.add_edges_from(full.edges())
        # Assign unique group IDs to nodes in each cycle
        group_id = i // grsize  # Calculate the group ID based on the current index
        nx.set_node_attributes(gfull, {node: group_id for node in full.nodes()}, name='groupid')
    # connect one node from each full graph to one other full graph
    for i in range(1, nm+1, grsize):
        gfull.add_edge(i, (i+grsize)% nm)
    gfull.remove_edges_from(nx.selfloop_edges(gfull))
    netname = 'clique_' + str(nm)
    netdict[netname] = gfull
   
    pr +=1

#draw the networks and save them
os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/Figures/Networks')
for key in netdict:
    print(key)
    print(len(netdict[key].nodes))
    net = netdict[key]
    plt.figure()
    plt.title(key)
    nx.draw_networkx(net, node_size=20, font_size=0)
    figname = key + '.pdf'
    plt.savefig(figname)
    plt.close()

"""
To Sobol or not to Sobol? The effects of sampling schemes in systems biology applications

Sobol sequences are a particularly common example of low- discrepancy sequences.
Low-discrepancy sequences (also known as quasi-random sequences) are deterministic sequences of numbers that converge quickly to a uniform distribution. 
Discrepancy refers to a measure of the deviation of a point set from a uniform distribution [16,17]. 
These sequences have better uniformity properties than ran- dom or pseudo-random samples [10]. 
Further, since these sequences are deterministic, computational experiments are much more easily reproducible. 
Identical Sobol samples can be generated without requiring any knowledge other than sample size, 
whereas generating identical random samples requires knowledge of the random number generator and seed.

[10]  S. Kucherenko, D. Albrecht, A. Saltelli, Exploring multi-dimensional spaces: A comparison of latin hypercube and quasi Monte Carlo sampling techniques, 2015,ArXiv, 1505.02350.
[16] H. Niederreiter, CBMS-NSF Regional conference series in applied mathematics: 63, in: RandOm Number Generation and Quasi-Monte Carlo Methods, Society for Industrial and Applied Mathematics, 1992, http://dx.doi.org/10.1137/1. 9781611970081.
[17] M. Drmota, R.F. Tichy, Discrepancy of sequences, in: M. Drmota, R.F. Tichy (Eds.), Sequences, Discrepancies and Applications, in: Lecture Notes in Mathematics, 1651, Springer, 1997, http://dx.doi.org/10.1007/BFb0093404.

Note: reps = 100 works in a shot for pop = 5 and 10, but when pop = 25 and 50 on a MacOS M1 with 32GB memory, one needs to lower
"""


param_ranges = [
    [0.01, 0.1],    # gr = natural growth of resource
    [0.005, 0.1],   # mr = when the resource is to low and generates collapse
    [10, 100],      # nd = level of resource population individuals need for functioning
    [0, 20],        # mg = magintude of the disturbance that may hit the system
    [0, 1],         # ta = mean of normal distribution from which ToM is extracted
    [0.1, 1],       # ts = standard deviation of normal distribution from which Systematizing is extracted
    [0, 1],         # sa = mean of normal distribution from which Systematizing is extracted
    [0.1, 1],       # ss = standard deviation of normal distribution from which Systematizing is extracted
]

#Fixed parameters across all simulations
reps = 100          #repetition of runs with the same parameter space (not used in pilot)
Time = 101          #max time-steps - 101 given how the for loop works, we will have a time 0 and 100 time-steps after that (101 excluded)
R = 1000            #baseline initial state of the resource 

# Create a Sobol instance
sobol = Sobol(d=len(param_ranges))
nsamples = 1024  # Power of 2 for Sobol sequences is better
# Generate Sobol sequences
sobol_samples = sobol.random(nsamples)

# Transform Sobol samples into your parameter space
def transform_sobol_sample(sample, param_space):
    transformed_sample = []
    for i, (param_min, param_max) in enumerate(param_space):
        value = param_min + sample[i] * (param_max - param_min)
        transformed_sample.append(value)
    return transformed_sample

parameter_samples = [transform_sobol_sample(sample, param_ranges) for sample in sobol_samples]

sobol_params = []

parcomb = 0
for sample in parameter_samples:
    # For each Sobol sample, create a set of samples for each network
    for netkey in netdict:
        parcomb += 1
        nr = 0  # Initialize nr for each parcomb
        for rep in range(1, reps+1):
            expanded_sample = list(sample)
            # Append the network key
            expanded_sample.append(netkey)
            net = netdict[netkey]
            # Append a deep copy of the network
            expanded_sample.append(net)
            expanded_sample.append(parcomb)
            nr += 1  # Increment nr for each rep within the same parcomb
            expanded_sample.append(nr)
            sobol_params.append(expanded_sample)

len_samples = len(sobol_params) 
print(len_samples)



#multiprocess routine
num_cores = 8 # or the following to use all cores multiprocessing.cpu_count()
# Run the simulations in parallel
model_result_core = Parallel(n_jobs=num_cores, verbose = 10)(delayed(run_simulation)(*params) for params in sobol_params)
model_results = pd.concat(model_result_core, ignore_index=True)

os.chdir('/Users/jbaggio/Documents/AAA_Study/AAA_Work/A_StressCog_DRMS/StressCog_Model/ModelOutcome')

#pickle and save the resulting model results and theparameters
pickle.dump(model_results, open('Sobolmodel_results' + str(nm) + '.p', 'wb'))
pickle.dump(sobol_params, open('SobolParams' + str(nm) +'.p', 'wb'))



