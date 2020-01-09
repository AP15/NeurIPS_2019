import numpy as np
from numpy import loadtxt
from numpy.random import binomial
from scipy.special import comb

from algorithms import KC
from algorithms import ACC_sqrt
from algorithms import ACCESS_sqrt
from algorithms import ACCESSp_sqrt
from algorithms import ACR_sqrt

# Utilities
def create_sim(filename):
    gold = loadtxt(filename).astype(int)
    n = int(gold.shape[0])
    S = np.zeros((n, n))

    count = 0
    for i in range(n):
        for j in range(n):
            if (gold[i, 1]==gold[j, 1] and i!=j):
                S[i, j] = 1
                count += 1
            else:
                S[i, j] = 0

    noise = (count/2)/comb(n, 2) 
    return S, noise, gold

def add_noise(S, p):
    n = S.shape[0]
    S_N = S

    for i in range(n):
        for j in range(i+1, n):
            flip = binomial(1, p)
            if (flip==1):
                if (S[i, j]==1):
                    S_N[i, j] = 0
                    S_N[j, i] = 0                    
                else:
                    S_N[i, j] = 1
                    S_N[j, i] = 1
    return S_N                

def compute_cost(S, Clustering):
    cost = 0
    n = S.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if ((Clustering[i]==Clustering[j]) and (S[i, j]==0)):
                cost += 1
            elif ((Clustering[i]!=Clustering[j]) and (S[i, j]==1)):
                cost += 1
    return cost

def jaccard_rec(C, Clustering):
    cl_idx = set(Clustering.values())
    jaccard = 0
    max_id = 0
    for idx in cl_idx:
        cluster = set([k for k,v in Clustering.items() if v == idx])
        jaccard_temp = len(C.intersection(cluster))/len(C.union(cluster))
        if (jaccard_temp > jaccard):
            jaccard = jaccard_temp
            max_id = idx
    #print('GT:', C)
    #print('Cluster:', set([k for k,v in Clustering.items() if v == max_id]))
    #print(jaccard)
    return jaccard

# Approximation ratio
def ex_approximation(S, rep, Q_value, p, noise=False, noise_rate=0):
    Delta_KC, Delta_ACC, Delta_ACCESS = np.zeros(rep), np.zeros((Q_value, rep)), np.zeros((Q_value, rep))
    Q_KC, Q_ACC, Q_ACCESS = np.zeros(rep), np.zeros((Q_value, rep)), np.zeros((Q_value, rep))
    
    n = int(S.shape[0])
    alpha = np.linspace(0, p, Q_value)
    B = n**alpha

    if (noise):
        S_temp = add_noise(S, noise_rate)
        S = S_temp

    # KC
    for i in range(rep):
        Clustering_KC, Q_KC[i] = KC(S)
        Delta_KC[i] = compute_cost(S, Clustering_KC)

    Delta_BL = np.mean(Delta_KC) * np.ones(Q_value)
    Q_BL = np.mean(Q_KC) *np.ones(Q_value)

    # ACC & ACCESS
    for q in range(0, Q_value):
        print('n = ' +str(n) + '. Tested Valued:', round(B[q]))
        for i in range(rep):
            Clustering_ACC, Q_ACC[q, i] = ACC_sqrt(S, round(B[q])-1, alpha[q])
            #Clustering_ACC, Q_ACC[q, i] = ACCESS_sqrt(S, alpha[q])
            Delta_ACC[q, i] = compute_cost(S, Clustering_ACC)/comb(n, 2) 
            #Clustering_ACCESS, Q_ACCESS[q, i] = ACCESS_sqrt(S, alpha[q])
            #Delta_ACCESS[q, i] = compute_cost(S, Clustering_ACCESS)/comb(n, 2) 

    Delta, Delta_var, Q, Q_var = np.mean(Delta_ACC, axis=1), np.mean(Delta_ACCESS, axis=1), \
                np.mean(Q_ACC, axis=1), np.mean(Q_ACCESS, axis=1)
    Delta_std, Delta_var_std, Q_std, Q_var_std = np.std(Delta_ACC, axis=1), np.std(Delta_ACCESS, axis=1), \
                np.std(Q_ACC, axis=1), np.std(Q_ACCESS, axis=1)
    
    return B, np.column_stack((Delta_BL, Q_BL)), np.column_stack((Delta, Delta_std)), \
                np.column_stack((Delta_var, Delta_var_std)), np.column_stack((Q, Q_std)), np.column_stack((Q_var, Q_var_std))

def ex_dataset1(dataset, rep, Q_value, p):
    S, density, dummy = create_sim('Data/' + dataset + '/gold.txt')
    print('Dataset: '+ dataset +'. Density: ', density)
    B, BL, Delta, Delta_var, Q, Q_var = ex_approximation(S, rep, Q_value, p)
    np.savez(dataset + 'new_old_1', B, BL, Delta, Q, Delta_var, Q_var)
    print('p = 0 - Done.')
    
    # Noise 
    B, BL, Delta, Delta_var, Q, Q_var = ex_approximation(S, rep, Q_value, p, True, 0.1*density)
    print('0.1 - Done.')
    np.savez(dataset + 'new_old_1_one', B, BL, Delta, Q, Delta_var, Q_var)

    B, BL, Delta, Delta_var, Q, Q_var = ex_approximation(S, rep, Q_value, p, True, 0.5*density)
    print('0.2 - Done.')
    np.savez(dataset + 'new_old_1_two', B, BL, Delta, Q, Delta_var, Q_var)

    B, BL, Delta, Delta_var, Q, Q_var = ex_approximation(S, rep, Q_value, p, True, 1*density)
    print('0.3 - Done.')
    np.savez(dataset + 'new_old_1_three', B, BL, Delta, Q, Delta_var, Q_var)

# Cluster Recovery
def ex_clrec(gt, S, rep, noise=False, noise_rate=0):
    range_k = set(gt[:, 1])
    avg_Jaccard_ACC, avg_Jaccard_ACR = np.zeros(len(range_k)), np.zeros(len(range_k))
    avg_Jaccard_ACC_rep, avg_Jaccard_ACR_rep = np.zeros((len(range_k), rep)), np.zeros((len(range_k), rep))

    # Sort Cluster Size
    clusters, sizes = np.unique(gt[:, 1], return_counts=True)
    gt_clusters_sorted = clusters[np.argsort(sizes)][::-1]+1

    if (noise):
        S_temp = add_noise(S, noise_rate)
        S = S_temp

    # ACC & ACR
    B = int(S.shape[0])**(1/2)
    for i in range(rep):
        Clustering_ACC = ACC_sqrt(S, round(B)-1, 1/2)[0]
        Clustering_ACR = ACR_sqrt(S, round(B)-1, 1/2, 0.1)[0]
        j = 0
        while ((j < len(range_k)) and (j < 10)):
            print('Top_'+str(j+1))
            temp_Jaccard_ACC, temp_Jaccard_ACR = 0, 0
            for l in range(j+1):
                #print('top_' + str(j+1), 'idx:', gt_clusters_sorted[l]-1)
                C = set(gt[gt[:, 1]==(gt_clusters_sorted[l]-1), 0])
                temp_Jaccard_ACC += jaccard_rec(C, Clustering_ACC)
                temp_Jaccard_ACR += jaccard_rec(C, Clustering_ACR)
            avg_Jaccard_ACC_rep[j, i], avg_Jaccard_ACR_rep[j, i] = temp_Jaccard_ACC/(j+1), temp_Jaccard_ACR/(j+1)
            j += 1
    
    avg_Jaccard_ACC_rep, avg_Jaccard_ACR_rep = np.mean(avg_Jaccard_ACC_rep, axis=1), np.mean(avg_Jaccard_ACR_rep, axis=1)
    return range_k, avg_Jaccard_ACC_rep, avg_Jaccard_ACR_rep

def ex_dataset2(dataset, rep):
    S, density, gt = create_sim('Data/' + dataset + '/gold.txt')
    # print('Dataset: '+ dataset +'. Density: ', density)
    # range_k, avg_Jaccard_ACC, avg_Jaccard_ACR = ex_clrec(gt, S, rep)
    # np.savez(dataset + '_cr', range(1, len(range_k)+1), avg_Jaccard_ACC, avg_Jaccard_ACR)
    print('Done.')
    #print(avg_Jaccard_ACC)
    #print(avg_Jaccard_ACR)
    
    # Noise 
    range_k, avg_Jaccard_ACC, avg_Jaccard_ACR = ex_clrec(gt, S, rep, True, 0.2*density)
    print('0.1 - Done.')
    np.savez(dataset + '_cr_one', range(1, len(range_k)+1), avg_Jaccard_ACC, avg_Jaccard_ACR)

    # B, BL, Delta, Delta_var, Q, Q_var = ex_clrec(S, rep, Q_value, p, True, 0.2*density)
    # print('0.2 - Done.')
    # np.savez(dataset + '_a_two', B, BL, Delta, Q, Delta_var, Q_var)

    # B, BL, Delta, Delta_var, Q, Q_var = ex_clrec(S, rep, Q_value, p, True, 0.3*density)
    # print('0.3 - Done.')
    # np.savez(dataset + '_a_three', B, BL, Delta, Q, Delta_var, Q_var)