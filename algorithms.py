from scipy.special import comb
import numpy as np
import math
from numpy.random import randint as usample
from numpy.random import binomial
from numpy.random import hypergeometric

# Algorithms
def KC(S):
    # Check input
    V = list(range(S.shape[0]))
    if (len(V)==0):
        return {}
    
    Clustering = {}
    Q = 0
    k = 1
    while V:
        # Choose pivot
        piv = V[usample(len(V))]
        Clustering[piv], V_left = k, []
        V.remove(piv)
        
        # Build pivot's cluster
        for i in V:
            Q += 1
            if (S[piv, i]==1):
                Clustering[i] = k
            else:
                V_left.append(i)
        k += 1
        V = V_left
        
    return Clustering, Q

def test_graph(V, S, f, n):
    # Test the graph
    t = int(comb(len(V), 2) * f/(n ** 2))
    S_sub = S[np.ix_(V, V)]
    count =  np.count_nonzero(S_sub)/2
    if (t == 0):
        return False 
    else:
        #samples = hypergeometric(count, len(V)*(len(V)-1)/2 - count, t)
        samples = binomial(t, count/(len(V)*(len(V)-1)/2))
        if (samples > 0):
            return True
        else:
            return False

def test_graph_alt(V, S, f, n):
    # Test the graph
    t = int(comb(len(V), 2) * f/(n ** 2))
    S_sub = S[np.ix_(V, V)]
    count =  np.count_nonzero(S_sub)/2
    #print(len(V)*(len(V)-1)/2, count, len(V)*(len(V)-1)/2 - count)
    if (t == 0):
        return False, 0 
    else:
        samples = hypergeometric(count, len(V)*(len(V)-1)/2 - count, t)
        if (samples > 0):
            return True, samples
        else:
            return False, 0

def test_neigh(piv, V, S, f):
    # Test neighbors
    S_sub = S[np.ix_([piv], V)]
    count =  np.count_nonzero(S_sub)
    if (f == 0):
        return False 
    else:
        samples = hypergeometric(count, len(V) - count, f)
        if (samples > 0):
            return True
        else:
            return False 

def ACCESSp_sqrt(S, alpha):
    # Check input
    V = list(range(S.shape[0]))
    if (not V):
        return {}
    
    n = len(V)    
    Clustering, Q, k, f_n, R = {}, 0, 1, (n**alpha), 0
    p = f_n/(n**2)
    while (len(V) >= 1):
        #Test the graph
        if (R<(1/p)):
            Q += int(comb(len(V), 2) * p)
            dump, count = test_graph_alt(V, S, f_n, n)
            R = count/p
            if (R<(1/p)):
                break
    	#Choose pivot
        piv = V[usample(len(V))]
        Clustering[piv], V_left = k, []
        V.remove(piv)
        V_r = len(V)
        #Build pivot's cluster
        f = int(len(V)**alpha)
        if (test_neigh(piv, V, S, f)):
        	Q += len(V)
        	cluster = [V[i] for i in np.nonzero(S[piv, V])[0].tolist()]
        	C_r = len(cluster)
        	for i in cluster:
        		Clustering[i] = k
        	V = [V[i] for i in list(np.where(S[piv, V]==0)[0])]
        else:
        	C_r = 1
        	Q += f
        k += 1
        R -= (comb(C_r, 2) + C_r*(V_r-C_r))
        
    for i in V:
        Clustering[i] = k
        k += 1
            
    return Clustering, Q/comb(n, 2) 

def ACCESS_sqrt(S, alpha):
    # Check input
    V = list(range(S.shape[0]))
    if (not V):
        return {}
    
    n = len(V)
    Clustering, Q, k, f_n = {}, 0, 1, (n**alpha)/3.5
    while (len(V)>=1):
        # Test the graph
        Q += int(comb(len(V), 2) * f_n/(n ** 2)) 
        if (not test_graph(V, S, f_n, n)):
            #print('Stopped with', len(V), 'yet.')
            break
        
        # Choose pivot
        piv = V[usample(len(V))]
        Clustering[piv], V_left = k, []
        V.remove(piv)
        
        #Build pivot's cluster
        f = int(len(V)**alpha)/3.5
        if (test_neigh(piv, V, S, f)):
        	Q += len(V)
        	cluster = [V[i] for i in np.nonzero(S[piv, V])[0].tolist()]
        	for i in cluster:
        		Clustering[i] = k
        	V = [V[i] for i in list(np.where(S[piv, V]==0)[0])]
        else:
        	Q += f
        k += 1
        
    for i in V:
        Clustering[i] = k
        k += 1
            
    return Clustering, Q/comb(n, 2) 

def ACC_sqrt(S, B, alpha):
    # Check input
    V = list(range(S.shape[0]))
    if (not V):
        return {}

    n = len(V)
    
    Clustering, Q, k, phi = {}, 0, 1, 0
    while ((phi<B) and (len(V)>=1)):
        # Choose pivot
        piv = V[usample(len(V))]
        Clustering[piv], V_left = k, []
        V.remove(piv)
        
        #Build pivot's cluster
        f = int(len(V)**alpha)
        if (test_neigh(piv, V, S, f)):
        	Q += len(V)
        	cluster = [V[i] for i in np.nonzero(S[piv, V])[0].tolist()]
        	for i in cluster:
        		Clustering[i] = k
        	V = [V[i] for i in list(np.where(S[piv, V]==0)[0])]
        else:
        	Q += f
        
        k += 1
        phi += 1
        
    for i in V:
        Clustering[i] = k
        k += 1
            
    return Clustering, Q/comb(n, 2) 

def ACR_sqrt(S, B, alpha, prob):
    # Check input
    V = list(range(S.shape[0]))
    n = len(V)
    if (n==0):
        return {}
    
    Clustering, C_f, Q, r = {}, np.zeros((n, n)), 0, math.ceil(128*np.log(n/prob))
    r = 3
    for i in range(r):
    	C, Q_temp = ACC_sqrt(S, B, alpha)
    	Q += Q_temp
    	for u in range(n):
    		for v in range(n):
    			if(C[u]==C[v] and u!=v):
    				C_f[u,v] += 1/r
    k = 1
    while (V):
        u = V.pop()
        Clustering[u] = k
        V_temp = []
        for v in V:
            if (C_f[u,v] > 0.5):
                Clustering[v] = k
            else:
                V_temp.append(v)
        V = V_temp
        k += 1
    return Clustering, Q/comb(n, 2) 