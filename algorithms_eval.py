import numpy as np
from sklearn.datasets import fetch_20newsgroups
import exp_utilities as expu

# Build the similarity matrices
n = 1000
S_sin = np.zeros((n, n))

# Run the experiment
# Parameters
rep = 20		
Q_value = 20
p = 1

expu.ex_dataset1('skew', rep, Q_value, p)
expu.ex_dataset1('sqrt', rep, Q_value, p)
expu.ex_dataset1('landmarks', rep, Q_value, p)
expu.ex_dataset1('gym', rep, Q_value, p)
expu.ex_dataset1('captchas', rep, Q_value, p)
expu.ex_dataset1('cora', rep, Q_value, p)
#expu.ex_dataset2('cora', rep)