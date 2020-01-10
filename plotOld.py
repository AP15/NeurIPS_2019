import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from scipy.special import comb
from matplotlib import rc
import matplotlib

def load_data(filename):
    data = np.load(filename)
    return data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']
    
def make_plot(filename, n):
    BL_zero, Delta_zero, Q_zero, Delta_zero_a, Q_zero_a = load_data(filename + 'new_old_a.npz')
    BL_one, Delta_one, Q_one, Delta_one_a, Q_one_a = load_data(filename + 'new_old_a_one.npz')
    BL_two, Delta_two, Q_two, Delta_two_a, Q_two_a = load_data(filename + 'new_old_a_two.npz')
    BL_three, Delta_three, Q_three, Delta_three_a, Q_three_a = load_data(filename + 'new_old_a_three.npz')
    
    data_plt_bl = np.row_stack((BL_zero[0, :], BL_one[0, :], BL_two[0, :], BL_three[0, :]))
    
    data_plt_zero = np.column_stack((Delta_zero[:, 0], Delta_zero[:, 1], Q_zero[:, 0], Q_zero[:, 1]))
    data_plt_one = np.column_stack((Delta_one[:, 0], Delta_one[:, 1], Q_one[:, 0], Q_one[:, 1]))
    data_plt_two = np.column_stack((Delta_two[:, 0], Delta_two[:, 1], Q_two[:, 0], Q_two[:, 1]))
    data_plt_three = np.column_stack((Delta_three[:, 0], Delta_three[:, 1], Q_three[:, 0], Q_three[:, 1]))
        
    Delta_vs_Q(filename, data_plt_bl, data_plt_zero, data_plt_one, data_plt_two, data_plt_three, n)    
    
def Delta_vs_Q(filename, data_plt_bl, data_plt_zero, data_plt_one, data_plt_two, data_plt_three, n):
    
    labels = ['$\eta=0$', '$\eta=0.1$', '$\eta=0.5$', '$\eta=1$']
    
    f = plt.figure(figsize=(10,10))
       
    plt.plot(data_plt_zero[:,2]*comb(n, 2), data_plt_zero[:,0]*comb(n, 2), '-.', color='magenta', linewidth=4, label=labels[0])
    plt.fill_between(data_plt_zero[:,2]*comb(n, 2), data_plt_zero[:,0]*comb(n, 2)-data_plt_zero[:,1]/2*comb(n, 2), data_plt_zero[:,0]*comb(n, 2)+data_plt_zero[:,1]/2*comb(n, 2), 
                     facecolor='magenta', alpha=0.5)
    plt.plot(data_plt_bl[0,1], data_plt_bl[0,0], 'mo', markersize=15, markerfacecolor='None', markeredgewidth=3)

    plt.plot(data_plt_one[:, 2]*comb(n, 2), data_plt_one[:, 0]*comb(n, 2), '-', color='blue', linewidth=4, label=labels[1])
    plt.fill_between(data_plt_one[:,2]*comb(n, 2), data_plt_one[:,0]*comb(n, 2)-data_plt_one[:,1]/2*comb(n, 2), data_plt_one[:,0]*comb(n, 2)+data_plt_one[:,1]/2*comb(n, 2), 
                     facecolor='blue', alpha=0.5)
    plt.plot(data_plt_bl[1,1], data_plt_bl[1,0], 'bo', markersize=15, markerfacecolor='None', markeredgewidth=3)

    plt.plot(data_plt_two[:, 2]*comb(n, 2), data_plt_two[:, 0]*comb(n, 2), '--', color='green', linewidth=4, label=labels[2])    
    plt.fill_between(data_plt_two[:,2]*comb(n, 2), data_plt_two[:,0]*comb(n, 2)-data_plt_two[:,1]/2*comb(n, 2), data_plt_two[:,0]*comb(n, 2)+data_plt_two[:,1]/2*comb(n, 2), 
                     facecolor='green', alpha=0.5)
    plt.plot(data_plt_bl[2,1], data_plt_bl[2,0], 'go', markersize=15, markerfacecolor='None', markeredgewidth=3)

    plt.plot(data_plt_three[:,2]*comb(n, 2), data_plt_three[:,0]*comb(n, 2), ':', color='red', linewidth=4, label=labels[3])
    plt.fill_between(data_plt_three[:,2]*comb(n, 2), data_plt_three[:,0]*comb(n, 2)-data_plt_three[:,1]/2*comb(n, 2), data_plt_three[:,0]*comb(n, 2)+data_plt_three[:,1]/2*comb(n, 2), 
                     facecolor='red', alpha=0.5)    
    plt.plot(data_plt_bl[3,1], data_plt_bl[3,0], 'ro', markersize=15, markerfacecolor='None', markeredgewidth=3)
    
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.ylabel(r'Clustering Cost', fontsize=33)
    plt.xlabel(r'Number of Queries', fontsize=33)
    plt.grid()
    rc('text', usetex=True)
    plt.legend(prop={'size': 25})
    
    #plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['legend.loc'] = 'upper right'

    # use LaTeX fonts in the plot
    #plt.rcParams.update({'font.size': 50, 'legend.fontsize': 80})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.show()

    f.savefig(filename + "new_old_ACCvsKC_scalex2_var.pdf", bbox_inches='tight')
    
#make_plot('skew', 900)
#make_plot('cora', 1879)
#make_plot('sqrt', 900)
#make_plot('landmarks', 266)    
#make_plot('gym', 94)
make_plot('captchas', 244)