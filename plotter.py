# plot learned solution corresponding to different number of iterations

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
# tf.compat.v1.disable_eager_execution()
import csv, time
import tikzplotlib
from numpy import genfromtxt

def plot_sol(numiter, datafile1, datafile2, datafile3):
    
    cur_date, cur_time = time.strftime("%m-%d"), time.strftime("%H-%M")
    data_ours = np.load('./sol_files/'+ datafile1)
    data_fbsnns = np.load('./sol_files/'+ datafile2)
    data_noocloss = np.load('./sol_files/'+ datafile3)
    # data_ours['t'], Y_test, Y_test_terminal, Y_pred, errors, legnd = data['t'], data['Y_true'].T,\
    #                                                   data['Y_true_T'], data['Y_pred'], data['rel_err'], data['legnd']
    samples = data_ours['t'].shape[0]
    plt.figure()      
    plt.subplot(2,1,1)
    RE0 = np.linalg.norm(data_ours['Y_pred'][:,0] - 
                         data_ours['Y_true'].T[:,0])/np.linalg.norm(data_ours['Y_true'].T[:,0])
    
    RE = np.linalg.norm(data_ours['Y_pred'] - 
                         data_ours['Y_true'].T)/np.linalg.norm(data_ours['Y_true'].T)
    for i in range(5):  
        plt.plot(data_ours['t'][i,:],data_ours['Y_pred'][i,:],'b',label='Learned u(t,Xt)')
        plt.plot(data_ours['t'][i,:],data_ours['Y_true'].T[i,:],'k--',label='Exact u(t,Xt)')
        plt.xlabel('t')
        plt.ylabel('Yt = u(t,Xt)')
        plt.title('RE = '+str(RE)+',RE0 = '+str(RE0))
    plt.subplot(2,1,2)
    RE0 = np.linalg.norm(data_fbsnns['Y_pred'][:,0] - 
                         data_fbsnns['Y_true'].T[:,0])/np.linalg.norm(data_fbsnns['Y_true'].T[:,0])
    
    RE = np.linalg.norm(data_fbsnns['Y_pred'] - 
                         data_fbsnns['Y_true'].T)/np.linalg.norm(data_fbsnns['Y_true'].T)
    for i in range(5):  
        plt.plot(data_ours['t'][i,:],data_fbsnns['Y_pred'][i,:],'r',label='Learned u(t,Xt)-fbsnn')
        plt.plot(data_ours['t'][i,:],data_fbsnns['Y_true'].T[i,:],'k--',label='Exact u(t,Xt)')
        plt.xlabel('t')
        plt.ylabel('Yt = u(t,Xt)')
        plt.title('RE = '+str(RE)+',RE0 = '+str(RE0))
    plt.subplots_adjust(left=0.1,
                    bottom=0.001, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    file2savefig = './figures/orig/HJB_both_ctrl'+'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) 
    tikzplotlib.save(file2savefig + '.tex')

data_ours = 'July_15/sol_relerr-ours20000-07-15-11-42-29.npz'    
data_fbsnns = 'July_15/sol_relerr-FBSNNs20000-07-15-18-15-06.npz'
data_noocloss = 'July_15/sol_relerr-no-ocloss20000-07-15-20-55-19.npz'
plot_sol(0, data_ours, data_fbsnns, data_noocloss)

data_ours = 'July_15/sol_relerr-ours30000-07-15-14-01-57.npz'    
data_fbsnns = 'July_15/sol_relerr-FBSNNs30000-07-15-18-40-11.npz'
data_noocloss = 'July_15/sol_relerr-no-ocloss30000-07-15-23-10-17.npz'
plot_sol(1, data_ours, data_fbsnns, data_noocloss)

# data_ours = 'July_15/sol_relerr-ours30000-07-15-16-21-39.npz'
# data_fbsnns = 'July_15/sol_relerr-FBSNNs30000-07-15-19-05-16.npz'
# data_noocloss = 'July_15/sol_relerr-no-ocloss30000-07-16-01-25-08.npz'
# plot_sol(2, data_ours, data_fbsnns, data_noocloss)

# data_ours = 'July_15/sol_relerr-ours20000-07-15-17-56-11.npz'
# data_fbsnns = 'July_15/sol_relerr-FBSNNs20000-07-15-19-23-29.npz'
# data_noocloss = 'July_15/sol_relerr-no-ocloss20000-07-16-02-56-24.npz'
# plot_sol(3, data_ours, data_fbsnns, data_noocloss)

# his_ours = genfromtxt('./sol_files/July_15/history/history100D_ctrl-True_alpha-20.0_shift_targ0_date-07-15_time-10-12.csv', delimiter=',')
# his_fbsnns = genfromtxt('./sol_files/July_15/history/history100D_ctrl-False_alpha-0.0_shift_targ0_date-07-15_time-18-00.csv', delimiter=',')
# his_nooloss = genfromtxt('./sol_files/July_15/history/history100D_ctrl-True_alpha-0.0_shift_targ0_date-07-15_time-19-27.csv', delimiter=',')

# for i in range(4):
#     plt.figure()
#     if i==0:
#         plt.semilogy(his_ours[0:200,0],his_ours[0:200,3],'b',label='ours')
#         plt.semilogy(his_ours[0:200,0],his_fbsnns[0:200,3],'r',label='fbsnn')
#         plt.semilogy(his_ours[0:200,0],his_nooloss[0:200,3],'m',label='no-ocloss')
#     elif i==1:
#         plt.semilogy(his_ours[200:500,0],his_ours[200:500,3],'b',label='ours')
#         plt.semilogy(his_ours[200:500,0],his_fbsnns[200:500,3],'r',label='fbsnn')
#         plt.semilogy(his_ours[200:500,0],his_nooloss[200:500,3],'m',label='no-ocloss')
#     elif i==2:
#         plt.semilogy(his_ours[500:800,0],his_ours[500:800,3],'b',label='ours')
#         plt.semilogy(his_ours[500:800,0],his_fbsnns[500:800,3],'r',label='fbsnn')
#         plt.semilogy(his_ours[500:800,0],his_nooloss[500:800,3],'m',label='no-ocloss')
#     elif i==3:
#         plt.semilogy(his_ours[800:1000,0],his_ours[800:1000,3],'b',label='ours')
#         plt.semilogy(his_ours[800:1000,0],his_fbsnns[800:1000,3],'r',label='fbsnn')
#         plt.semilogy(his_ours[800:1000,0],his_nooloss[800:1000,3],'m',label='no-ocloss')
            
#     plt.xlabel('t')
#     plt.ylabel('J')
#     plt.title('OC cost for 100-dimensional Hamilton-Jacobi-Bellman')
#     plt.legend()
#     plt.show()