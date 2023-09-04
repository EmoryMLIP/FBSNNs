import numpy as np
import tensorflow as tf
from Neural_SOC import NeuralSOC
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
import csv, time, gc, os
import pickle as pl

class HamiltonJacobiBellman(NeuralSOC):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, sigma_val, ctrl_dyn, oc_loss, alpha, betas, shift_targ = 0):
        # alpha: 0 sets oc cost weight to zero
        # oc_cost_weight: weight to the oc cost
        # ctrl_dyn: 1 or 0 to have control in the dynamics or not 
        
        self.x_target  = 0*np.ones([1,D])
        if shift_targ:
            self.x_target  = 2*np.ones([1,D])
        self.sigma_val = sigma_val
        self.ctrl_dyn = ctrl_dyn
        self.oc_loss = oc_loss
        if not os.path.exists('./history'):
            os.makedirs('./history')
        self.file2store = './history/history100D_ctrl-'+str(ctrl_dyn) + '_alpha-' + str(alpha) + '_shift_targ' + str(shift_targ) +'_date-' + time.strftime("%m-%d") + '_time-' + time.strftime("%H-%M") + '.csv'


                
        super().__init__(Xi, T,
                         M, N, D,
                         layers, alpha, betas)
        
        
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        if self.ctrl_dyn:    
            return -tf.reduce_sum(Z**2, 1, keepdims = True) # M x 1, return zero f, for non zero control
        elif  self.ctrl_dyn!=1:
            return tf.reduce_sum(input_tensor=Z**2, axis=1, keepdims = True) # M x 1, for zero control
    
    def g_tf(self, X): # M x D
        return tf.math.log(0.5 + 0.5*tf.reduce_sum(input_tensor=(X - self.x_target)**2, axis=1, keepdims = True)) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        if self.ctrl_dyn:    
            return -2*Z # M x D, non zero mu (control)
        elif  self.ctrl_dyn!=1:
             return super().mu_tf_zero(t, X, Y, Z) # M x D, zero mu (control)
        
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return self.sigma_val*super().sigma_tf(t, X, Y) # M x D x D
    
    def phi_tf_zero(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x 
        return tf.reduce_sum(input_tensor=Z**2, axis=1, keepdims = True) # M x 1, for zero control
    
    def mu_tf_zero(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf_zero(t, X, Y, Z) # M x D, zero mu (control)
    ###########################################################################

# Define exact solution 
def g(X, X_targ): # MC x NC x D
        return np.log(0.5 + 0.5*np.sum((X - X_targ)**2, axis=2, keepdims=True)) # MC x N x 1
        
def u_exact(t, X, X_targ): # NC x 1, NC x D
    MC = 10**5
    NC = t.shape[0]
    
    np.random.seed(0)
    W = np.random.normal(size=(MC,NC,D)) # MC x NC x D
    
    return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0*np.abs(T-t))*W, X_targ)),axis=0))

# get learned solution and relative error corresponding to different number of iterations in 1st dim
def get_Phi(model, numiter, legnd, samples=5, save_sol = 0):
    cur_date = time.strftime("%m-%d-%H-%M-%S")
    t_test, W_test = model.fetch_minibatch()

    X_pred, Y_pred, his = model.predict(Xi, t_test, W_test)
    Y_test = u_exact(t_test[0, :, :], X_pred[0, :, :], model.x_target)

    Y_test_terminal = np.log(0.5 + 0.5 * np.sum((X_pred[:, -1, :] - model.x_target) ** 2, axis=1, keepdims=True))

    Y_true= Y_test
    errors = np.reshape(np.sqrt((Y_test - Y_pred[0, :, :]) ** 2 / Y_test ** 2), [-1, 1])
    for i in range(1, samples):
        Y_test = u_exact(t_test[i, :, :], X_pred[i, :, :], model.x_target)
        Y_true = np.append(Y_true,Y_test, axis = 1)
        errors = np.append(errors, np.reshape(np.sqrt((Y_test - Y_pred[i, :, :]) ** 2 / Y_test ** 2), [-1, 1]), axis=1)

    if save_sol != 1:
        return t_test, Y_pred, Y_true, Y_test_terminal, errors
    else:
        if not os.path.exists('./sol_files'):
            os.makedirs('./sol_files')
        np.savez('./sol_files/sol_relerr-' + legnd + str(numiter) + "-" + cur_date, t = t_test[:samples,:,0], Y_true = Y_true,
                 Y_true_T = Y_test_terminal[:samples,0], Y_pred = Y_pred[:samples,:,0], rel_err = errors,
                 oc_loss = np.array(his[0]))


# plot learned solution corresponding to different number of iterations
def plotting(model, numiter, datafile):
    cur_date, cur_time = time.strftime("%m-%d"), time.strftime("%H-%M")
    data = np.load(datafile)
    t_test, Y_test, Y_test_terminal, Y_pred, errors = data['t'], data['Y_true'].T,\
                                                      data['Y_true_T'], data['Y_pred'], data['rel_err']
    samples = t_test.shape[0]
    for i in range(samples):
        plt.figure()
        plt.plot(t_test[i,:],Y_pred[i,:],'b',label='Learned u(t,Xt)')
        # plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
        plt.plot(t_test[i,:],Y_test[i,:],'r--',label='Exact u(t,Xt)')
        plt.plot(t_test[i,-1],Y_test_terminal[i],'ks',label='YT = u(T,XT)')
        # plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
        plt.plot([0],Y_test[i,0],'ko',label='Y0 = u(0,X0)')
        plt.xlabel('t')
        plt.ylabel('Yt = u(t,Xt)')
        plt.title('100-dimensional Hamilton-Jacobi-Bellman')
        plt.legend()
        file2savefig = './figures/comparision/HJB_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha) + '_shift_targ' + str(shift_targ) +'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) + '_i-' + str(i) + '.png'
        if not os.path.exists('./figures/comparision/'):
            os.makedirs('./figures/comparision/')
        plt.savefig(file2savefig , dpi = 100)

    mean_err = np.mean(errors, 1)
    std_err = np.std(errors, 1)
    plt.figure()
    plt.plot(t_test[0,:],mean_err,'b')
    plt.fill_between(t_test[0,:], mean_err-std_err, mean_err+std_err, alpha = 0.5)
    # plt.plot(t_test[0,:,0],errors,'b')
    plt.xlabel('t')
    plt.ylabel('relative error')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    # plt.legend()
    file2savefig = './figures/comparision/HJB_rel-error_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha)+ '_shift_targ' + str(shift_targ)+'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter)  + '.png'
    plt.savefig(file2savefig , dpi = 100)

   


if __name__ == "__main__":
    
    # set seed 
    # tf.compat.v1.set_random_seed(1)
    # tf.random.set_seed(1)
    #
    legnd = ['ours', 'FBSNNs', 'no-ocloss']
    
    plt.close('all')
    M = 64 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 100 # number of dimensions
    
    layers = [D+1] + 4*[256] + [1]

    Xi = np.zeros([1,D])
    T = 1.0
    sigma_val = tf.sqrt(2.0)
     
    # shifted target problem
    shift_targ = 0
    # to add oc cost and weight in the loss function
    alpha = 20.0
    betas = [1.0, 1.0, 1.0]
    n_iter = [2*10**4, 3*10**4, 3*10**4, 2*10**4]
    model_ours = HamiltonJacobiBellman(Xi, T,
                                       M, N, D,
                                       layers, sigma_val, True, True, alpha, betas, shift_targ = shift_targ)
    

    model_ours.train(N_Iter=n_iter[0], learning_rate=1e-3)
    get_Phi(model_ours, n_iter[0], legnd[0], samples=10, save_sol=1)
    model_ours.train(N_Iter=n_iter[1], learning_rate=1e-4)
    get_Phi(model_ours, n_iter[1], legnd[0], samples=10, save_sol=1)
    model_ours.train(N_Iter=n_iter[2], learning_rate=1e-5)
    get_Phi(model_ours, n_iter[2], legnd[0], samples=10, save_sol=1)
    model_ours.train(N_Iter=n_iter[3], learning_rate=1e-6)
    get_Phi(model_ours, n_iter[3], legnd[0], samples=10, save_sol=1)
    
    del model_ours
    gc.collect()

    # Training: FBSNNs
    model_Raissi = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers, sigma_val, False, False, 0.0, betas, shift_targ = shift_targ)

    model_Raissi.train(N_Iter=n_iter[0], learning_rate=1e-3)
    get_Phi(model_Raissi, n_iter[0], legnd[1], samples=10, save_sol=1)
    model_Raissi.train(N_Iter=n_iter[1], learning_rate=1e-4)
    get_Phi(model_Raissi, n_iter[1], legnd[1], samples=10, save_sol=1)
    model_Raissi.train(N_Iter=n_iter[2], learning_rate=1e-5)
    get_Phi(model_Raissi, n_iter[2], legnd[1], samples=10, save_sol=1)
    model_Raissi.train(N_Iter=n_iter[3], learning_rate=1e-6)
    get_Phi(model_Raissi, n_iter[3], legnd[1], samples=10, save_sol=1)
    
    del model_Raissi
    gc.collect()
    
    
    model_ctrl = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers, sigma_val, True, False, 0.0, betas, shift_targ = shift_targ)

    model_ctrl.train(N_Iter=n_iter[0], learning_rate=1e-3)
    get_Phi(model_ctrl, n_iter[0], legnd[2], samples=10, save_sol=1)
    model_ctrl.train(N_Iter=n_iter[1], learning_rate=1e-4)
    get_Phi(model_ctrl, n_iter[1], legnd[2], samples=10, save_sol=1)
    model_ctrl.train(N_Iter=n_iter[2], learning_rate=1e-5)
    get_Phi(model_ctrl, n_iter[2], legnd[2], samples=10, save_sol=1)
    model_ctrl.train(N_Iter=n_iter[3], learning_rate=1e-6)
    get_Phi(model_ctrl, n_iter[3], legnd[2], samples=10, save_sol=1)
    del model_ctrl
    gc.collect()
    
    