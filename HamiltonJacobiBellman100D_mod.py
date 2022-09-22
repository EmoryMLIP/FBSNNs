import numpy as np
import tensorflow as tf
from Neural_SOC import NeuralSOC
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
import csv, time, gc

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
    
    W = np.random.normal(size=(MC,NC,D)) # MC x NC x D
    
    return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0*np.abs(T-t))*W, X_targ)),axis=0))

# plot learned solution corresponding to different number of iterations
def plotting(model, numiter):
    
    cur_date, cur_time = time.strftime("%m-%d"), time.strftime("%H-%M")
    t_test, W_test = model.fetch_minibatch()
    
    X_pred, Y_pred, his = model.predict(Xi, t_test, W_test)   
    
    content  = np.array([his])
    # save optimal cost while testing
    file2store_OC = './history/OC_cost_test_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha) + '_shift_targ' + str(shift_targ)+'_date-' + cur_date + '_time-' + cur_time + '.csv'
    with open(file2store_OC, 'a') as record_append:
        np.savetxt(record_append, np.asarray([content[0]]), delimiter=',')
    
    Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:], model.x_target)
    
    Y_test_terminal = np.log(0.5 + 0.5*np.sum((X_pred[:,-1,:]- model.x_target)**2, axis=1, keepdims=True))
    plt.figure()
    plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned u(t,Xt)')
    # plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
    plt.plot(t_test[0,:,0].T,Y_test[:,0].T,'r--',label='Exact u(t,Xt)')
    plt.plot(t_test[0:1,-1,0],Y_test_terminal[0:1,0],'ks',label='YT = u(T,XT)')
    # plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
    plt.plot([0],Y_test[0,0],'ko',label='Y0 = u(0,X0)')
    plt.xlabel('t')
    plt.ylabel('Yt = u(t,Xt)')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    plt.legend()
    
    file2savefig = './figures/orig/HJB_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha) + '_shift_targ' + str(shift_targ) +'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) + '.png'
    plt.savefig(file2savefig , dpi = 100)
    
    samples = 5
    errors= np.reshape(np.sqrt((Y_test-Y_pred[0,:,:])**2/Y_test**2),[-1,1])
    for i in range(1,samples):
        Y_test = u_exact(t_test[i,:,:], X_pred[i,:,:], model.x_target)
        errors= np.append( errors, np.reshape(np.sqrt((Y_test-Y_pred[i,:,:])**2/Y_test**2),[-1,1]),axis=1)
        plt.figure()
        plt.plot(t_test[i:i+1,:,0].T,Y_pred[i:i+1,:,0].T,'b',label='Learned u(t,Xt)')
        # plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
        plt.plot(t_test[i,:,0].T,Y_test[:,0].T,'r--',label='Exact u(t,Xt)')
        plt.plot(t_test[i:i+1,-1,0],Y_test_terminal[i:i+1,0],'ks',label='YT = u(T,XT)')
        # plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
        plt.plot([0],Y_test[0,0],'ko',label='Y0 = u(0,X0)')
        plt.xlabel('t')
        plt.ylabel('Yt = u(t,Xt)')
        plt.title('100-dimensional Hamilton-Jacobi-Bellman')
        plt.legend()
        file2savefig = './figures/orig/HJB_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha) + '_shift_targ' + str(shift_targ) +'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) + '_i-' + str(i) + '.png'
        plt.savefig(file2savefig , dpi = 100)
    mean_err = np.mean(errors,1)
    std_err = np.std(errors,1)
    
    
    # if model.ctrl_dyn:
    #     plt.savefig('./figures/HJB_Jan_13_50_ours_' + str(numiter), dpi = 100)
    # elif model.ctrl_dyn !=1:
    #     plt.savefig('./figures/HJB_Jan_13_50_Raissis_' + str(numiter), dpi = 100)
    # plt.clf()
    
    
    plt.figure()
    plt.plot(t_test[0,:,0],mean_err,'b')
    plt.fill_between(t_test[0,:,0], mean_err-std_err, mean_err+std_err, alpha = 0.5)
    # plt.plot(t_test[0,:,0],errors,'b')
    plt.xlabel('t')
    plt.ylabel('relative error')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    # plt.legend()
    file2savefig = './figures/orig/HJB_rel-error_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha)+ '_shift_targ' + str(shift_targ)+'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter)  + '.png'
    plt.savefig(file2savefig , dpi = 100)
    # if model.ctrl_dyn:
    #     plt.savefig('./figures/HJB_Jan_13_50_errors_ours' + str(numiter), dpi = 100)
    # elif model.ctrl_dyn!=1:
    #     plt.savefig('./figures/HJB_Jan_13_50_errors_Raissis' + str(numiter), dpi = 100)
    # plt.clf()
    
    
    # using non zero control
    if model.ctrl_dyn !=1:
        model.ctrl_dyn = 1
        X_pred, Y_pred, his = model.predict(Xi, t_test, W_test)
        content  = np.array([his[0]])
        file2store_OC = './history/OC_cost_test_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha)+ '_shift_targ' + str(shift_targ)+'_date-' + cur_date + '_time-' + cur_time + '.csv'
        with open(file2store_OC, 'a') as record_append:
            np.savetxt(record_append, np.asarray([content[0]]), delimiter=',')
            
        Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:], model.x_target)
        
        Y_test_terminal = np.log(0.5 + 0.5*np.sum((X_pred[:,-1,:]- model.x_target)**2, axis=1, keepdims=True))
        
        plt.figure()
        plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned u(t,Xt)')
        # plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
        plt.plot(t_test[0,:,0].T,Y_test[:,0].T,'r--',label='Exact u(t,Xt)')
        plt.plot(t_test[0:1,-1,0],Y_test_terminal[0:1,0],'ks',label='YT = u(T,XT)')
        # plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
        plt.plot([0],Y_test[0,0],'ko',label='Y0 = u(0,X0)')
        plt.xlabel('t')
        plt.ylabel('Yt = u(t,Xt)')
        plt.title('100-dimensional Hamilton-Jacobi-Bellman')
        plt.legend()
        
        file2savefig = './figures/mod/HJB_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha) + '_shift_targ' + str(shift_targ) +'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) + '.png'
        plt.savefig(file2savefig , dpi = 100)
        
        samples = 5
        errors= np.reshape(np.sqrt((Y_test-Y_pred[0,:,:])**2/Y_test**2),[-1,1])
        for i in range(1,samples):
            Y_test = u_exact(t_test[i,:,:], X_pred[i,:,:], model.x_target)
            errors= np.append( errors, np.reshape(np.sqrt((Y_test-Y_pred[i,:,:])**2/Y_test**2),[-1,1]),axis=1)
            plt.figure()
            plt.plot(t_test[i:i+1,:,0].T,Y_pred[i:i+1,:,0].T,'b',label='Learned u(t,Xt)')
            # plt.plot(t_test[1:5,:,0].T,Y_pred[1:5,:,0].T,'b')
            plt.plot(t_test[i,:,0].T,Y_test[:,0].T,'r--',label='Exact u(t,Xt)')
            plt.plot(t_test[i:i+1,-1,0],Y_test_terminal[i:i+1,0],'ks',label='YT = u(T,XT)')
            # plt.plot(t_test[1:5,-1,0],Y_test_terminal[1:5,0])
            plt.plot([0],Y_test[0,0],'ko',label='Y0 = u(0,X0)')
            plt.xlabel('t')
            plt.ylabel('Yt = u(t,Xt)')
            plt.title('100-dimensional Hamilton-Jacobi-Bellman')
            plt.legend()
            file2savefig = './figures/mod/HJB_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha) + '_shift_targ' + str(shift_targ) +'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) + '_i-' + str(i) + '.png'
            plt.savefig(file2savefig , dpi = 100)
        mean_err = np.mean(errors,1)
        std_err = np.std(errors,1)
        
        
        # if model.ctrl_dyn:
        #     plt.savefig('./figures/HJB_Jan_13_50_ours_' + str(numiter), dpi = 100)
        # elif model.ctrl_dyn !=1:
        #     plt.savefig('./figures/HJB_Jan_13_50_Raissis_' + str(numiter), dpi = 100)
        # plt.clf()
        
        
        plt.figure()
        plt.plot(t_test[0,:,0],mean_err,'b')
        plt.fill_between(t_test[0,:,0], mean_err-std_err, mean_err+std_err, alpha = 0.5)
        # plt.plot(t_test[0,:,0],errors,'b')
        plt.xlabel('t')
        plt.ylabel('relative error')
        plt.title('100-dimensional Hamilton-Jacobi-Bellman')
        # plt.legend()
        file2savefig = './figures/mod/HJB_rel-error_ctrl-'+str(model.ctrl_dyn) + '_alpha-' + str(model.alpha)+ '_shift_targ' + str(shift_targ)+'_date-' + cur_date + '_time-' + cur_time + '_iter-' + str(numiter) + '.png'
        plt.savefig(file2savefig , dpi = 100)
        # if model.ctrl_dyn:
        #     plt.savefig('./figures/HJB_Jan_13_50_errors_ours' + str(numiter), dpi = 100)
        # elif model.ctrl_dyn!=1:
        #     plt.savefig('./figures/HJB_Jan_13_50_errors_Raissis' + str(numiter), dpi = 100)
        # plt.clf()
        model.ctrl_dyn = 0



if __name__ == "__main__":
    
    
    plt.close('all')
    M = 100 # number of trajectories (batch size)
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
    model_ours.train(N_Iter=n_iter[1], learning_rate=1e-4)
    plotting(model_ours,n_iter[1])
    model_ours.train(N_Iter=n_iter[2], learning_rate=1e-5)
    plotting(model_ours,n_iter[2])
    # model_ours.train(N_Iter=n_iter[3], learning_rate=1e-6)
    
    del model_ours
    gc.collect()

    # Training: FBSNNs
    model_Raissi = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers, sigma_val, False, False, 0.0, betas, shift_targ = shift_targ)

    model_Raissi.train(N_Iter=n_iter[0], learning_rate=1e-3)
    model_Raissi.train(N_Iter=n_iter[1], learning_rate=1e-4)
    plotting(model_Raissi,n_iter[1])
    model_Raissi.train(N_Iter=n_iter[2], learning_rate=1e-5)
    plotting(model_Raissi,n_iter[2])
    # model_Raissi.train(N_Iter=n_iter[3], learning_rate=1e-6)
    
    del model_Raissi
    gc.collect()
    
    
    # Training: oc_loss, no ctrl
    model_oc_loss = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers, sigma_val, False, True, alpha, betas, shift_targ = shift_targ)

    model_oc_loss.train(N_Iter=n_iter[0], learning_rate=1e-3)
    model_oc_loss.train(N_Iter=n_iter[1], learning_rate=1e-4)
    plotting(model_oc_loss,n_iter[1])
    model_oc_loss.train(N_Iter=n_iter[2], learning_rate=1e-5)
    plotting(model_oc_loss,n_iter[2])
    # model_oc_loss.train(N_Iter=n_iter[3], learning_rate=1e-6)
    
    del model_oc_loss
    gc.collect()
    
    # Training: no oc_loss, ctrl
    model_ctrl = HamiltonJacobiBellman(Xi, T,
                                  M, N, D,
                                  layers, sigma_val, True, False, 0.0, betas, shift_targ = shift_targ)

    model_ctrl.train(N_Iter=n_iter[0], learning_rate=1e-3)
    model_ctrl.train(N_Iter=n_iter[1], learning_rate=1e-4)
    plotting(model_ctrl,n_iter[1])
    model_ctrl.train(N_Iter=n_iter[2], learning_rate=1e-5)
    plotting(model_ctrl,n_iter[2])
    # model_ctrl.train(N_Iter=n_iter[3], learning_rate=1e-6)
    