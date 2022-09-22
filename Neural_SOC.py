import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
import csv
from abc import abstractmethod
import sys
import time
# setting previous directory to path
# sys.path.append('/Users/lruthot/Google Drive/FBSNNs/')
# sys.path.append('FBSNNs')

from FBSNNs import FBSNN


class NeuralSOC(FBSNN):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers,alpha,betas):
        self.alpha = alpha
        self.betas = betas

        super().__init__(Xi, T,
                         M, N, D,
                         layers)
        
        self.loss, self.X_pred, self.Y_pred, self.Y0_pred, self.his = self.oc_loss_function(self.t_tf, self.W_tf, self.Xi_tf)

        self.X_predsig0, self.Y_predsig0 = self.get_solsig0(self.t_tf, self.W_tf, self.Xi_tf)
        
    def oc_loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D

            t0 = t[:,0,:]

            W0 = W[:,0,:]
            X0 = tf.tile(Xi,[self.M,1]) # M x D
            Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D
            
            # set up cost
            cost_L = tf.reduce_sum(input_tensor=tf.square(Z0))
            cost_TD = 0.0

            X_list = [X0]
            Y_list = [Y0]

            for n in range(0,self.N):
                t1 = t[:,n+1,:]
                W1 = W[:,n+1,:]
                # X1 = X0 + sigma*dW
                X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1])
                Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)* (t1-t0) + tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims=True)
                Y1, Z1 = self.net_u(t1,X1)
                
                #not correct?
                # cost_TD += tf.square(tf.norm((Y1 - Y1_tilde)*(t1-t0)))
                # this mactches with FBSNNs
                cost_TD += tf.reduce_sum(tf.square(Y1 - Y1_tilde))
                
                # running cost
                # cost_L += tf.square(tf.norm(Z1*(t1-t0)))
                cost_L +=  tf.reduce_sum(input_tensor=tf.square(Z1)*(t1-t0)) 
                t0 = t1
                W0 = W1
                X0 = X1
                Y0 = Y1
                Z0 = Z1
                
                X_list.append(X0)
                Y_list.append(Y0)
                
            cost_HJBfin     = tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
            cost_gradHJBfin = tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))
            
            # optimal control cost
            oc_cost = (cost_L + tf.reduce_sum(self.g_tf(X1)))
            
            # loss function
            loss =  self.alpha*oc_cost + self.betas[0]* cost_TD + self.betas[1]*cost_HJBfin + self.betas[2]*cost_gradHJBfin
            # if self.oc_loss:
            #     loss += self.alpha*oc_cost
            his = [oc_cost, cost_TD, cost_HJBfin, cost_gradHJBfin]

            X = tf.stack(X_list,axis=1)
            Y = tf.stack(Y_list,axis=1)
            
            return loss, X, Y, Y[0,0,0], his

    def get_solsig0(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        X_list = []
        Y_list = []
        
        t0 = t[:,0,:]
        # W0 = W[:,0,:]
        X0 = tf.tile(Xi,[self.M,1]) # M x D
        Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D
        
        X_list.append(X0)
        Y_list.append(Y0)
        
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) 
            Y1, Z1 = self.net_u(t1,X1)
            
            
            t0 = t1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)
            
        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        
        return X, Y
    
    def train(self, N_Iter, learning_rate):
        
        start_time = time.time()
        print("%s train: iter=%d,ctrl=%d,oc_loss=%d,alpha=%1.1e,betas=[%1.1e,%1.1e,%1.1e] %s" % (4 * "-",N_Iter,self.ctrl_dyn,self.oc_loss,self.alpha,self.betas[0],self.betas[1],self.betas[2],4 * "-"))
        print("%6s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s" % ("iter", "loss", "c_oc", "c_TD", "c_HJBfin", "cHJBfingrad","time","lr"))

        for it in range(N_Iter):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            
            tf_dict = {self.Xi_tf: self.Xi, self.t_tf: t_batch, self.W_tf: W_batch, self.learning_rate: learning_rate}
            
            self.sess.run(self.train_op, tf_dict)
            # with open('results.txt') as myfile:
            #         myfile.write('It: \t, Loss: \t, Y0: \t, OC Cost: \t, Time: \t, Learning Rate: \n', 'w')
            # Print

            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value, Y0_value, learning_rate_value, his = self.sess.run([self.loss, self.Y0_pred, self.learning_rate, self.his], tf_dict)
                print("%6d\t%1.4e\t%1.4e\t%1.4e\t%1.4e\t%1.4e\t%1.4e\t%1.4e" %
                      (it, loss_value, his[0], his[1], his[2], his[3], elapsed, learning_rate_value))
                content  = np.array([it, loss_value, Y0_value, his[0], elapsed, learning_rate_value])
                with open(self.file2store, 'a') as record_append:
                    np.savetxt(record_append, np.asarray([content]), delimiter=',')
                # # with open("results.txt") as myfile:
                #     content  = str(np.array([it, loss_value, Y0_value, oc_cost_val, elapsed, learning_rate_value]))
                #     myfile.write(content + '\n', 'a')
                start_time = time.time()
    
    def predict(self, Xi_star, t_star, W_star):
        
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}
        
        X_star = self.sess.run(self.X_pred, tf_dict)
        Y_star = self.sess.run(self.Y_pred, tf_dict)
        his = self.sess.run(self.his, tf_dict)
        
        return X_star, Y_star, his
    
    
    
    def predict_ctrl(self, Xi_star, t_star, W_star):
        
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}
        
        X_star = self.sess.run(self.X_ctrl, tf_dict)
        Y_star = self.sess.run(self.Y_ctrl, tf_dict)
        oc_cost = self.sess.run(self.OC_ctrl, tf_dict)
        return X_star, Y_star, oc_cost
    
    def predictsig0(self, Xi_star, t_star, W_star):
        
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}
        
        X_star = self.sess.run(self.X_predsig0, tf_dict)
        Y_star = self.sess.run(self.Y_predsig0, tf_dict)
        
        return X_star, Y_star
    
    @abstractmethod
    def phi_tf_zero(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1   
    
    @abstractmethod
    def mu_tf_zero(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return np.zeros([M,D]) # M x D