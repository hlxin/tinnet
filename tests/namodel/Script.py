#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf


class NAModel:

    def __init__(self):
        # Initialize the class
        self.num_image = 245
        self.num_na_input = 7
        self.num_datapoints = 3501
        
        self.ergy = np.linspace(-20, 15, self.num_datapoints)
        self.alpha_ergy = np.linspace(-20, 15, self.num_datapoints)
        
        self.alpha_ergy[self.alpha_ergy>0] = 0
        self.alpha_ergy[self.alpha_ergy<0] = 1
        
        # Load data for NA model
        self.feature, self.dos_sp, self.dos_d, self.target \
            = self.read_files(self.num_image)
        
    def read_files(self, num_image):
        # Load data for NA model
        feature = np.array([np.loadtxt('../Database/Features/Features_' \
                  + str(i+1).zfill(3) + '.txt') for i in range(num_image)])
        
        dos_sp = np.load('../Database/GenInput/dos_sp.npy')
        dos_d = np.load('../Database/GenInput/dos_d.npy')
        target = np.load('../Database/GenInput/dos_ads.npy')
        
        np.savetxt('Ground_Truth.txt', target.T)
        return feature, dos_sp, dos_d, target 
        
    def namodel(self, namodel_in, ergy, alpha_ergy, dos_sp, dos_d, num_image,
                num_datapoints):
        
        effadse = tf.tanh(namodel_in[:,0]) * 20.0 + 0.0
        vak2_sp = tf.sigmoid(namodel_in[:,1]) * 20.0 + 0.0
        vak2_d = tf.sigmoid(namodel_in[:,2]) * 20.0 + 0.0
        alpha_sp = tf.sigmoid(namodel_in[:,3]) * 1.0 + 0.0
        alpha_d = tf.sigmoid(namodel_in[:,4]) * 1.0 + 0.0
        gamma_sp = tf.sigmoid(namodel_in[:,5]) * 1.0 + 0.0
        gamma_d = tf.sigmoid(namodel_in[:,6]) * 1.0 + 0.0

        namodel_in_tf = tf.transpose(tf.convert_to_tensor([effadse, vak2_sp,
                                                           vak2_d, alpha_sp,
                                                           alpha_d, gamma_sp,
                                                           gamma_d]))
        
        de = tf.abs(effadse[:,None]-ergy[None,:])
    
        wdos = np.pi * (vak2_sp[:,None] * dos_sp[None,:]
                        * tf.exp(alpha_sp[:,None] * alpha_ergy[None,:]
                        * ergy[None,:]) * tf.exp(-gamma_sp[:,None] * de)
                        + vak2_d [:,None] * dos_d [None,:]
                        * tf.exp(alpha_d [:,None] * alpha_ergy[None,:]
                        * ergy[None,:]) * tf.exp(-gamma_d [:,None] * de))
        
        htwdos = []
        
        for i in range(num_image):
            
            af = tf.signal.fft(tf.cast(wdos[i], tf.complex64))
            h = np.zeros(num_datapoints)
            
            if num_datapoints % 2 == 0:
                h[0] = h[num_datapoints // 2] = 1
                h[1:num_datapoints // 2] = 2
            else:
                h[0] = 1
                h[1:(num_datapoints+1) // 2] = 2
                
            h = tf.convert_to_tensor(h, tf.complex64)
            htwdos += [tf.math.imag(tf.signal.ifft(af*h))]
            
        htwdos = tf.convert_to_tensor(htwdos)

        dos_ads_na = wdos / ((ergy[None,:]-effadse[:,None]-htwdos)**2
                             + wdos**2) / np.pi

        ans_namodel = dos_ads_na / tf.reduce_sum(dos_ads_na, axis=1)[:,None]
        
        return ans_namodel*100 , namodel_in_tf
    
    
class ANN(NAModel):
    
    def __init__(self, hidden_layers, fraction_train, fraction_test):
        # Initialize the class
        
        # Initialize Physical Model
        NAModel.__init__(self)

        # Initialize Indexes
        self.index_train, self.index_test, self.index_val, \
        self.num_train, self.num_val, self.num_test \
            = self.initialize_indexes(self.num_image,
                                      fraction_train,
                                      fraction_test)
            
        # Initialize ANNs
        self.layers = [self.feature.shape[1]] + hidden_layers \
                      + [self.num_na_input]
        
        self.weights, self.biases = self.initialize_ann(self.layers)

        # tf Placeholders
        self.feature_tf = tf.placeholder(tf.float32,
                                         shape=[self.feature.shape[0],
                                                self.feature.shape[1]])
    
        self.target_tf = tf.placeholder(tf.float32,
                                        shape=[self.target.shape[0],
                                               self.target.shape[1]])

        # tf Graphs
        self.nn_out = self.neural_net(self.feature_tf,
                                      self.weights,
                                      self.biases)

        self.dos_pred, self.na_parm = self.namodel(self.nn_out,
                                                   self.ergy, 
                                                   self.alpha_ergy, 
                                                   self.dos_sp, 
                                                   self.dos_d, 
                                                   self.num_image, 
                                                   self.num_datapoints)

        # Loss
        self.loss_train = tf.sqrt(tf.reduce_mean(tf.square(self.target_tf
                          - self.dos_pred) * self.index_train[:,None]
                          * self.num_image / self.num_train))
    
        self.loss_test = tf.sqrt(tf.reduce_mean(tf.square(self.target_tf
                         - self.dos_pred) * self.index_test[:,None]
                         * self.num_image / self.num_test))
    
        self.loss_val = tf.sqrt(tf.reduce_mean(tf.square(self.target_tf
                        - self.dos_pred) * self.index_val[:,None]
                        * self.num_image / self.num_val))

        for l2_index in self.weights:
            self.loss_train += tf.nn.l2_loss(l2_index) * 0.0010
            self.loss_test += tf.nn.l2_loss(l2_index) * 0.0010
            self.loss_val += tf.nn.l2_loss(l2_index) * 0.0010

        # Optimizers
        self.optimizer_adam = tf.train.AdamOptimizer()
        self.train_op_adam = self.optimizer_adam.minimize(self.loss_train)
        
        # tf session
        self.sess = tf.Session(config = tf.ConfigProto(
                               allow_soft_placement=True,
                               log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_indexes(self, num_image, fraction_train, fraction_test):
        # Initialize indexes
        np.random.seed(1234)
        
        index_total = np.arange(num_image)
        
        index_train = np.zeros(num_image)
        index_test = np.zeros(num_image)
        index_val = np.zeros(num_image)
        
        np.random.shuffle(index_total)
        
        indexes = np.split(index_total, [int(fraction_train*num_image),
                           int((fraction_train+fraction_test)*num_image)])
        
        index_train[indexes[0]] = 1
        index_test[indexes[1]] = 1
        index_val[indexes[2]] = 1
        
        num_train = np.sum(index_train)
        num_val = np.sum(index_val)
        num_test = np.sum(index_test)
        
        np.savetxt('Training_index.txt', index_train)
        np.savetxt('Validation_index.txt', index_val)
        np.savetxt('Test_index.txt', index_test)

        return index_train, index_test, index_val, num_train, num_val, num_test

    def initialize_ann(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        
        for i in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[i], layers[i+1]])
            b = tf.Variable(tf.zeros([1,layers[i+1]], dtype=tf.float32),
                            dtype=tf.float32)
            
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim+out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                      stddev=xavier_stddev),
                                                      dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        for i in range(0, num_layers-2):
            W = weights[i]
            b = biases [i]
            X = tf.tanh(tf.add(tf.matmul(X, W), b))
            
        W = weights[-1]
        b = biases [-1]
        Y = tf.add(tf.matmul(X, W), b)
        return Y
    
    def train(self, niter):
        
        tf_dict = {self.feature_tf: self.feature, self.target_tf: self.target}
        
        start_time = time.time()
        
        for it in range(niter):
            self.sess.run(self.train_op_adam, tf_dict)
            
            # Print
            if it % 1 == 0:
                
                elapsed = time.time() - start_time
                
                loss_train_value = self.sess.run(self.loss_train, tf_dict)
                loss_val_value = self.sess.run(self.loss_val, tf_dict)
                loss_test_value = self.sess.run(self.loss_test, tf_dict)

                print('It: %d, Loss_Training: %.10e, Loss_Validation: %.10e, \
                      Loss_Test: %.10e, Time: %.5f' %(it, loss_train_value, 
                      loss_val_value, loss_test_value, elapsed))
                
                start_time = time.time()
                
        for i in range(len(self.layers)-1):
            W = self.sess.run(self.weights[i])
            B = self.sess.run(self.biases[i])
            np.savetxt('Weight_' + str(i) + '.txt', W)
            np.savetxt('Bias_' + str(i) + '.txt', B)

        np.savetxt('Model.txt', self.sess.run(self.dos_pred, tf_dict).T)
        np.savetxt('NA_Parameters.txt', self.sess.run(self.na_parm, tf_dict))


if __name__ == "__main__":
    # Main code
    fraction_train = 0.7
    fraction_test = 0.1
    
    epochs = 10

    hidden_layers = [15, 15]

    # Training - Tensorflow
    model = ANN(hidden_layers, fraction_train, fraction_test)
    
    start_time = time.time()
    
    model.train(epochs)
    
    elapsed = time.time() - start_time
    
    print('Training time: %.5f' % (elapsed))