#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf

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
