from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os


class Model: 
    "minimalistic TF model for HTR"

    # model constants
    batchSize = 48
    imgSize = (128, 32)
    save_epoch = 1

    def __init__(self, modelDir, mustRestore=False):
        "init model: add CNN, RNN and Output layers and initialize TF"
        self.mustRestore = mustRestore
        self.snapID = 0
        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # self.dropout = tf.placeholder(tf.float32, shape=[])

        # setup CNN, RNN
        self.setupCNN()
        #modified version of RNN used in the original implementation
        self.setupRNN()

        self.setupOut()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
        
        # self.initialize_remaining_variables()
        self.sess, self.saver = self.setupTF(modelDir)

    def initialize_remaining_variables(self):
        # Initialize all uninitialized variables
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)

        self.sess.run(init_new_vars_op)

    def setupCNN(self):
        "create CNN layers and return output of these layers"
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1),trainable=False)
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train,trainable=False)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        self.cnnOut4d = pool


    def setupRNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

        # cells = [tf.contrib.rnn.DropoutWrapper(lstm_cells[i], output_keep_prob = 1 - self.dropout) for i in range(2)]
        
        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
                                    
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
        
        # modifications
        # project output to chars (including blank): BxTx1x2H -> BxTx1x1 -> BxT

        # kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1], stddev=0.1), name='output_collapse_conv')
        # self.rnnOut3d = tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME')
        # self.rnnOut1d = tf.reshape(self.rnnOut3d, [-1,32])

        # BxTx1x2H --> BxTx1x4 ---> Since T = 32, Bx32x1x4 --> Bx128
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 4], stddev=0.1), name='output_collapse_conv')
        self.rnnOut3d = tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME')
        self.rnnOut1d = tf.reshape(self.rnnOut3d, [-1,128])

        return


    def setupOut(self):
        # # Squish everything between 0 to 1
        self.out = tf.sigmoid(self.rnnOut1d)

        # Define label Tensor
        self.targetSplits = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0]))
        # self.targetSplits = tf.placeholder(tf.float32, shape=(None, Model.imgSize[1]))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targetSplits,logits = self.rnnOut1d))

        return

    def setupTF(self, modelDir):
        "initialize TF"
        print('Python: '+sys.version)
        print('Tensorflow: '+tf.__version__)


        # modelDir = '../model/'
        print('modelDir:', modelDir)
        latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?
        print('LatestSnaphshot:',latestSnapshot)

        sess=tf.Session() # TF session

        # Initialize all global variables..
        sess.run(tf.global_variables_initializer())

        saved_variable_names = [var_name+':0' for var_name, var_shape in tf.train.list_variables(latestSnapshot)]
        # print('---Saved Variables---')
        # print(type(saved_variable_names))
        # for variable in saved_variable_names:
        #     print(type(variable),':',variable)

        saved_var_name_set = set(saved_variable_names)

        model_variable_names = [var.name for var in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)]
        # print('---Model Variables---')
        # for variable in model_variable_names:
        #     print(type(variable),':',variable)

        model_var_name_set = set(model_variable_names)

        variables_can_be_restored = list(model_var_name_set.intersection(saved_var_name_set)) 
        variables_can_be_restored = [sess.graph.get_tensor_by_name(var_name) for var_name in variables_can_be_restored]

        # exit(0)
        saver = tf.train.Saver(variables_can_be_restored,max_to_keep=5) # saver saves model to file

        # inspect_list = tf.train.list_variables(latestSnapshot)
        # print('------Printing Variables in Checkpoint------:')

        # for variable in inspect_list:
        #     print(variable)

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            try:
                saver.restore(sess, latestSnapshot)
            except:
                tf.errors.NotFoundError('')

        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())


        # redeclare which variables can be saved...
        saver = tf.train.Saver(max_to_keep=5)

        return (sess,saver)


    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        numBatchElements = len(batch.imgs)
        
        rate = 0.001 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs : batch.imgs, self.targetSplits: batch.targetSplits,
                    self.learningRate : rate, self.is_train: True}
        (_, lossVal) = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal

    def testBatch(self, batch):
        "feed a batch into the NN to test it"
        numBatchElements = len(batch.imgs)
        
        feedDict = {self.inputImgs : batch.imgs, self.targetSplits: batch.targetSplits, self.is_train: False}
        lossVal, predictedSplits = self.sess.run([self.loss,self.out], feedDict)
        # print(predictedSplits)
        # self.batchesTrained += 1
        return lossVal, predictedSplits

    def inferBatch(self, batch):
        "feed a batch into the NN for inference"
        numBatchElements = len(batch.imgs)
        
        feedDict = {self.inputImgs : batch.imgs, self.is_train: False}
        predictedSplits = self.sess.run(self.out, feedDict)
        # print(predictedSplits)
        # self.batchesTrained += 1
        return predictedSplits

    def save(self, path):
        "save model to file"
        self.snapID += self.save_epoch
        self.saver.save(self.sess, path, global_step=self.snapID)
        return

