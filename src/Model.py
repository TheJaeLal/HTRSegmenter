from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model: 
    "minimalistic TF model for HTR"

    # model constants
    batchSize = 48
    imgSize = (128, 32)
    maxTextLen = 32
    save_epoch = 2

    def __init__(self, modelDir, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        # self.dump = dump
        # self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0
        self.restore_variables = []
        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()

        #modified version of RNN
        concat = self.setupRNN()

        # modifications
        # project output to chars (including blank): BxTx1x2H -> BxTx1x1 -> BxT
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1], stddev=0.1), name='output_collapse_conv')
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2,3])
        
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

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
                                    
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
           
        # # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        # kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, 79 + 1], stddev=0.1))
        # self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
        
        return concat

    def setupCTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words) 
            chars = str().join(self.charList)
            wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
            corpus = open('../data/corpus.txt').read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

    def setupOut(self):
        # # Squish everything between 0 to 1
        self.out = tf.sigmoid(self.rnnOut3d)

        # Define label Tensor
        self.targetSplits = tf.placeholder(tf.float32, shape=(None, Model.imgSize[1]))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targetSplits,logits = self.rnnOut3d))


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


    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0] # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)


    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"
        
        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank=len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label==blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor 
            decoded=ctcOutput[0][0] 

            # go over all indices and save mapping: batch -> values
            idxDict = { b : [] for b in range(batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0] # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


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
        print(predictedSplits)
        # self.batchesTrained += 1
        return lossVal, predictedSplits

    def inferBatch(self, batch):
        "feed a batch into the NN for inference"
        numBatchElements = len(batch.imgs)
        
        feedDict = {self.inputImgs : batch.imgs, self.is_train: False}
        predictedSplits = self.sess.run(self.out, feedDict)
        print(predictedSplits)
        # self.batchesTrained += 1
        return predictedSplits

    def dumpNNOutput(self, rnnOutput):
        "dump the output of the NN to CSV file(s)"
        dumpDir = '../dump/'
        if not os.path.isdir(dumpDir):
            os.mkdir(dumpDir)

        # iterate over all batch elements and create a CSV file for each one
        maxT, maxB, maxC = rnnOutput.shape
        for b in range(maxB):
            csv = ''
            for t in range(maxT):
                for c in range(maxC):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            fn = dumpDir + 'rnnOutput_'+str(b)+'.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)


    # def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
    #     "feed a batch into the NN to recognize the texts"
        
    #     # decode, optionally save RNN output
    #     numBatchElements = len(batch.imgs)
    #     evalRnnOutput = self.dump or calcProbability
    #     evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
    #     feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
    #     evalRes = self.sess.run(evalList, feedDict)
    #     decoded = evalRes[0]
    #     texts = self.decoderOutputToText(decoded, numBatchElements)
        
    #     # feed RNN output and recognized text into CTC loss to compute labeling probability
    #     probs = None
    #     if calcProbability:
    #         sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
    #         ctcInput = evalRes[1]
    #         evalList = self.lossPerElement
    #         feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
    #         lossVals = self.sess.run(evalList, feedDict)
    #         probs = np.exp(-lossVals)

    #     # dump the output of the NN to CSV file(s)
    #     if self.dump:
    #         self.dumpNNOutput(evalRes[1])

    #     return (texts, probs)

    def save(self, path):
        "save model to file"
        self.snapID += self.save_epoch
        self.saver.save(self.sess, path, global_step=self.snapID)


