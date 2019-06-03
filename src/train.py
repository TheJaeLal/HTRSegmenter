from __future__ import division
from __future__ import print_function

import cv2

from DataLoader import DataLoader, Batch
from Model import Model
import OutputProcessor

loader = DataLoader('../data', Model.batchSize, Model.imgSize)
model = Model('../model/', mustRestore=True)
epoch = 0 # number of training epochs since start
max_epochs = 10

while True:
    epoch += 1
    print('Epoch:', epoch)

    # train
    print('Train NN')
    loader.trainSet()
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        batch = loader.getNext()
        test_batch = loader.getTestBatch()
        
        train_loss = model.trainBatch(batch)
        test_loss,predictedSplits = model.testBatch(test_batch)
        
        test_batch.targetSplits = predictedSplits
        loader.vizOutputSplits(test_batch)

        print('Batch:', iterInfo[0],'/', iterInfo[1], 'train_loss:', train_loss, 'test_loss:', test_loss)

    if epoch%model.save_epoch==0:
        print('Saving model!')
        model.save('../model/new_models/snapshot')

    if epoch >= max_epochs:
        break