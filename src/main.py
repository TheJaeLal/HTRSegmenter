from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2

from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    fnTrain = '../data/'

def train(model, loader):
    "train NN"
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


def infer(model, loader):
    infer_batch = loader.getInferBatch()
    predictedSplits = model.inferBatch(infer_batch)
    infer_batch.targetSplits = predictedSplits
    loader.vizOutputSplits(infer_batch)    

def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')

    args = parser.parse_args()
    loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize)
    
    if args.train:
        print("----Training----")
        train_model = Model('../model/', mustRestore=True)
        train(train_model, loader)

    else:
        print("----Inference----")
        infer_model = Model('../model/new_models/', mustRestore=True)
        infer(infer_model, loader)

if __name__ == '__main__':
    main()

