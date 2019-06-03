from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess


class Sample:
    "sample from the dataset (X,y)"
    def __init__(self, targetSplit, filePath):
        self.targetSplit = targetSplit
        self.filePath = filePath


class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, targetSplits, imgs, scaleFactors = (1,1)):
        self.imgs = np.stack(imgs, axis=0)
        self.targetSplits = targetSplits
        self.scaleFactors = scaleFactors

class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

    def __init__(self, filePath, batchSize, imgSize):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert filePath[-1]=='/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.inferSamples = []
        self.viz_dir = '../viz_dir/'
        self.inferDir = '../data/test_words/'
        # Writing your custom loader..
        with open(filePath+'words.txt') as f:
            labels = f.read().split('\n')

        # chars = set()
        # bad_samples = []
        # bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']

        for line in labels:
        # for line in f:
            # ignore comment line
            if not line or line[0]=='#':
                continue
            
            lineSplit = line.strip().split(',')
            
            fileName = filePath + 'words/' + lineSplit[0]
            
            # to be extended to the size of the image...
            splitText = [int(x) for x in lineSplit[1:]]

            # check if image is not empty
            # if not os.path.getsize(fileName):
            #   bad_samples.append(lineSplit[0] + '.png')
            #   continue

            # put sample into list
            self.samples.append(Sample(splitText, fileName))

        # some images in the IAM dataset are known to be damaged, don't show warning for them
        # if set(bad_samples) != set(bad_samples_reference):
        #   print("Warning, damaged images found:", bad_samples)
        #   print("Damaged images expected:", bad_samples_reference)

        self.trainSamples = self.samples
        self.testSamples = []

        with open(filePath+'test_words.txt') as f:
            test_labels = f.read().split('\n')

        for line in test_labels:
            lineSplit = line.strip().split(',')
            
            fileName = filePath + 'test_words/' + lineSplit[0]
            
            # to be extended to the size of the image...
            splitText = [int(x) for x in lineSplit[1:]]

            self.testSamples.append(Sample(splitText, fileName)) 

        
        self.testSize = len(test_labels)
        # number of randomly chosen samples per epoch for training 
        # This number may not be helpful!
        self.numTrainSamplesPerEpoch = len(labels)-1
        
        # start with train set
        self.trainSet()


    def trainSet(self):
        "switch to randomly chosen subset of training set"
        # self.dataAugmentation = True
        self.dataAugmentation = False
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    
    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def testSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.testSamples


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)
        
        
    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        
        targetSplits = []
        
        for i in batchRange:
            output_array = [0]*self.imgSize[0]
            # output_array = [0]*self.imgSize[1]
            for split in self.samples[i].targetSplit:
                #output width size is 128

                split = int(split)
                # ratio = self.imgSize[0] / self.imgSize[1]
                # split = int(split / ratio)
                try:
                    output_array[split] = 1
                except IndexError as e:
                    print(self.samples[i].filePath)
                    exit(0)

            targetSplits.append(output_array)

        imgs = []
        scaleFactors = []
        for i in batchRange:
            img = cv2.imread(self.samples[i].filePath+'.png', cv2.IMREAD_GRAYSCALE)
            img, scaleX, scaleY = preprocess(img, self.imgSize, False)
            imgs.append(img)
            scaleFactors.append((scaleX, scaleY))
        
        self.currIdx += self.batchSize
        return Batch(targetSplits, imgs, scaleFactors)

    def getInferBatch(self):
        self.inferSamples = os.listdir(self.inferDir)

        imgs = []
        scaleFactors = []
        for imgName in self.inferSamples:
            img = cv2.imread(self.inferDir + imgName, cv2.IMREAD_GRAYSCALE)
            img, scaleX, scaleY = preprocess(img, self.imgSize, False) 
            imgs.append(img)
            scaleFactors.append((scaleX, scaleY))

        # Create a batch with blank labels/targetSplits
        # for i in range(len(imgs)):
        #     c2v.imwrite('infer_images/'+str(i)+'.png', imgs[i].T.astype(np.uint8))

        return Batch(None, imgs, scaleFactors)

    def getTestBatch(self):
        # gray scale images of shape (Model.size[1], Model.size[0])
        targetSplits = []
        
        for i in range(self.testSize):
            output_array = [0]*self.imgSize[0]
            # output_array = [0]*self.imgSize[1]
            for split in self.testSamples[i].targetSplit:
                #output width size is 32 and not 128
                split = int(split)
                # ratio = self.imgSize[0] / self.imgSize[1]
                # split = int(split / ratio)
                output_array[split] = 1
            targetSplits.append(output_array)

        imgs = []
        scaleFactors = []
        for i in range(self.testSize):
            img = cv2.imread(self.testSamples[i].filePath+'.png', cv2.IMREAD_GRAYSCALE)
            img, scaleX, scaleY = preprocess(img, self.imgSize, False) 
            imgs.append(img)
            scaleFactors.append((scaleX, scaleY))

        # for i in range(len(imgs)):
        #     cv2.imwrite('test_images/'+str(i)+'.png', imgs[i].T.astype(np.uint8))


        return Batch(targetSplits, imgs, scaleFactors)

    def vizOutputSplits(self, test_batch):
        numBatchElements = len(test_batch.imgs)
        confidence_threshold = 0.1

        for i in range(numBatchElements):
            img = test_batch.imgs[i].T
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            predictedSplits = np.array(test_batch.targetSplits[i])

            # print("type(predictedSplits)\n", type(predictedSplits))
            # Get the index of splits having confidence greater than threshold!
            predictedSplits = np.where(predictedSplits >= confidence_threshold)[0]

            for split in predictedSplits:
                split = int(split)
                # ratio = self.imgSize[0] / self.imgSize[1]
                # split = int(split * ratio)
                cv2.line(img, (split,0), (split, img.shape[0]), (0,0,255), 1)

            cv2.imwrite(self.viz_dir+str(i)+'.png', img)