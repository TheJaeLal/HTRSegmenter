import cv2
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from DataLoader import DataLoader
from Model import Model
import OutputProcessor
from path_config import path

import cvutils

if not path['model'].exists():
	print('Error: Unable to find model at path:',str(path['model']))
	raise FileNotFoundError

if not path['input_images'].exists():
	print('Error: Unable to find input images at path:',str(path['input_images']))
	raise FileNotFoundError

path['slant_corrected_images'].mkdir(parents=True, exist_ok=True)
path['output_images'].mkdir(parents=True, exist_ok=True)

# Load the neural network model
model = Model(str(path['model']), mustRestore=True)

# Correct slant of input images and store them in a separate directory
for img_file in path['input_images'].glob('*'):
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cvutils.correct_slant(img)
    cv2.imwrite( str(path['slant_corrected_images'] / img_file.name), img)

# Get all the (slant-corrected) inference/test images in a batch (Refer 'Batch' class in 'DataLoader')
infer_samples = list(path['slant_corrected_images'].glob('*'))
infer_batch = DataLoader.getInferBatch(infer_samples, Model.imgSize)

# Get the predictions (targetSplits --> x-cordinates of vertical lines segmenting words into characters)
predictedSplits = model.inferBatch(infer_batch)
infer_batch.targetSplits = predictedSplits

# Apply Non Max Suppression to eliminate overlapping/low-confidence output Splits
# Rescale them to original image size
# (Image is resized for Neural network, so outputs need to be resized to match original image size)
final_splits = OutputProcessor.apply_nms_and_rescale_outputs(infer_batch)

# Iterate through every image in (slant-corrected) test/infer Samples and draw the corresponding splits
for i in range(len(infer_samples)):
    img = cv2.imread( str(path['slant_corrected_images'] / infer_samples[i].name) )
    # print(str(path['slant_corrected_images'] / infer_samples[i].name))
    # img = cvutils.viz_splits_all(cv2.imread( str(path['input_images'] / infer_samples[4].name) ), infer_batch.targetSplits[4], confidence_threshold=0.1)

    img = cvutils.viz_splits_final(img,final_splits[i])
    cv2.imwrite( str(path['output_images'] / infer_samples[i].name), img)