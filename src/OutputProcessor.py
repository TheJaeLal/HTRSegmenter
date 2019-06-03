import numpy as np

def _nms(predictedSplits,confidenceSplits, merge_distance_threshold = 2, individual_split_threshold = 0.15):
    total_splits = []
    if len(predictedSplits) > 1:
        merge_splits = [predictedSplits[0]]
        for i in range(1, len(predictedSplits)):
            if predictedSplits[i] - predictedSplits[i-1] <= merge_distance_threshold:
                merge_splits.append(predictedSplits[i])
            else:
                # if this is a single split, then apply individual split threshold 
                if len(merge_splits)==1:
                    if confidenceSplits[merge_splits[0]] >= individual_split_threshold:
                        total_splits.append(merge_splits[0])
                else:                                    
                    # merge the similar splits, by picking the one with the maximum confidence amongs all of them.
                    total_splits.append(merge_splits[np.argmax([confidenceSplits[mergeSplit] for mergeSplit in merge_splits])])
    #                 print('predictedSplits[i]', predictedSplits[i])
    #                 print('total_splits', total_splits)

                # Reset merge_splits
                merge_splits = [predictedSplits[i]]

        # print('merge_slits:', merge_splits)

        if len(merge_splits) == 1:
            if confidenceSplits[merge_splits[0]] >= individual_split_threshold:
                total_splits.append(merge_splits[0])
        elif len(merge_splits) > 1:
            total_splits.append(merge_splits[np.argmax([confidenceSplits[mergeSplit] for mergeSplit in merge_splits])])

    return total_splits

def apply_nms_and_rescale_outputs(infer_batch):

    final_splits = []
    for i in range(len(infer_batch.targetSplits)):
        confidenceSplits = np.array(infer_batch.targetSplits[i])
        # Get the index of splits having confidence greater than threshold!
        predictedSplits = np.array(np.where(confidenceSplits >= 0.1)[0])

        # confidence = {predictedSplit:confidenceSplits[predictedSplit] for predictedSplit in predictedSplits}
        # print(loader.inferSamples[i])
        # print(confidence)
        # print('-----------')

        total_splits = _nms(predictedSplits, confidenceSplits)

        scaleX,_ = infer_batch.scaleFactors[i]

        scaledSplits = [int(scaleX * split) for split in total_splits]

        final_splits.append(scaledSplits)
        
    return final_splits