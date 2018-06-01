#!/usr/bin/env python3

import argparse
import os
from scoring_utils import get_score
from glob import glob
import numpy as np

parser = argparse.ArgumentParser(
    description='scoring script for text localization')
parser.add_argument('--reference', type=str, required=True,
                    help='reference directory of test data, contains np array')
parser.add_argument('--hypothesis', type=str, required=True,
                    help='hypothesis directory of test data, contains np array')
parser.add_argument('--result', type=str, required=True,
                    help='the file to store final statistical results')
args = parser.parse_args()


def main():
    mean_ar = 0
    stat_dict = {}
    img_count = 0
    for img_ref_path, img_hyp_path in zip(glob(args.reference+"/*.mask.npy"),
                                        glob(args.hypothesis+"/*.mask.npy")):
        img_count += 1
        ref_arr = np.load(img_ref_path)
        hyp_arr = np.load(img_hyp_path)
        image_id = os.path.basename(img_ref_path).split('.mask.npy')[0]
        thresholds = np.arange(0.50, 1.0, 0.05)
        average_recall = 0
        for threshold in thresholds:
            recall = get_score(ref_arr, hyp_arr, threshold)['recall']
            average_recall += recall
            if image_id in stat_dict.keys():
                stat_dict[image_id][threshold] = recall
            else:
                stat_dict[image_id] = dict()
                stat_dict[image_id][threshold] = recall
        average_recall /= len(thresholds)
        stat_dict[image_id]['Average Recall'] = average_recall
        mean_ar += average_recall
    mean_ar /= img_count
    print("Mean average recall: {}".format(mean_ar))
    with open(args.result, 'w') as fh:
        fh.write('Mean Average Recall: {}\n'.format(mean_ar))
        fh.write('ImageID\tThreshold\tRecall\n')
        for image_id in stat_dict.keys():
            for threshold in stat_dict[image_id].keys():
                recall = stat_dict[image_id][threshold]
                fh.write('{}\t{}\t{}\n'.format(image_id, threshold, recall))

if __name__ == '__main__':
      main()
