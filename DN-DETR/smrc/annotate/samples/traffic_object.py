#!/bin/python
import argparse
import os
# import sys
from smrc.annotate.TrafficObject import AnnotateTrafficObject

if __name__ == '__main__':
    # change to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='FastBBoxAnnotationTool')
    # parser_5d_det.add_argument('-i', '--image_dir', default=None, type=str, help='Path to image directory')
    # parser_5d_det.add_argument('-l', '--label_dir', default=None, type=str, help='Path to label directory')
    parser.add_argument('-u', '--user_name', default=None, type=str, help='User name')
    args = parser.parse_args()

    # modify image dir, label dir, class file name
    IMAGE_DIR = 'test_data/ML_Samples_20181109b'
    LABEL_DIR = 'test_data/ML_Samples_20181109b-Label' #truck-images-labels-v1
    CLASS_LIST_FILE = 'config/class_list.txt'

    # we can specify the active directory by using args_5d_det.input_dir
    # AnnotateTrackletAndBBoxInDirectory
    annotation_tool = AnnotateTrafficObject(
        user_name=args.user_name,
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        class_list_file=CLASS_LIST_FILE
    )
    #annotation_tool.curve_fitting_overlap_suppression_thd = 0.85
    annotation_tool.main_loop()

