#!/bin/python
import argparse
import os
# import sys
from smrc.annotate.SparseBbox import AnnotateSparseBBox

if __name__ == '__main__':
    # change to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='LicensePlateAnnotationTool')
    # parser_5d_det.add_argument('-i', '--input_dir', default=None, type=str, help='Path to image directory')
    # parser_5d_det.add_argument('-l', '--label_dir', default=None, type=str, help='Path to label directory')
    parser.add_argument('-u', '--user_name', default=None, type=str, help='User name')
    parser.add_argument('-blur', '--blur_bbox', default='False', type=str, help='Blur BBox or not')
    args = parser.parse_args()

    IMAGE_DIR = 'test_data/DB_Image_out'
    LABEL_DIR = 'test_data/DB_Image_out_labels'  #Truck_SampleData_licensePlatelabels_fitted
    CLASS_LIST_FILE = 'config/class_list_license_plate.txt'

    if args.blur_bbox.lower() == 'false' or args.blur_bbox == '0':
        blur_bbox = False
    else:
        blur_bbox = True

    # we can specify the active directory by using args_5d_det.input_dir
    annotation_tool = AnnotateSparseBBoxInDirectory(
        user_name=args.user_name,
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        class_list_file=CLASS_LIST_FILE,
        blur_bbox=blur_bbox
    )
    annotation_tool.main_loop()

