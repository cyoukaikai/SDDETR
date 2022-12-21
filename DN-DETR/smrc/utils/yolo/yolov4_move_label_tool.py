# import os
import sys
import argparse
from smrc.utils.annotate import prepare_yolov4_label
import smrc.utils


if __name__ == "__main__":
    """
    sample code:
        #(siammask) smrc@smrc:~/object_tracking/ensemble_tracking_exports$ 
        python smrc/tools/bbox_number_tool.py -l smrc/flp/driver-classification/DB_Image_in_result_label_dir_extracted/
        -r first1000videos
    """
    # change to the directory of this script
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='bbox_number')
    parser.add_argument('-l', '--yolo_label_root_dir', default='', type=str, help='Path to yolo_label_root_dir')
    parser.add_argument('-i', '--target_image_dir', default='', type=str, help='Path to target_image_dir')
    # either 'label_dir' or 'image_dir'
    args = parser.parse_args()

    smrc.utils.assert_dir_exist(args.target_image_dir)
    smrc.utils.assert_dir_exist(args.yolo_label_root_dir)
    print(f'yolo_label_root_dir = {args.yolo_label_root_dir}, '
          f'target_image_dir = {args.target_image_dir}')

    # count_empty_txt_file(args_5d_det.yolo_label_root_dir)
    prepare_yolov4_label(
        image_root_dir=args.target_image_dir,
        yolo_label_root_dir=args.yolo_label_root_dir
    )
    # remove_txt_files(args_5d_det.target_image_dir)
    # copy_labels(
    #     yolo_label_root_dir=args_5d_det.yolo_label_root_dir,
    #     target_image_dir=args_5d_det.target_image_dir
    # )
    # padding_empty_files(image_root_dir=args_5d_det.target_image_dir)

