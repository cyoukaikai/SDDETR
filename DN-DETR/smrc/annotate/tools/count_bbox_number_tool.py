# import os
import argparse
from smrc.utils import count_bbox_number


if __name__ == "__main__":
    """
    sample code:
        python smrc/tools/bbox_number_tool.py -l smrc/flp/driver-classification/DB_Image_in_result_label_dir_extracted/
        -r first1000videos
    """
    # change to the directory of this script
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='bbox_number')
    parser.add_argument('-l', '--label_dir', default='', type=str, help='Path to label directory')
    parser.add_argument('-r', '--result_file_name', default=None, type=str, help='Json file dir')
    # either 'label_dir' or 'image_dir'
    args = parser.parse_args()

    count_bbox_number(
        label_dir=args.label_dir,
        result_file_name=args.result_file_name
    )
