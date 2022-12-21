import argparse
import smrc.utils
from smrc.utils import json_yolov4_to_yolov3

if __name__ == "__main__":
    # change to the directory of this script
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-v4', '--v4_json_root_dir', default=None, type=str, help='Path to v4_json_root_dir')
    parser.add_argument('-v3', '--v3_json_root_dir', default=None, type=str, help='Path to v3_json_root_dir')
    args = parser.parse_args()

    smrc.utils.generate_dir_if_not_exist(args.v3_json_root_dir)
    json_yolov4_to_yolov3(
        json_yolov4_dir=args.v4_json_root_dir,
        json_yolov3_dir=args.v3_json_root_dir
    )
