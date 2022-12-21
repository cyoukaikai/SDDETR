# import smrc.line
import argparse
from smrc.utils.annotate.bbox_statistic import estimate_and_report_statistics


if __name__ == "__main__":
    # change to the directory of this script
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='estimate_and_report_statistics')
    parser.add_argument('-l', '--label_dir', default='', type=str, help='Path to label directory')
    parser.add_argument('-r', '--class_list_name', default=None, type=str, help='Json file dir')
    # either 'label_dir' or 'image_dir'
    args = parser.parse_args()

    estimate_and_report_statistics(
        root_dir=args.label_dir,
        class_list_name=args.class_list_name
    )
