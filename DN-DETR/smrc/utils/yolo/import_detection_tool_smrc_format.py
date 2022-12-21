#!/bin/python
import argparse

# sys.path.append("..")
from smrc.utils.annotate.det2label import ImportDetection
# print(os.getcwd())

if __name__ == "__main__":
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-j', '--json_file_dir', default=None, type=str, help='Json file dir')
    parser.add_argument('-l', '--label_dir', default=None, type=str, help='Path to label directory')
    parser.add_argument('-s', '--score_thd', default=None, type=float,
                        help='Confidence level threshold')
    # only for face and license plate
    parser.add_argument('-n', '--nms_thd', default=None, type=float,
                        help='Non max suppression threshold')
    parser.add_argument('--source_class_list_file', default=None, type=str,
                        help='File that defines the source class labels')
    parser.add_argument('--target_class_list_file', default=None, type=str,
                        help='File that defines the target class labels')

    args = parser.parse_args()

    assert args.json_file_dir is not None
    # print(os.path.abspath(args_5d_det.json_file_dir))
    my_tool = ImportDetection(
        json_file_dir=args.json_file_dir,
        label_dir=args.label_dir,
        score_thd=args.score_thd,
        nms_thd=args.nms_thd,
        source_class_list_file=args.source_class_list_file,
        target_class_list_file=args.target_class_list_file
    )


##########################
# (siammask) smrc@smrc:~/tuat/smrc$ python tools/import_detection_tool_smrc_format.py --json_file_dir test_data/detection_json_files/det_smrc
# /home/smrc/tuat/smrc
# /home/smrc/tuat/smrc/test_data/detection_json_files/det_smrc
# ====================================================================
# self.json_file_dir = /home/smrc/tuat/smrc/test_data/detection_json_files/det_smrc ...
# self.label_dir = /home/smrc/tuat/smrc/test_data/detection_json_files/det_smrc_labelsNone_nmsNone ...
# score_thd = None, non_max_suppression_thd = None
# ====================================================================
# test_data/detection_json_files/det_smrc
# 2 json files loaded with the format smrc_*.json.
# 1/2, Handling test_data/detection_json_files/det_smrc/smrc_2.json ...
#     0 object_detection ignored due to score_thd None, remaining 2008 detections ...
#     0 object_detection ignored due nms_thd None, remaining 2008 detections ...
# 2/2, Handling test_data/detection_json_files/det_smrc/smrc_daySequence1.json ...
#     0 object_detection ignored due to score_thd None, remaining 3493 detections ...
#     0 object_detection ignored due nms_thd None, remaining 3493 detections ...
##########################