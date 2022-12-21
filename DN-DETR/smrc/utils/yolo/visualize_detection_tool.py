import argparse
from smrc.utils.annotate.visualize_detection import VisualizeDetection


if __name__ == "__main__":
    # change to the directory of this script
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-i', '--image_dir', default='images', type=str, help='Path to image directory')
    parser.add_argument('-l', '--label_dir', default='labels', type=str, help='Path to label directory')
    parser.add_argument('-c', '--class_list_file', default='class_list_traffic.txt', type=str,
                        help='File that defines the class labels')
    parser.add_argument('-u', '--user_name', default=None, type=str, help='User name')
    parser.add_argument('-j', '--json_file_dir', default='detection_json_files', type=str, help='Json file dir')
    # either 'label_dir' or 'image_dir'
    parser.add_argument('-a', '--auto_load_directory', default=None, type=str,
                        help='Where to load the directory')
    args = parser.parse_args()

    # print('args_5d_det.image_dir =' , args_5d_det.image_dir)
    visualization_tool = VisualizeDetection(
        user_name=args.user_name,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        class_list_file=args.class_list_file,
        auto_load_directory=args.auto_load_directory,
        json_file_dir=args.json_file_dir
        )
    visualization_tool.main_loop()