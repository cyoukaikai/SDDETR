import argparse
from smrc.utils.annotate.visualize_label import VisualizeLabel
from smrc.annotate.future_work.ann_tool_argparse_set import argparse_annotation_setting

if __name__ == "__main__":
    # usage example:
    # (siammask) smrc@smrc:~/object_tracking/ensemble_tracking_exports$
    # python smrc/tools/visualize_annotation_tool.py
    # -i test_data/DB_Image_out
    # -l  test_data/DB_Image_out_labels-thd0.05
    # -c config/class_list_license_plate.txt

    # image_dir, label_dir, class_list_file, user_name, auto_load_directory \
    #     = argparse_annotation_setting()

    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-i', '--image_dir', default='images', type=str, help='Path to image directory')
    parser.add_argument('-l', '--label_dir', default='labels', type=str, help='Path to label directory')
    parser.add_argument('-c', '--class_list_file', default='class_list.txt', type=str,
                        help='File that defines the class labels')
    parser.add_argument('-u', '--user_name', default=None, type=str, help='User name')
    # either 'label_dir' or 'image_dir'
    parser.add_argument('-a', '--auto_load_directory', default=None, type=str,
                        help='The root dir for directories to load')
    args = parser.parse_args()

    AUTO_LOAD_DIRECTORY = None
    if args.auto_load_directory is not None:
        if args.auto_load_directory.find('image') >= 0:
            AUTO_LOAD_DIRECTORY = 'image_dir'
        elif args.auto_load_directory.find('label') >= 0:
            AUTO_LOAD_DIRECTORY = 'label_dir'
        else:
            print(f'AUTO_LOAD_DIRECTORY should be in "image_dir", "label_dir"')

    # print('args_5d_det.image_dir =' , args_5d_det.image_dir)
    visualization_tool = VisualizeLabel(
        user_name=args.user_name,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        class_list_file=args.class_list_file,
        auto_load_directory=AUTO_LOAD_DIRECTORY
        )
    visualization_tool.main_loop()
