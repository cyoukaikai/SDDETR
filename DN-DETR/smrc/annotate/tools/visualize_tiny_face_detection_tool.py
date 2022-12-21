import argparse
import os

from smrc.utils.annotate.visualize_detection import VisualizeDetection
from smrc.utils.det.txt_det_process import load_txt_detection_to_dict
from smrc.utils.det.detection_process import ScorePosition


class VisualizeTinyFaceDetection(VisualizeDetection):
    def __init__(self, image_dir, label_dir, txt_det_file_dir,
                 class_list_file, auto_load_directory=None,
                 user_name=None
                 ):

        # inherit the variables and functions from AnnotationTool
        VisualizeDetection.__init__(
            self, image_dir=image_dir,
            label_dir=label_dir,
            class_list_file=class_list_file,
            json_file_dir=None,
            auto_load_directory=auto_load_directory,
            user_name=user_name
        )
        # self.active_directory_detection_dict = {}
        # self.json_file_dir = json_file_dir
        # self.score_thd = 10  # 0.10
        # self.TRACKBAR_SCORE = 'Confidence Level'
        # self.show_score_on = True
        self.txt_det_file_dir = txt_det_file_dir

    def load_active_directory_additional_operation(self):
        self.active_directory_detection_dict = load_txt_detection_to_dict(
            image_sequence_dir=os.path.join(self.IMAGE_DIR, self.active_directory),
            detection_dir_name=os.path.join(self.txt_det_file_dir, self.active_directory),
            score_position=ScorePosition.Second,
            detection_dict=None,
            score_thd=0.0,
            short_image_path=True
        )


if __name__ == "__main__":
    # change to the directory of this script
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-i', '--image_dir', default=None, type=str, help='Path to image directory')
    parser.add_argument('-l', '--label_dir', default='deprecated', type=str, help='Path to label directory')
    parser.add_argument('-c', '--class_list_file', default=None, type=str,
                        help='File that defines the class labels')
    parser.add_argument('-u', '--user_name', default=None, type=str, help='User name')
    parser.add_argument('-j', '--json_file_dir', default=None, type=str, help='Json file dir')
    # either 'label_dir' or 'image_dir'
    parser.add_argument('-t', '--txt_det_file_dir', default=None, type=str, help='Json file dir')
    # either 'label_dir' or 'image_dir'
    parser.add_argument('-a', '--auto_load_directory', default=None, type=str,
                        help='Where to load the directory')
    # parser_5d_det.add_argument('-d', '--dir_list_root_dir', default=None, type=str,
    #                     help='Where to load the directory')
    args = parser.parse_args()

    # print('args_5d_det.image_dir =' , args_5d_det.image_dir)
    visualization_tool = VisualizeTinyFaceDetection(
        user_name=args.user_name,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        class_list_file=args.class_list_file,
        auto_load_directory=args.auto_load_directory,
        txt_det_file_dir=args.txt_det_file_dir
        )
    visualization_tool.main_loop()
