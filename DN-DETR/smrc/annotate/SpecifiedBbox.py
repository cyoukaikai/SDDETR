#######################################################
# annotate bbox with given image list or ann path list
#######################################################
import os
import cv2

import smrc.utils

from .Bbox import AnnotateBBox
from .SparseBbox import AnnotateSparseBBox


class AnnotateSpecifiedBBox(AnnotateBBox):
    def __init__(
            self, image_dir, label_dir, class_list_file,
            specified_image_list=None,
            specified_ann_path_list=None
            ):
        AnnotateBBox.__init__(
            self, image_dir=image_dir, label_dir=label_dir,
            class_list_file=class_list_file
            )
       
        # print(f'class_list_file = {class_list_file}')
        self.IMAGE_DIR = image_dir
        self.LABEL_DIR = label_dir
        self.class_list_file = class_list_file
        # print(f'self.class_list_file = {self.class_list_file}')

        # self.active_directory = None
        self.specified_image_list = specified_image_list
        self.specified_ann_path_list = specified_ann_path_list

        self.IMAGE_WINDOW_NAME = 'VisualizeSpecifiedData'
        self.TRACKBAR_IMG = 'Image'

        print('self.IMAGE_DIR:', self.IMAGE_DIR)
        print('self.LABEL_DIR:', self.LABEL_DIR)
        print('self.class_list_file:', self.class_list_file)

        if self.class_list_file is not None and os.path.isfile(self.class_list_file):
            self.init_class_list_and_class_color()
        # #     # print(self.CLASS_LIST)
        # #     # sys.exit(0)

    def Event_FinishActiveDirectoryAnnotation(self):
        print(f'Do nothing for this operation. ')
        pass

    def show_annotation_setting(self):
        print('===================================== Information for annotation')
        print('self.IMAGE_DIR:', self.IMAGE_DIR)
        print('self.LABEL_DIR:', self.LABEL_DIR)
        print('self.class_list_file:', self.class_list_file)
        print('self.CLASS_LIST:', self.CLASS_LIST)
        print('=====================================')

    def load_specified_annotation_file_list(self):
        assert self.specified_ann_path_list is not None and \
               len(self.specified_ann_path_list) > 0
        image_path_list = [
            smrc.utils.get_image_or_annotation_path(
                f, self.LABEL_DIR, self.IMAGE_DIR, '.jpg'
            ) for f in self.specified_ann_path_list
        ]
        self.load_specified_image_list(specified_image_list=image_path_list)

    def load_specified_image_list(self, specified_image_list=None):
        if specified_image_list is None:
            specified_image_list = self.specified_image_list

        assert specified_image_list is not None and len(specified_image_list) > 0

        self.IMAGE_PATH_LIST = specified_image_list
        # self.IMAGE_PATH_LIST = []
        # for f_path in specified_image_list:
        #     # check if it is an image
        #     assert smrc.utils.is_image(f_path)
        #     self.IMAGE_PATH_LIST.append(f_path)
        self.LAST_IMAGE_INDEX = len(self.IMAGE_PATH_LIST) - 1
        print('{} images are loaded to self.IMAGE_PATH_LIST'.format(len(self.IMAGE_PATH_LIST)))
     
    def annotate_specified_data(self):
        # self.annotation_done_flag = False

        # this function must be after self.load_image_sequence(), otherwise, the trackBar
        # for image list can not be initialized (as number of image not known)
        self.init_image_window_and_mouse_listener()

        # load the first image in the IMAGE_PATH_LIST, 
        # initilizing self.active_image_index and related information for the active image
        self.set_image_index(0)

        while True:
            # load the class index and class color for plot
            # color = self.CLASS_BGR_COLORS[self.active_class_index].tolist()

            # copy the current image
            # tmp_img = self.active_image.copy()
            tmp_img = self.display_annotation_in_active_image()
            # get annotation paths
            image_path = self.IMAGE_PATH_LIST[self.active_image_index]  # image_path is not a global variable
            self.active_image_annotation_path = self.get_annotation_path(image_path)

            # print('image_path=', image_path)
            # print('annotation_path =', self.active_image_annotation_path)

            # display annotated bboxes
            self.draw_bboxes_from_file(tmp_img, self.active_image_annotation_path)  # , image_width, image_height
            self.set_active_bbox_idx_if_NONE()
            # set the active directory based on mouse cursor
            # self.set_active_directory_based_on_mouse_position()
            self.draw_active_bbox(tmp_img)

            pressed_key = self.read_pressed_key()
            if pressed_key & 0xFF == 27:  # Esc key is pressed
                # cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                break
            else:
                self.keyboard_listener(pressed_key, tmp_img)

            self.game_controller_listener(tmp_img)

            cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)

            # print(f'count={count}')
            # count += 1
            # self.previously_pressed_key = pressed_key
            if self.WITH_QT:
                # if window gets closed then quit
                if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                    break

    def main_loop(self):
        print('self.CLASS_LIST = ', self.CLASS_LIST)
        # load the image list from specified
        if self.specified_image_list is not None:
            self.load_specified_image_list()
        elif self.specified_ann_path_list is not None:
            self.load_specified_annotation_file_list()

        self.annotate_specified_data()

        # if quit the annotation tool, close all windows
        cv2.destroyAllWindows()


class AnnotateSparseSpecifiedBBox(AnnotateSpecifiedBBox, AnnotateSparseBBox):
    def __init__(
            self, image_dir, label_dir, class_list_file,
            specified_image_list=None,
            specified_ann_path_list=None
            ):

        AnnotateSpecifiedBBox.__init__(
            self, image_dir=image_dir,
            label_dir=label_dir, class_list_file=class_list_file,
            specified_image_list=specified_image_list,
            specified_ann_path_list=specified_ann_path_list
        )
        AnnotateSparseBBox.__init__(
            self, image_dir=image_dir,
            label_dir=label_dir, class_list_file=class_list_file,
        )

    def main_loop(self):
        print('self.CLASS_LIST = ', self.CLASS_LIST)

        # load the image list from specified
        if self.specified_image_list is not None:
            self.load_specified_image_list()
        elif self.specified_ann_path_list is not None:
            self.load_specified_annotation_file_list()

        self.load_all_detection()

        self.annotate_specified_data()

        # if quit the annotation tool, close all windows
        cv2.destroyAllWindows()
