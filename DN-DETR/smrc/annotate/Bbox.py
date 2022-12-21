import os
import cv2
import shutil
import time
import sys
import numpy as np

from smrc.utils.annotate.annotation_tool import AnnotationTool
from smrc.utils.annotate.select_directory import SelectDirectory
from smrc.utils.annotate.game_controller import GameController
from smrc.utils.annotate.curve_fit import BBoxCurveFitting
from smrc.utils.annotate.user_log import UserLog
import smrc.utils


class AnnotateBBox(AnnotationTool, GameController):
    def __init__(self, image_dir, label_dir,
                 class_list_file, user_name=None,
                 music_on=False
                 ):
        AnnotationTool.__init__(self, image_dir=image_dir, label_dir=label_dir,
                                class_list_file=class_list_file)
        GameController.__init__(self, music_on)

        ###################################################################
        # future version
        ###################################################################
        # super(AnnotationTool, self).__init__()
        # super(GameController, self).__init__(music_on)
        # self.game_ctl = GameController(music_on)
        # self.keyboard_encoder = KeyboardCoding()

        self.user_logger = UserLog()
        self.IMAGE_DIR = image_dir
        self.LABEL_DIR = label_dir

        #  use the lower case
        self.user_name = user_name
        self.class_list_file = class_list_file

        # if this flag is set to 1, we will close the window or move to another window
        self.annotation_done_flag = False
        self.active_directory = None

        self.IMAGE_WINDOW_NAME = 'FastBBoxRefiningTool'
        self.TRACKBAR_IMG = 'Image'
        self.TRACKBAR_CLASS = 'Class'
        self.TRACKBAR_ANNOTATION_DONE = 'Done'

        self.ANNOTATION_FINAL_DIR = None  # 'annotation_finished'
        self.directory_list_file = None  # 'directory_list.txt'
        self.active_directory_file = None  # 'active_directory.txt'
        self.DIRECTORY_LIST = []

        self.curve_fitter = None
        # do not show the 'Done' bar
        self.show_done_bar = True

        self.move_label = False
        self.full_screen = True

    def set_image_index(self, ind):
        self.active_image_index = ind
        img_path = self.IMAGE_PATH_LIST[self.active_image_index]  # local variable
        # print(f'self.IMAGE_PATH_LIST = {self.IMAGE_PATH_LIST}')
        # print(f'img_path = {img_path}')
        self.active_image = cv2.imread(img_path)
        self.active_image_height, self.active_image_width = self.active_image.shape[:2]
        # if self.WITH_QT:
        #     text = 'Showing image {}/{}, path: {}'.format(self.active_image_index, self.LAST_IMAGE_INDEX,
        #                                                   img_path)
        #     self.display_text(text, 1000)

        checkTrackBarPos = cv2.getTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME)
        if checkTrackBarPos > -1 and checkTrackBarPos != self.active_image_index and \
                0 <= self.active_image_index <= self.LAST_IMAGE_INDEX:
            cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME,
                               self.active_image_index)
            self.reset_after_image_change()

        # print(f'{text}')

    def set_class_index(self, ind):
        self.active_class_index = ind
        # if self.WITH_QT:
        #     text = 'Selected class {}/{} -> {}'.format(self.active_class_index, self.LAST_CLASS_INDEX,
        #                                                self.CLASS_LIST[self.active_class_index])
        #     self.display_text(text, 3000)

        checkTrackBarPos = cv2.getTrackbarPos(self.TRACKBAR_CLASS, self.IMAGE_WINDOW_NAME)

        # never put this in the while loop, otherwise, error 'tuple object
        # is not callable' (probably multiple createTrackbar generated)
        # -1, means bar not exist
        if 0 <= self.active_class_index < len(self.CLASS_LIST) and \
                checkTrackBarPos > -1 and checkTrackBarPos != self.active_class_index:
            cv2.setTrackbarPos(
                self.TRACKBAR_CLASS, self.IMAGE_WINDOW_NAME,
                self.active_class_index
            )

    def visualization_annotation(self, dir_list, skip_empty_annotation=True, result_dir=None):
        self.annotation_tool_initialization()
        if result_dir is None:
            result_dir = self.IMAGE_DIR + '_annotation_visualization'

        for dir_name in dir_list:
            self.active_directory = dir_name
            self.init_annotation_for_active_directory()
            ann_path_list = smrc.utils.get_file_list_in_directory(
                os.path.join(self.LABEL_DIR, self.active_directory)
            )
            if skip_empty_annotation and len(ann_path_list) == 0:
                continue

            smrc.utils.generate_dir_if_not_exist(
                os.path.join(result_dir, self.active_directory)
            )
            for k, image_path in enumerate(self.IMAGE_PATH_LIST):
                self.set_image_index(k)

                image_path = self.IMAGE_PATH_LIST[self.active_image_index]  # image_path is not a global variable
                self.active_image_annotation_path = self.get_annotation_path(image_path)

                if skip_empty_annotation:
                    bbox_list = smrc.utils.load_bbox_from_file(self.active_image_annotation_path)
                    if len(bbox_list) == 0:
                        continue

                tmp_img = self.active_image.copy()
                self.display_additional_infor(tmp_img)
                self.draw_bboxes_from_file(tmp_img, self.active_image_annotation_path)  # , image_width, image_height
                # self.display_additional_annotation_in_active_image(tmp_img)

                cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)
                new_image_file = image_path.replace(self.IMAGE_DIR, result_dir)
                cv2.imwrite(new_image_file, tmp_img)

    def main_loop(self):
        self.annotation_tool_initialization()

        # the loop for the whole annotation tool
        while True:
            # self.active_directory = '284010'
            if self.active_directory is None:
                # a while loop until the self.active_directory is set
                # select directory to conduct object object_tracking
                # select_directory_tool = SelectDirectory(self.DIRECTORY_LIST)

                # specify the directory_list by txt file name
                select_directory_tool = SelectDirectory(
                    directory_list_file=os.path.abspath(self.directory_list_file)
                )
                self.active_directory = select_directory_tool.set_active_directory()

                # update the DIRECTORY_LIST (the user may manually update the dir_list in self.directory_list_file)
                self.DIRECTORY_LIST = select_directory_tool.DIRECTORY_LIST.copy()

            # annotate the active directory if it is set
            if self.active_directory is not None:
                # # the result are written in files (we can read them from disk)
                # tracker.object_tracking(self.active_directory, self.tracking_method)

                # visualize the object_tracking result

                self.annotate_active_directory()

                # reinitialize the self.active_directory
                self.active_directory = None

        # # if quit the annotation tool, close all windows
        # cv2.destroyAllWindows()

    def annotate_active_directory(self):

        self.annotation_done_flag = False

        self.init_annotation_for_active_directory()

        while self.annotation_done_flag is False:

            tmp_img = self.display_annotation_in_active_image()

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

    def annotation_tool_initialization(self):
        self.init_file_path_for_user()

        self.init_directory_list()

        # initialize the class list, last class index, and class index
        # defined in AnnotationTool
        self.init_class_list_and_class_color()
        print('self.CLASS_LIST = ', self.CLASS_LIST)

        self.init_and_play_music()

    def init_file_path_for_user(self):
        # specify the sub directory to save the annotated directory for easy checking
        # of every day's work
        if self.user_name is None or len(self.user_name) == 0:
            self.ANNOTATION_FINAL_DIR = 'annotation_finished'
            self.directory_list_file = 'directory_list.txt'
            self.active_directory_file = 'active_directory.txt'
        else:
            self.user_name = self.user_name.lower()
            self.ANNOTATION_FINAL_DIR = os.path.join('annotation_finished', self.user_name)  # ''
            self.directory_list_file = 'directory_list_' + self.user_name + '.txt'
            self.active_directory_file = 'active_directory_' + self.user_name + '.txt'

        # create the directory or file if they do not exist
        # if not os.path.isdir(self.ANNOTATION_FINAL_DIR):
        #     os.makedirs(self.ANNOTATION_FINAL_DIR)
        if not os.path.isfile(self.directory_list_file):
            open(self.directory_list_file, 'a').close()
        # if not os.path.isfile(self.active_directory_file):
        #     open(self.active_directory_file, 'a').close()

    def init_directory_list(self):
        self.DIRECTORY_LIST = smrc.utils.load_1d_list_from_file(self.directory_list_file)
        for ann_dir in self.DIRECTORY_LIST:
            # check if the directory exists
            f_path = os.path.join(self.IMAGE_DIR, ann_dir)
            assert os.path.isdir(f_path), f'directory {f_path}  does not exist, please check' \
                                          f' {self.directory_list_file}.'
            # self.DIRECTORY_LIST.append(ann_dir)
        print('DIRECTORY_LIST Total:', len(self.DIRECTORY_LIST))

    def update_active_directory_txt_file(self):
        # # Specifying the user name enables different users to use the annotation tool at the same time
        # with open(self.active_directory_file, 'w') as new_file:
        #     txt_line = self.active_directory
        #     new_file.write(txt_line + '\n')
        # new_file.close()

        if self.user_name is None:
            user_name = 'Anonymous'
        else:
            user_name = self.user_name

        assert len(self.IMAGE_PATH_LIST) > 0

        # get the image height and width by loading the first image
        # Note that saving the image height and width for non video frames does not
        # make sense, but we will automatically save the information, and let the user
        # to decide if to use it or not.
        img_height, img_width, _ = cv2.imread(self.IMAGE_PATH_LIST[0]).shape

        self.user_logger.modify_or_add_user(
            user_name, image_dir=self.IMAGE_DIR, label_dir=self.LABEL_DIR,
            active_directory=self.active_directory,
            img_height=img_height, img_width=img_width
        )

    def init_annotation_for_active_directory(self):
        print('Start annotating directory {}'.format(self.active_directory))

        # initialize the self.IMAGE_PATH_LIST, self.LAST_IMAGE_INDEX
        self.load_active_directory()

        # record the active directory to file
        self.update_active_directory_txt_file()

        self.point_1, self.point_2 = None, None
        self.active_image_index = 0
        self.active_class_index = 0
        # this function must be after self.load_image_sequence(), otherwise, the trackBar
        # for image list can not be initialized (as number of image not known)
        self.init_image_window_and_mouse_listener()
        self.init_annotation_for_active_directory_additional()

    def init_annotation_for_active_directory_additional(self):
        pass

    def init_image_window_and_mouse_listener(self):
        # reset the window size, line thickness, font scale if the width of the first image
        # in the image path list is greater than 1000.
        # Here we assume all the images in the directory has the same size, if this is not the
        # case, we may need to design more advanced setting (load all images, or sampling 5 images from
        # the image path list, and then use the maximum size of the image)
        self.init_window_size_font_size()

        # create window
        # do not use cv2.WINDOW_AUTOSIZE, otherwise, changing the window size in the code will not work
        cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
        cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener)

        # show the image index bar, self.set_image_index is defined in AnnotationTool()
        cv2.createTrackbar(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, 0, self.LAST_IMAGE_INDEX, self.set_image_index)

        # show the class index bar only if we have more than one class
        if self.LAST_CLASS_INDEX != 0:
            cv2.createTrackbar(self.TRACKBAR_CLASS, self.IMAGE_WINDOW_NAME, 0, self.LAST_CLASS_INDEX,
                               self.set_class_index)

        # show the annotation status bar, if annotation done is set (the annotation of this directory is done),
        # then move it to "annotation_finished"
        if self.show_done_bar:
            cv2.createTrackbar(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 0, 1, self.move_annotation_result)

        # load the first image in the IMAGE_PATH_LIST
        if self.active_image_index is not None:
            self.set_image_index(self.active_image_index)
        else:
            self.set_image_index(0)
        # self.display_text('Welcome!\n Press [h] for help.', 2000)

    def Event_ChangeLineThickness(self):
        """
        Switch the line thickness to be either 1 or 2.
        :return:
        """
        if self.LINE_THICKNESS == 1:
            self.LINE_THICKNESS = 2
        elif self.LINE_THICKNESS == 2:
            self.LINE_THICKNESS = 1

    def init_window_size_font_size(self):
        # reset the window size, line thickness, font scale if the width of the first image
        # in the image path list is greater than 1000.
        # Here we assume all the images in the directory has the same size, if this is not the
        # case, we may need to design more advanced setting (load all images, or sampling 5 images from
        # the image path list, and then use the maximum size of the image)

        if self.full_screen:
            monitor_width, monitor_height = smrc.utils.annotate.ann_utils.get_monitor_resolution()
            self.window_width = monitor_width - 30 # 1000
            self.window_height = monitor_height - 80  # 700
            self.class_name_font_scale = 0.7
            print(f'Window size: height={self.window_width}, width={self.window_height}')
        else:
            self.window_width = 1920
            self.window_height = 1080
            if len(self.IMAGE_PATH_LIST) > 0:
                image_path = self.IMAGE_PATH_LIST[0]
                height, width = smrc.utils.get_image_size(image_path)

                # change the setting for the window and line thickness if the image width > 1000
                if width > 1000:
                    self.window_width = width + 20  # 1000
                    self.window_height = height + 20  # 700
                    # self.LINE_THICKNESS = 1
                    # self.window_width = 1300  # 1000
                    # self.window_height = 800  # 700
                    self.class_name_font_scale = 0.7
                else:
                    # self.LINE_THICKNESS = 1
                    self.window_width = 1000
                    self.window_height = 700
                    self.class_name_font_scale = 0.6
                print(
                    f'image size: height={self.window_height}, width={self.window_width} '
                    f'for {os.path.dirname(image_path)}')

    def display_additional_trackbar(self, tmp_img):
        """
        use this to add additional trackbar
        :return:
        """
        pass

    def display_additional_infor(self, tmp_img):
        """
        display additional information on the active image
        :param tmp_img:
        :return:
        """
        img_path = self.IMAGE_PATH_LIST[self.active_image_index]
        # image_name = img_path.split(os.path.sep)[-1]
        display_image_path = smrc.utils.file_path_last_two_level(img_path)
        text_content = f'[{display_image_path}] '

        text_content += self.text_infor_for_curve_fitting_status()
        smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)

    def text_infor_for_curve_fitting_status(self):
        text_content = ''
        if self.fitting_mode_on:
            text_content += f'Curve Fitting Mode: On\n'
            # # Random float x, 0.0 <= x < 1.0
            # # conduct curve fitting by chance if an acitve bbox is updated
            # if self.curve_fitting_dict_updated and random.random() > 0.99:
            #     self.curve_fit_manually()

            # cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
            # num_known_bbox = len(self.curve_fitting_dict)
            # if num_known_bbox > 0:
            image_ids = [*self.curve_fitting_dict]
            text_content += f'Known BBox ({len(image_ids)}): '
            if self.window_width <= 1000:
                text_content += self.split_list_to_text_rows(
                    image_ids, num_per_row=8, max_num_row_to_display=2
                )
            else:
                text_content += self.split_list_to_text_rows(
                    image_ids, num_per_row=10,
                    max_num_row_to_display=2
                )
            # ' {image_ids}' + '\n'
            if len(image_ids) > 0:
                image_id_full = list(range(min(image_ids), max(image_ids) + 1))
                image_id_to_fit = [x for x in image_id_full if x not in image_ids]
            else:
                image_id_to_fit = []
            num_bbox_to_fit = len(image_id_to_fit)

            # if num_fitted_data < 20:
            # image_id_list = [x[0] for x in self.last_fitted_bbox]
            text_content += f'BBox to fit ({num_bbox_to_fit}): '
            if self.window_width <= 1000:
                text_content += self.split_list_to_text_rows(
                    image_id_to_fit, num_per_row=6, max_num_row_to_display=1
                )
            else:
                text_content += self.split_list_to_text_rows(
                    image_id_to_fit, num_per_row=10, max_num_row_to_display=1
                )

            # text_content += f'Number of bbox to fit: {num_bbox_to_fit} \n'

        # num_fitted_data = len(self.last_fitted_bbox)
        # if not self.fitting_mode_on and num_fitted_data > 0:
        #     # if num_fitted_data < 20:
        #     image_id_list = [x[0] for x in self.last_fitted_bbox]
        #     text_content += f'fitted bbox: ' + self.split_list_to_text_rows(image_id_list)
        #     text_content += f'Number of fitted bbox: {num_fitted_data}' + '\n'
        return text_content

    def display_annotation_in_active_image(self):
        # load the class index and class color for plot
        # print(f'self.CLASS_BGR_COLORS={self.CLASS_BGR_COLORS}')
        color = self.CLASS_BGR_COLORS[self.active_class_index].tolist()

        # copy the current image
        tmp_img = self.active_image.copy()

        # show the edge if edges_on = True
        if self.edges_on:
            tmp_img = self.draw_edges(tmp_img)  # we have to use the returned tmp_img

        self.display_additional_trackbar(tmp_img)
        self.display_additional_infor(tmp_img)

        # get annotation paths
        image_path = self.IMAGE_PATH_LIST[self.active_image_index]  # image_path is not a global variable
        self.active_image_annotation_path = self.get_annotation_path(image_path)
        # print('annotation_path=', annotation_path)

        # display annotated bboxes
        self.draw_bboxes_from_file(tmp_img, self.active_image_annotation_path)  # , image_width, image_height

        self.set_active_bbox_idx_if_NONE()

        if self.dragging_on or self.moving_on:  # mouse on dragging mode, or self.game_controller_on or self.moving_on
            self.draw_active_bbox(tmp_img)
        else:  # otherwise if mouse is not on dragging mode
            if self.point_1 is not None:  # the mouse is on adding bbox mode
                cv2.rectangle(tmp_img, self.point_1, (self.mouse_x, self.mouse_y), color, self.LINE_THICKNESS)
            else:  # the mouse is on wandering mode
                # print('the mouse is in wandering mode......')

                # find the smallest active region of a bbox that the mouse is in and change the active_bbox_idx
                self.set_active_bbox_idx_based_on_mouse_position(allow_none=False)

                # print('self.label_changed_flag = ',self.label_changed_flag)
                if self.active_bbox_idx is not None:
                    # if the label of the active bbox is just changed, then change the color of its 8 anchors
                    # to make the modification more visible.
                    if self.label_changed_flag:  # == True
                        self.draw_changed_label_bbox(tmp_img)
                        self.label_changed_flag = False  # reinitialize the label_changed_flag
                    else:
                        self.draw_active_bbox(tmp_img)

                # what ever self.active_bbox_idx is None or not, we need the reset the self.active_anchor_position
                # if we only update active_anchor_position when self.active_bbox_idx is None,
                # the self.active_anchor_position
                # will never have a chance to be reset to None and thus it will disable the draw_line function.
                self.set_active_anchor_position()
                if self.active_anchor_position is not None:
                    self.draw_active_anchor(tmp_img)

        # we do not draw the mouse_cursor when (dragging_on is True) or (self.active_anchor_position is not None)
        if not self.dragging_on and self.active_anchor_position is None:
            # and (self.game_controller_on is False) and (self.moving_on is False)
            self.draw_line(tmp_img, self.mouse_x, self.mouse_y, self.active_image_height, self.active_image_width,
                           color, self.LINE_THICKNESS)  #

        # print('self.display_last_added_bbox_on =', self.display_last_added_bbox_on)
        if self.display_last_added_bbox_on:
            self.draw_last_added_bbox(tmp_img)
            self.display_last_added_bbox_on = False

        self.display_additional_annotation_in_active_image(tmp_img)

        return tmp_img

    @staticmethod
    def split_list_to_text_rows(display_list, num_per_row=10, max_num_row_to_display=None):
        if len(display_list) == 0:
            return '[]\n'
        else:
            num_row_total = int(np.ceil(len(display_list) / num_per_row))
            if max_num_row_to_display is not None:
                # print(f'len(display_list) = {len(display_list)}')
                num_row = min(max_num_row_to_display, num_row_total)
            else:
                num_row = num_row_total

            # print(f'num_row = {num_row}')
            # if num_row <= 1:
            #     text_content = '[' + ', '.join(map(str, display_list)) + ']\n'
            # else:  # > 1
            text_content = '['
            for x in range(num_row - 1):
                # print(f'x={x}, {display_list[10*x:10*(x+1)]}')
                text_content += ', '.join(map(str, display_list[num_per_row * x:num_per_row * (x + 1)])) + '\n'

            # for the last row
            if num_row == num_row_total:
                text_content += ', '.join(map(str, display_list[num_per_row * (num_row - 1):])) + ']\n'
            else:
                text_content += ', '.join(map(str, display_list[num_per_row * (num_row - 1):num_per_row * num_row])) \
                                + ' ... ]\n'
            return text_content

    def display_additional_annotation_in_active_image(self, tmp_img):
        if self.fitting_mode_on:
            if self.active_image_index in self.curve_fitting_dict:
                bbox = self.curve_fitting_dict[self.active_image_index]

                # if the bbox is deleted, then we do not show this bbox
                # Note that, when an bbox that is already used as an known data for curve fitting
                # is deleted, we do not
                if bbox in self.active_image_annotated_bboxes:
                    # self.draw_special_bbox(
                    #     tmp_img, bbox=bbox,
                    #     special_color=self.RED
                    # )
                    self.draw_single_bbox(tmp_img, bbox, self.RED, self.RED, self.RED)

                    # bbox_idx = self.active_image_annotated_bboxes.index(bbox)
                    # self.active_bbox_idx = bbox_idx
            # else:
            #     # a list of [image_id, bbox]
            #     image_id_list = [x[0] for x in self.fitted_bbox]
            #     print(f'image_id_list = {image_id_list}')
            #     # print('Enter self.fitted_bbox')
            #     if self.active_image_index in image_id_list:
            #         bbox = self.fitted_bbox[image_id_list.index(self.active_image_index)][1]
            #         # bbox could be None, but it does not matter, as None will not be in
            #         # self.active_image_annotated_bboxes
            #         # print(f'bbox = {bbox} ...')
            #         # if bbox in self.active_image_annotated_bboxes:
            #             # self.active_image_annotated_bboxes.append(bbox)
            #             # self.active_bbox_idx = len(self.active_image_annotated_bboxes) - 1
            #         self.draw_fitted_bbox(
            #             tmp_img, bbox=bbox
            #         )
            #
            #         # bbox_idx = self.active_image_annotated_bboxes.index(bbox)
            #             # self.active_bbox_idx = bbox_idx
            #
            #         # smrc.utils.save_bbox_to_file_incrementally()

    # def draw_fitted_bbox(self, tmp_img, bbox):
    #     # the data format should be int type, class_idx is 0-index.
    #     class_idx, xmin, ymin, xmax, ymax = bbox
    #     #print('get_bbox_area(xmin, ymin, xmax, ymax) = ', get_bbox_area(xmin, ymin, xmax, ymax))
    #
    #     rectangle_color = self.BLUE
    #     #print('The area of this bbox is too small, we use LINE_THICKNESS = 1' )
    #     cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, 2)
    #     # self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, anchor_rect_color, 1)
    #
    #     class_name = self.CLASS_LIST[class_idx]
    #     class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
    #     text_shadow_color = class_color
    #     text_color = (0, 0, 0) #, i.e., black
    #     self.draw_box_id(tmp_img, (xmin, ymin), class_name, self.BLUE, text_color)  #
    #
    # def draw_special_bbox(self, tmp_img, bbox, special_color=(0, 0, 255)):
    #     # the data format should be int type, class_idx is 0-index.
    #     # class_idx = bbox[0]
    #     # # draw bbox
    #     # class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
    #     # class_color = self.ACTIVE_BBOX_COLOR
    #     self.draw_single_bbox(tmp_img, bbox, special_color, special_color, special_color)
    #
    #     # self.draw_single_bbox(tmp_img, bbox, rectangle_color, anchor_rect_color,
    #     #                      text_shadow_color, text_color=(0, 0, 0))

    # mouse callback function
    def mouse_listener(self, event, x, y, flags, param):
        # print('mouse_listener...')
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            # print('mouse move,  EVENT_MOUSEMOVE')

            # Update the dragged bbox when mouse is moving to make it more sensitive to the dragging operation
            # If I put the following command in the train loop, it will have to wait all the mouse event
            # are handled (as mouse event is of high priority).

            # the mouse is on dragging mode, in this mode, active_bbox_idx is already appropriately set
            if self.dragging_on:  # == True
                # update the dragged bbox based on the position of self.mouse_x, self.mouse_y
                self.update_dragging_active_bbox()

            # turn off the moving_on mode (only place)
            if self.moving_on:  # == True
                self.moving_on = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            print('left click, EVENT_LBUTTONDOWN')

            # The active_bbox_idx has been set in the train loop by set_()
            # when the mouse is on wandering mode.
            # If mouse is not on wandering mode (e.g., drawing a bbox), active_bbox_idx remains None.
            # Note that the the active_anchor_position has been set by set_active_anchor_position(),
            # so we do not need to check the active_anchor_position again here
            if self.active_anchor_position is not None:
                # this is the only place to turn dragging mode on
                self.dragging_on = True

                # Each time the dragging_mode is triggered, we record the position of the mouse
                # The initial_dragging_position will be used when we moving the active bbox (x_center, y_center)
                self.initial_dragging_position = (x, y)
                # print('left click, EVENT_LBUTTONDOWN, initial_dragging_position = ', self.initial_dragging_position)
            # no any active_anchor is dragged
            else:
                self.mouse_move_handler_for_dragging(x, y)

        # elif event == cv2.EVENT_MBUTTONDOWN:
        #     print('right button click,  EVENT_RBUTTONDOWN')
        #
        #     # # if self.last_added_bbox is not None:
        #     # #     class_idx, xmin, ymin, xmax, ymax = self.last_added_bbox
        #     # #
        #     # #     # add self.last_added_bbox and update it to the latest added bbox
        #     # #     self.add_bbox(self.active_image_annotation_path, class_idx, (xmin, ymin),
        #     # #                                          (xmax, ymax))
        #     # #     if self.last_added_bbox in self.active_image_annotated_bboxes:
        #     # #         self.active_bbox_idx = self.active_image_annotated_bboxes.index(self.last_added_bbox)
        #     # #         self.moving_on = True
        #     # #     print('self.last_added_bbox =', self.last_added_bbox)
        #     # # if self.active_bbox_idx is not None:
        #     # #     # print('self.active_bbox_idx  =', self.active_bbox_idx)
        #     # #     self.delete_active_bbox_non_max_suppression(self.overlap_suppression_thd)
        #     self.Event_DeleteActiveObject()

        elif event == cv2.EVENT_RBUTTONDOWN:
            print('right button click,  EVENT_RBUTTONDOWN')

            # conduct the delete operation
            # active_bbox_idx has been assigned by function set_active_bbox_idx each time MouseMove event is triggered.
            # print('active_bbox_idx=', self.active_bbox_idx)
            if self.active_bbox_idx is not None:
                # record the self.active_bbox_idx is this is the active we were operating in the last image frame
                self.active_bbox_previous_image = self.active_image_annotated_bboxes[self.active_bbox_idx]

                self.delete_active_bbox()  # active_bbox_idx is global
                self.active_bbox_idx = None  # we must reset the active_bbox_idx to be None, otherwise, strange behaviou, a lot of bbox were deleted.

            # need other initialization here?
            self.dragging_on = False

        elif event == cv2.EVENT_LBUTTONUP:
            print('left button up, EVENT_LBUTTONUP')

            # turn off the dragging mode each time the left button is released
            self.dragging_on = False
            self.initial_dragging_position = None
            '''
            # if the mouse positions of EVENT_LBUTTONUP and EVENT_LBUTTONDOWN are same, indicating adding operation is conducting
            # otherwise, we may conduct dragging operation (modify previously annotated bbox).
            if pointsInSamePosition(initial_dragging_position, (x,y) ):
                #update p1 or p2
                if point_1 is not None: #top left corner of the bbox is already decided
                    point_2 = (x, y)

                    #if point_2 is valid for adding box, then add bbox, otherwise, do nothing
                    add_bbox(annotation_path, class_index, point_1, point_2)

                    point_1, point_2 = None, None
                else: #top left corner of the bbox is not decided yet
                    point_1 = (x, y)


            print('point_1, point_2 = ', point_1, point_2)
            # each time the EVENT_LBUTTONUP is triggered, we eliminate initial_dragging_position

            print('left click, EVENT_LBUTTONUP, initial_dragging_position = ', initial_dragging_position)
            '''
        self.mouse_listener_additional(event, x, y, flags, param)

    def mouse_move_handler_for_dragging(self, x, y):
        if self.point_1 is not None:  # top left corner of the bbox is already decided
            # if point_2 is valid for adding box, then add bbox, otherwise, do nothing
            if abs(x - self.point_1[0]) > self.BBOX_WIDTH_OR_HEIGHT_THRESHOLD \
                    or abs(y - self.point_1[1]) > self.BBOX_WIDTH_OR_HEIGHT_THRESHOLD:
                # second click
                self.point_2 = (x, y)
                self.add_bbox(
                    self.active_image_annotation_path,
                    self.active_class_index, self.point_1,
                    self.point_2
                )
                # print(f'add_bbox is running ')
                # print('self.last_added_bbox =', self.last_added_bbox)
                self.point_1, self.point_2 = None, None
        else:  # first click (top left corner of the bbox is not decided yet)
            self.point_1 = (x, y)

    def mouse_listener_additional(self, event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            print('middle button click,  EVENT_MBUTTONDOWN')
        elif event == cv2.EVENT_RBUTTONUP:
            print('right button up, EVENT_RBUTTONUP')
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print('left button click,  EVENT_LBUTTONDBLCLK')
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            print('right button click,  EVENT_RBUTTONDBLCLK')

        elif flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_MOUSEMOVE:
            print('pressing CTRL key ')
        elif flags == cv2.EVENT_FLAG_SHIFTKEY:
            print('pressing SHIFT key ')
        elif flags == cv2.EVENT_FLAG_ALTKEY:
            print('pressing ALT key ')

        elif flags == cv2.EVENT_FLAG_CTRLKEY and cv2.EVENT_FLAG_LBUTTON:
            print("Left mouse button is clicked while pressing CTRL key - position (", x, ", ", y, ")")
        elif flags == cv2.EVENT_FLAG_RBUTTON + cv2.EVENT_FLAG_SHIFTKEY:
            print("Right mouse button is clicked while pressing SHIFT key - position (", x, ", ", y, ")")
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_ALTKEY:
            print("Mouse is moved over the window while pressing ALT key - position (", x, ", ", y, ")")

    def keyboard_listener(self, pressed_key, tmp_img=None):
        """
                handle keyboard event
                :return:
                """
        # 255 Linux or Windows cv2.waitKeyEx() & 0xFF ,
        # -1  Windows or Linux cv2.waitKeyEx()
        #  0 Windows cv2.waitKey(), -1 Linux  cv2.waitKey()

        if pressed_key == 255 or pressed_key == 0 or pressed_key == -1:
            return
        print('pressed_key=', pressed_key)
        if ord('a') <= pressed_key <= ord('z'):  # 'a': 97, 'z': 122
            if pressed_key == ord('a'): self.Event_MoveToPrevImage()
            elif pressed_key == ord('d'): self.Event_MoveToNextImage()
            elif pressed_key == ord('w'): self.Event_MoveToPrevClass()
            elif pressed_key == ord('e'): self.Event_MoveToNextClass()
            elif pressed_key == ord('s'): self.Event_MoveToNextBBox()

            # elif pressed_key == ord('h'): self.Event_ShowHelpInfor()
            # elif pressed_key == ord('e'): self.Event_SwitchEdgeShowing()
            elif pressed_key == ord('c'): self.Event_CopyBBox()
            elif pressed_key == ord('v'): self.Event_PasteBBox()
            elif pressed_key == ord('b'): self.Event_PasteBBoxForAllSequence()
            elif pressed_key == ord('q'): self.Event_CancelDrawingBBox()

            elif pressed_key == ord('r'): self.Event_UndoDeleteSingleDetection()
            elif pressed_key == ord('u'): self.Event_DeleteActiveBBox()
            elif pressed_key == ord('f'): self.Event_SwitchFittingMode()
            # elif pressed_key == ord('l'): self.Event_DeleteAndMoveToNextImage(tmp_img)
            # elif pressed_key == ord('j'): self.Event_DeleteAndMoveToPrevImage(tmp_img)
            elif pressed_key == ord('x'): self.Event_ChangeLineThickness()

            elif pressed_key in [ord('k'), ord('l'), ord('j'), ord('h'),
                                 ord('i'), ord('o'), ord('n'), ord('m')]:
                if pressed_key == ord('k'): self.Event_MoveBBoxEast()
                # elif pressed_key == ord('l'): T = ['right', self.move_unit]
                elif pressed_key == ord('j'): self.Event_MoveBBoxWest()
                # elif pressed_key == ord('h'):  T = ['left', -self.move_unit]
                elif pressed_key == ord('i'): self.Event_MoveBBoxNorth()
                # elif pressed_key == ord('o'): T = ['top', self.move_unit]
                # elif pressed_key == ord('n'): T = ['bottom', self.move_unit]
                elif pressed_key == ord('m'): self.Event_MoveBBoxSouth()
            elif pressed_key == ord('g'): self.Event_DeleteActiveImageBBox()

        elif pressed_key == self.keyboard['LEFT']: self.Event_MoveBBoxWest()
        elif pressed_key == self.keyboard['UP']: self.Event_MoveBBoxNorth()
        elif pressed_key == self.keyboard['RIGHT']: self.Event_MoveBBoxEast()
        elif pressed_key == self.keyboard['DOWN']: self.Event_MoveBBoxSouth()

        elif pressed_key == ord(','): self.Event_MoveToPrevNoneEmptyImage()
        elif pressed_key == ord('.'): self.Event_MoveToNextNoneEmptyImage()

        # edit key
        elif pressed_key == self.keyboard['HOME']: self.Event_MoveToFirstImage()
        elif pressed_key == self.keyboard['END']: self.Event_MoveToLastImage()
        # elif pressed_key == self.keyboard['HOME']:
        #     self.Event_MoveBBoxNorthWest()
        # elif pressed_key == self.keyboard['PAGEUP']:
        #     self.Event_MoveBBoxNorthEast()
        # elif pressed_key == self.keyboard['END']:
        #     self.Event_MoveBBoxSouthWest()
        # elif pressed_key == self.keyboard['PAGEDOWN']:
        #     self.Event_MoveBBoxSouthEast()

        elif pressed_key == self.keyboard['SPACE']:
            self.Event_FinishActiveDirectoryAnnotation()

        elif pressed_key == self.keyboard['-']: self.Event_DeleteAllBBoxForActiveDirectory()
        elif pressed_key == self.keyboard['+']: self.Event_UnDeleteAllBBoxForActiveDirectory()
        # elif ord('1') <= pressed_key <= ord('9'):
        #     self.Event_ChangActiveBBoxClassLabel(pressed_key - ord('1')) # use 0-index

        elif pressed_key & 0xFF == 13:  # Enter key is pressed
            print('Enter key is pressed.')
            self.Event_SelectFittingKnownData(tmp_img)

        # elif event.key == pygame.K_RETURN: self.Event_MoveBBoxSouthWest()
        # elif event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_SHIFT:
        #     print("pressed: SHIFT + A")
        # elif event.key == pygame.K_a:
        #     print("pressed: SHIFT + A")
        #

        # keyboard listener for additional event
        self.Event_MoveBBoxAlternativeKeySetting(pressed_key)
        self.keyboard_listener_additional(pressed_key, tmp_img)

    def Event_MoveBBoxAlternativeKeySetting(self, pressed_key):
        # whatever windows (remote cannot use <-, ->) or Linux
        if pressed_key in [self.keyboard['F9'], self.keyboard['F10'],
                           self.keyboard['F11'], self.keyboard['F12']]:
            if pressed_key == self.keyboard['F9']: self.Event_MoveBBoxWest()
            elif pressed_key == self.keyboard['F10']: self.Event_MoveBBoxEast()
            elif pressed_key == self.keyboard['F11']: self.Event_MoveBBoxNorth()
            elif pressed_key == self.keyboard['F12']: self.Event_MoveBBoxSouth()

    def keyboard_listener_additional(self, pressed_key, tmp_img):
        # print('Entering keyboard_listener_additional in AnnotatingBbox.')
        if ord('1') <= pressed_key <= ord('9'):
            self.Event_ChangActiveBBoxClassLabel(pressed_key - ord('1'))
            # use 1-index for annotation, but transfer it to 0-index for modifying the class id

    def game_controller_listener(self, tmp_img=None):
        """
        # Possible joystick actions:
        # JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        :param tmp_img:
        :return:
        """
        # and self.game_controller_on
        pass

    def delete_active_directory(self):
        if self.active_directory is None:
            print('The active directory is None, please check ...')
            return

        if self.active_directory not in self.DIRECTORY_LIST:
            self.init_directory_list()

        if self.active_directory in self.DIRECTORY_LIST:
            # We need to update the self.DIRECTORY_LIST so that the self.DIRECTORY_LIST is up to date when
            # we are doing team work.
            self.DIRECTORY_LIST = smrc.utils.remove_one_item_from_1d_list_file(
                list1d_file=self.directory_list_file, item=self.active_directory,
                assert_none_exist=False
                # It is possible that this directory is already been done by user B when the user A is annotating
                # it. In that case, no error will be thrown.
            )
            # Caution: save the remaining directories of users may cause an endless loop to add the finished
            # directories into the self.directory_list_file if they do the annotation work with the same
            # self.directory_list_file

            print(f'{self.active_directory} removed, remaining {len(self.DIRECTORY_LIST)} directories ...')
        else:
            print('self.active_directory not in self.DIRECTORY_LIST, even the '
                  'self.DIRECTORY_LIST is reloaded when executing in the beginning of'
                  ' self.delete_active_directory() ...')

        # print('self.DIRECTORY_LIST =', self.DIRECTORY_LIST)

    def move_annotation_result(self, trackBar_ind):
        if self.move_label and not os.path.isdir(self.ANNOTATION_FINAL_DIR):
            os.makedirs(self.ANNOTATION_FINAL_DIR)

        if trackBar_ind == 1 and self.active_directory is not None:  # if ind is not 0
            source_dir = os.path.join(self.LABEL_DIR, self.active_directory)
            target_dir = os.path.join(self.ANNOTATION_FINAL_DIR, self.active_directory)

            if self.move_label:
                if os.path.isdir(target_dir):  # exception check
                    print(f'The annotation directory {target_dir} already exist, cannot not move '
                          f'{source_dir} to {target_dir}.')
                    smrc.utils.display_text_on_image_top_middle(
                        tmp_img=self.active_image,
                        text_content='The annotation directory already exist, please check.',
                        font_color=smrc.utils.RED
                    )
                    # time.sleep(2)  # wait for 1 second.
                    cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 0)
                    sys.exit(0)
                else:
                    shutil.move(source_dir, self.ANNOTATION_FINAL_DIR)
                # if self.generate_mask:
                # self.auto_masking_active_directory()

            # self.display_text('[Moving annotation result: done].', 1000)
            self.annotation_done_flag = True
            # update the directory list file
            self.delete_active_directory()
            # self.active_directory = None
            # time.sleep(0.5)  # wait for 1 second.

    # def move_annotation_result(self, trackBar_ind):
    #     """
    #     # if annotation done is set (the annotation of this directory is done),
    #     # then move it to "annotation_finished"
    #     # this is only function we allow to modify a global variable.
    #     """
    #     if self.move_label and not os.path.isdir(self.ANNOTATION_FINAL_DIR):
    #         os.makedirs(self.ANNOTATION_FINAL_DIR)
    #
    #     if trackBar_ind == 1 and self.active_directory is not None:  # if ind is not 0
    #         source_dir = os.path.join(self.LABEL_DIR, self.active_directory)
    #         target_dir = os.path.join(self.ANNOTATION_FINAL_DIR, self.active_directory)
    #
    #         if os.path.isdir(target_dir):
    #             print('The annotation directory already exist, please check.')
    #             self.display_text('The annotation directory already exist, please check.', 3000)
    #             time.sleep(1)  # wait for 1 second.
    #             cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 0)
    #         else:
    #             if self.move_label:
    #                 shutil.move(source_dir, self.ANNOTATION_FINAL_DIR)
    #                 self.display_text('[Moving annotation result: done].', 1000)
    #                 self.annotation_done_flag = True
    #
    #                 # update the directory list file
    #                 self.delete_active_directory()
    #             # self.active_directory = None
    #             # time.sleep(1)  # wait for 1 second.

    def curve_fit_manually(self):
        """
        The event will be triggered by
            adding bbox by mouse clicking when fitting mode is on
            selecting an existing bbox by pressing 'Enter' key
                (whatever the fitting mode is on or not, if the fitting mode is not on,
                it will be turned on automatically)
            deleting bbox by mouse clicking when fitting mode is on
            deselecting an existing anchor bbox (known data for curve-fitting) when fitting
                mode is on

            updating a bbox whatever it is a normal bbox or anchor bbox will indirectly trigger this
            function, as it will cause deleting a bbox

        :return:
        """
        # do nothing if len(self.curve_fitting_dict) == 1 as one known point
        # is not enough for curve fitting
        if len(self.curve_fitting_dict) >= 2:
            self.curve_fitter = BBoxCurveFitting(
                image_dir=self.IMAGE_DIR, label_dir=self.LABEL_DIR,
                image_path_list=self.IMAGE_PATH_LIST
            )

            bbox_data_to_fit = []
            class_idx_list = []
            for image_id, bbox in self.curve_fitting_dict.items():
                bbox_data_to_fit.append([image_id, bbox])
                class_idx_list.append(bbox[0])

            # we can instead use majority voting
            new_class_label = class_idx_list[0]

            self.fitted_bbox = self.curve_fitter.bbox_fitting(
                bbox_data=bbox_data_to_fit, class_label_to_fill=new_class_label
            )
            self.fitted_bbox = [x for x in self.fitted_bbox if x[1] is not None]

            self.curve_fitter.save_fitted_bbox_list_overlap_delete_former(
                self.fitted_bbox, overlap_iou_thd=self.curve_fitting_overlap_suppression_thd
            )
            self.curve_fitter.save_fitted_bbox_list_overlap_delete_former(
                bbox_data_to_fit, overlap_iou_thd=self.curve_fitting_overlap_suppression_thd
            )
            # self.curve_fitting_dict_updated = False
            # print(f'self.fitted_bbox = {self.fitted_bbox}')

    def Event_MoveToFirstImage(self):
        self.set_image_index(0)

    def Event_MoveToLastImage(self):
        self.set_image_index(self.LAST_IMAGE_INDEX)

    def Event_MoveToPrevImage(self):
        image_index = smrc.utils.decrease_index(
            self.active_image_index, self.LAST_IMAGE_INDEX
        )
        self.set_image_index(image_index)

    def Event_MoveToNextImage(self):
        image_index = smrc.utils.increase_index(
            self.active_image_index, self.LAST_IMAGE_INDEX
        )
        self.set_image_index(image_index)

    def Event_MoveToPrevNoneEmptyImage(self, max_img_bar_move=500):
        """
        Move to pre image with annotations.
        :param max_img_bar_move: The max number of images allowed to move.
        :return:
        """
        print('Move to previous image that with annotations.')
        if self.active_image_index == 0: return

        old_image_index = self.active_image_index
        for k in range(max_img_bar_move):
            image_index = smrc.utils.decrease_index(
                self.active_image_index, self.LAST_IMAGE_INDEX
            )
            self.active_image_index = image_index

            ann_path = self.get_annotation_path(
                self.IMAGE_PATH_LIST[image_index]
            )
            # print(f'ann_path = {ann_path}')
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            if len(bbox_list) > 0 or image_index == 0:
                self.set_image_index(image_index)
                break
        if self.active_image_index == 0:
            print('Moved to the first image')
        if abs(self.active_image_index - old_image_index) == max_img_bar_move:
            print(f'Moved {max_img_bar_move} images, press again if you want to continue')

    def Event_MoveToNextNoneEmptyImage(self, max_img_bar_move=500):
        """
        Move to next image with annotation.
        :param max_img_bar_move: The max number of images allowed to move.
        :return:
        """
        print(f'Move to next image that with annotation. ')
        if self.active_image_index == self.LAST_IMAGE_INDEX: return

        old_image_index = self.active_image_index
        for k in range(max_img_bar_move):
            image_index = smrc.utils.increase_index(
                self.active_image_index, self.LAST_IMAGE_INDEX
            )
            self.active_image_index = image_index
            # self.set_image_index(image_index)

            ann_path = self.get_annotation_path(
                self.IMAGE_PATH_LIST[image_index]
            )
            # print(f'ann_path = {ann_path}')
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)

            if len(bbox_list) > 0 or image_index == self.LAST_IMAGE_INDEX:
                self.set_image_index(image_index)
                break

        if self.active_image_index == self.LAST_IMAGE_INDEX:
            print('Moved to the last image')
        if abs(self.active_image_index - old_image_index) == max_img_bar_move:
            print(f'Moved {max_img_bar_move} images, press again if you want to continue')

    def Event_MoveToPrevClass(self):
        class_ind = smrc.utils.decrease_index(
            self.active_class_index, self.LAST_CLASS_INDEX
        )
        self.set_class_index(class_ind)

    def Event_MoveToNextClass(self):
        class_ind = smrc.utils.increase_index(
                self.active_class_index, self.LAST_CLASS_INDEX
            )
        self.set_class_index(class_ind)

    def Event_MoveToPrevOrNextBBox(self, prev_or_next):
        if len(self.active_image_annotated_bboxes) < 2:
            return

        if prev_or_next in ['prev', 'next']:
            if prev_or_next == 'prev':
                # print('before change value, self.active_bbox_idx =', self.active_bbox_idx)
                self.active_bbox_idx = smrc.utils.decrease_index(
                    self.active_bbox_idx,
                    len(self.active_image_annotated_bboxes) - 1
                )
            else:
                self.active_bbox_idx = smrc.utils.increase_index(
                    self.active_bbox_idx,
                    len(self.active_image_annotated_bboxes) - 1
                )

    def Event_MoveToPrevBBox(self):
        self.Event_MoveToPrevOrNextBBox('prev')

    def Event_MoveToNextBBox(self):
        self.Event_MoveToPrevOrNextBBox('next')

    def Event_ShowHelpInfor(self):
        text = ('[e] to show edges;\n'
                '[q] to cancel adding bbox;\n'
                '[a] or [d] to change Image;\n'
                '[w] or [s] to change Class.\n'
                '[Esc] to quit the annotation tool.\n'
                )
        self.display_text(text, 1000)

    def Event_SwitchEdgeShowing(self):
        if self.edges_on:
            self.edges_on = False
            self.display_text('Edges turned OFF!', 1000)
        else:
            self.edges_on = True
            self.display_text('Edges turned ON!', 1000)

    def Event_CopyBBox(self):
        self.copy_active_bbox()

    def Event_PasteBBox(self):
        self.paste_last_added_bbox()

    def Event_PasteBBoxForAllSequence(self, nms_thd=0.35):
        # print(f'self.last_added_bbox={self.last_added_bbox}')
        if self.active_bbox_idx is not None and \
                self.active_bbox_idx < len(self.active_image_annotated_bboxes):
            self.curve_fitter = BBoxCurveFitting(
                image_dir=self.IMAGE_DIR, label_dir=self.LABEL_DIR,
                image_path_list=self.IMAGE_PATH_LIST
            )
            bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]
            bbox_data = [[k, bbox] for k in range(len(self.IMAGE_PATH_LIST))]
            self.curve_fitter.save_fitted_bbox_list_overlap_delete_former(
                bbox_data,
                overlap_iou_thd=nms_thd
            )
            # set the active bbox
            if self.last_added_bbox in self.active_image_annotated_bboxes:
                self.active_bbox_idx = self.active_image_annotated_bboxes.index(self.last_added_bbox)
        else:
            print(f'Please select a bbox to copy before pasting it.')

    def Event_UndoPasteBBoxForAllSequence(self):
        if self.active_bbox_idx is not None and \
                self.active_bbox_idx < len(self.active_image_annotated_bboxes):
            self.curve_fitter = BBoxCurveFitting(
                image_dir=self.IMAGE_DIR, label_dir=self.LABEL_DIR,
                image_path_list=self.IMAGE_PATH_LIST
            )
            bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]

            deleted_dict = {}
            for image_path in self.IMAGE_PATH_LIST:
                ann_path = self.get_annotation_path(image_path)
                bbox_list = smrc.utils.load_bbox_from_file(ann_path)
                if bbox in bbox_list:
                    deleted_dict[ann_path] = [bbox]
                    smrc.utils.delete_one_bbox_from_file(ann_path=ann_path, bbox=bbox)
            self.deleted_bbox_history.append(deleted_dict)

    def Event_CancelDrawingBBox(self):
        if self.point_1 is not None:
            self.point_1 = None  # reset point_1
        self.Event_CancelCurveFitting()

    def Event_CancelCurveFitting(self):
        if self.fitting_mode_on:
            self.fitting_mode_on = False
            self.last_fitted_bbox = []
            self.curve_fitting_dict = {}
            self.fitted_bbox = []

    def Event_MoveBBox(self, direction):
        if self.active_bbox_idx is not None:
            if direction == 'west':  # <-- leftward
                T = (-self.move_unit, 0)
            elif direction == 'north':  # | upward
                T = (0, -self.move_unit)
            elif direction == 'east':  # --> rightward
                T = (self.move_unit, 0)
            elif direction == 'south':  # | downward
                T = (0, self.move_unit)

            elif direction == 'northwest':  # | northwest
                T = (-self.move_unit, -self.move_unit)
            elif direction == 'northeast':  # | northeast
                T = (self.move_unit, -self.move_unit)
            elif direction == 'southwest':  # | southwest
                T = (-self.move_unit, self.move_unit)
            elif direction == 'southeast':  # | southeast
                T = (self.move_unit, self.move_unit)
            else:
                raise NotImplementedError

            self.translate_active_bbox(T)
            self.moving_on = True

    def Event_MoveBBoxWest(self):
        self.Event_MoveBBox(direction='west')

    def Event_MoveBBoxNorth(self):
        self.Event_MoveBBox(direction='north')

    def Event_MoveBBoxEast(self):
        self.Event_MoveBBox(direction='east')

    def Event_MoveBBoxSouth(self):
        self.Event_MoveBBox(direction='south')

    def Event_MoveBBoxNorthWest(self):
        self.Event_MoveBBox(direction='northwest')

    def Event_MoveBBoxNorthEast(self):
        self.Event_MoveBBox(direction='northeast')

    def Event_MoveBBoxSouthWest(self):
        self.Event_MoveBBox(direction='southwest')

    def Event_MoveBBoxSouthEast(self):
        self.Event_MoveBBox(direction='southeast')

    def Event_TranslateActiveBBoxBottomBoundary(self, direction, move_unit):
        assert direction in ['top', 'bottom', 'left', 'right']
        if move_unit is None:
            move_unit = self.move_unit
        T = [direction, move_unit]
        self.translate_active_bbox_boundary(T)

        # if btn_name == 'Y':  # | upward
        #     T = ['top', -self.move_unit]
        # elif btn_name == 'A':  # | downward
        #     T = ['bottom', self.move_unit]
        # elif btn_name == 'X':  # <-- leftward
        #     T = ['left', -self.move_unit]
        # elif btn_name == 'B': # --> rightward
        #     T = ['right', self.move_unit]
        # self.translate_active_bbox_boundary(T)
        # self.moving_on = True

    def Event_MoveActiveBBoxBottom_Upward(self, direction, move_unit):
        assert direction in ['top', 'bottom', 'left', 'right']
        if move_unit is None:
            move_unit = self.move_unit
        T = [direction, -abs(move_unit)]
        self.translate_active_bbox_boundary(T)

    def Event_MoveActiveBBoxBottom_Downward(self, direction, move_unit):
        assert direction in ['top', 'bottom', 'left', 'right']
        if move_unit is None:
            move_unit = self.move_unit
        T = [direction, abs(move_unit)]
        self.translate_active_bbox_boundary(T)

    def Event_EnlargeActiveBBox(self, enlarge_unit):
        self.enlarge_active_bbox(wt=enlarge_unit, ht=enlarge_unit)

    def Event_EnlargeActiveBBox_Horizontal(self, enlarge_unit):
        self.enlarge_active_bbox(wt=enlarge_unit, ht=0)

    def Event_EnlargeActiveBBox_Vertical(self, enlarge_unit):
        self.enlarge_active_bbox(wt=0, ht=enlarge_unit)

    def Event_ChangActiveBBoxClassLabel(self, target_class_idx):
        if target_class_idx < len(self.CLASS_LIST):  # '1': 49, '9':57
            # print('HAHA, why self.label_changed_flag ', self.label_changed_flag)
            if self.active_bbox_idx is not None:
                source_class_idx = self.active_image_annotated_bboxes[self.active_bbox_idx][0]

                # print('HAHA, why self.label_changed_flag ', self.label_changed_flag)
                if target_class_idx != source_class_idx:
                    self.update_active_bbox_label(target_class_idx)
                    self.label_changed_flag = True
                    # print('.................self.label_changed_flag ', self.label_changed_flag)
                    # print('To change label of the active bbox to ', self.CLASS_LIST[target_class_idx])
                else:
                    # print('source_class_idx and target_class_idx are same')
                    self.display_text(
                        'Label is not changed, because source_class_idx and target_class_idx are same.',
                        1000)
            else:
                self.display_text('No active bbox is selected.', 1000)
        else:
            self.display_text(f'Class {target_class_idx} not in self.CLASS_LIST {self.CLASS_LIST}', 1000)

    def Event_UndoDeleteSingleDetection(self):
        self.undo_delete_single_bbox()

    def Event_SwitchFittingMode(self, **kwargs):
        """
        # turn on or off the fitting mode
        :return:
        """
        if self.fitting_mode_on:  # fitting mode already on
            # reinitialize the curve_fitting_dict
            self.curve_fit_manually()

            # self.save_fitted_bbox_list(self.fitted_bbox)

            # self.save_fitted_bbox_list_overlap_delete_former(
            #     self.fitted_bbox, overlap_iou_thd=0.5
            # )
            # [ [image_id, bbox] ] not [ [image_id, bbox_list], ... ]
            # self.delete_fitted_bbox(self.fitted_bbox)
            # if self.curve_fitting_dict_updated:
            #     self.curve_fit_manually()

            # self.save_fitted_bbox_list_overlap_delete_former(
            #     self.fitted_bbox, overlap_iou_thd=0.2
            # )
            # # self.save_fitted_bbox_list_overlap_delete_former(
            # #     bbox_data_to_fit, overlap_iou_thd=0.3
            # # )
            self.last_fitted_bbox = self.fitted_bbox[:]
            self.curve_fitting_dict = {}
            self.fitted_bbox = []

        self.fitting_mode_on = not self.fitting_mode_on

    def Event_UndoCurveFitting(self):
        if len(self.last_fitted_bbox) > 0:
            # delete all the fitted bbox of the latest curve fitting operation
            self.curve_fitter.delete_fitted_bbox(self.last_fitted_bbox)

    def Event_SelectFittingKnownData(self, tmp_img=None):
        """
        We do not save the bbox id, as it is sensitive to the change of id for a
        specific bbox (deleting an early existing bbox, update an early existing bbox).
        Empirical experiemnts show that using bbox is better than bbox id.
        1) we do not need to load all bbox_list from ann_path when conduct curve fitting
        2) We can avoid ambiguity issue by using bbox id when adding, updating, deleting a bbox happened.
        only handle 'Enter' event
        For an bbox that is deleted
        :return:
        """
        active_bbox = self.get_active_bbox()
        if active_bbox is not None:
            # if the bbox is already in the dict, then delete it (undo select)
            if self.active_image_index in self.curve_fitting_dict and \
                    self.curve_fitting_dict[self.active_image_index] == \
                    active_bbox:
                del self.curve_fitting_dict[self.active_image_index]

                # turn off the fitting mode if nothing left
                if len(self.curve_fitting_dict) == 0 and self.fitting_mode_on:
                    self.fitting_mode_on = False
            else:
                # self.active_image_index not in self.curve_fitting_dict, or
                # self.active_bbox_idx not equal to self.curve_fitting_dict[self.active_image_index]
                self.curve_fitting_dict[self.active_image_index] = active_bbox

                # turn on the fitting mode if something is selected to fit
                if not self.fitting_mode_on:
                    self.fitting_mode_on = True

            # self.curve_fit_manually()

    def Event_FinishActiveDirectoryAnnotation(self):
        # any of the following two lines will work
        # setTrackbarPos aleady call the function banded with it (i.e., self.move_annotation_result() )
        cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 1)

        # call this function directly also works
        # self.move_annotation_result(1) # move annotation result

    def Event_DeleteAndMoveToPrevImage(self, tmp_img):
        smrc.utils.display_text_on_image_top_middle(tmp_img, '\n\n\n\ndeleting bbox', self.BLUE)
        self.Event_DeleteActiveBBox()
        # smrc.utils.display_text_on_image_top_middle(tmp_img, 'deleting bbox', self.BLUE)

        self.Event_MoveToPrevImage()
        # # time.sleep(0.1)  # wait for 1 second.
        # if self.active_image_index not in self.detection_dict:
        #     self.Event_MoveToPrevDetection(tmp_img)

    def Event_DeleteAndMoveToNextImage(self, tmp_img):
        smrc.utils.display_text_on_image_top_middle(tmp_img, '\n\n\n\ndeleting bbox', self.BLUE)
        self.Event_DeleteActiveBBox()

        # move to next image no matter how many bbox remains for the current image
        self.Event_MoveToNextImage()
        # # time.sleep(0.1)  # wait for 1 second.
        # move to next image only when the current image has no bbox to check
        # if self.active_image_index not in self.detection_dict:
        #     self.Event_MoveToNextDetection(tmp_img)

    def Event_DeleteActiveBBox(self):
        if self.get_active_bbox() is not None:
            # record the self.active_bbox_idx is this is the active we were operating in the last image frame
            # print('active_bbox_idx=', self. active_bbox_idx)
            self.delete_active_bbox()  # active_bbox_idx is global
            self.active_bbox_idx = None

    def Event_DeleteActiveImageBBox(self):
        image_path = self.IMAGE_PATH_LIST[self.active_image_index]
        ann_path = self.get_annotation_path(image_path)
        bbox_list = smrc.utils.load_bbox_from_file(ann_path)
        # assert len(bbox_list) > 0, 'bbox_list should have at least one bbox'
        if len(bbox_list) > 0:
            deleted_dict = {ann_path: bbox_list}
            smrc.utils.empty_annotation_file(ann_path)
            self.deleted_bbox_history.append(deleted_dict)

    def Event_UndoDeleteActiveImageBBox(self):
        self.undo_delete_single_tracklet()
        print(f'Recover bbox for activate image succeed.')

    def Event_UnDeleteAllBBoxForActiveDirectory(self):
        self.undo_delete_single_tracklet()
        print(f'Recover bbox succeed.')

    def Event_DeleteAllBBoxForActiveDirectory(self):
        deleted_dict = {}
        for image_path in self.IMAGE_PATH_LIST:
            ann_path = self.get_annotation_path(image_path)
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            # assert len(bbox_list) > 0, 'bbox_list should have at least one bbox'

            # if we do not update the self.IMAGE_PATH_LIST_WITH_DETECTION_FILE
            if len(bbox_list) > 0:
                deleted_dict[ann_path] = bbox_list
                smrc.utils.empty_annotation_file(ann_path)
        self.deleted_bbox_history.append(deleted_dict)

    def Event_SetAnnotationDone(self):
        self.annotation_done_flag = True

    def AxisEvent_MoveToPrevOrNextBBox(self, axis_name, axis_value):
        if len(self.active_image_annotated_bboxes) < 2:
            return

        if abs(axis_value) > abs(self.axis_states[axis_name]) and \
                abs(axis_value) > 0.2 and \
                len(self.active_image_annotated_bboxes) > 1:

            if self.game_controller_axis_moving_on is False:
                self.game_controller_axis_moving_on = True

                if self.active_bbox_idx is not None:
                    if axis_value > 0:
                        # print('before change value, self.active_bbox_idx =', self.active_bbox_idx)
                        self.active_bbox_idx = smrc.utils.increase_index(
                            self.active_bbox_idx,
                            len(self.active_image_annotated_bboxes) - 1
                        )
                        # print('after change value, self.active_bbox_idx =',
                        #      self.active_bbox_idx)
                    elif axis_value < 0:
                        self.active_bbox_idx = smrc.utils.decrease_index(
                            self.active_bbox_idx,
                            len(self.active_image_annotated_bboxes) - 1
                        )
                        #     ind = self.active_image_index = smrc.utils.decrease_index(self.active_bbox_sorted_index,
                        #                                                  len(self.active_image_annotated_bboxes) - 1)
                        # self.active_bbox_idx = self.active_image_annotated_bboxes_sorted_idx[0][ind]
            else:  # self.game_controller_axis_moving_on = True and axis is keep moving on
                print('the game controller is moving due to momentum, we do nothing.')

        # turn off the game_controller_axis_moving_on mode if the axis is returning to its
        #                         # original position
        elif abs(axis_value) < abs(self.axis_states[axis_name]):
            self.game_controller_axis_moving_on = False

    def AxisEvent_MoveToPrevOrNextImage(self, axis_name, axis_value):
        # ============================================did not work yet
        if self.is_axis_triggered(axis_name, axis_value):
            if axis_value > 0:
                self.Event_MoveToNextImage()
            else:
                self.Event_MoveToPrevImage()
        # ============================================

    def AxisEvent_TranslateActiveBBox(self, axis_name, direction, axis_value):
        if abs(axis_value) > abs(self.axis_states[axis_name]) and \
                abs(axis_value) > 0.2 and \
                len(self.active_image_annotated_bboxes) > 0:  # note: here 0 not 1

            print('Entered the loop')
            if self.game_controller_axis1_moving_on is False:
                self.game_controller_axis1_moving_on = True

                if self.active_bbox_idx is not None:
                    T = None
                    if direction == 'x' and axis_value > 0:  # --> rightward
                        T = (self.move_unit, 0)
                    elif direction == 'x' and axis_value < 0:  # <-- leftward
                        T = (-self.move_unit, 0)
                    elif direction == 'y' and axis_value > 0:  # | downward
                        T = (0, self.move_unit)
                    elif direction == 'y' and axis_value < 0:  # | upward
                        T = (0, -self.move_unit)

                    assert T is not None
                    self.translate_active_bbox(T)
                    self.moving_on = True
            else:  # self.game_controller_axis_moving_on = True and axis is keep moving on
                print('the game controller is moving due to momentum, we do nothing.')

        # turn off the game_controller_axis_moving_on mode if the axis is returning to its
        # original position
        elif abs(axis_value) < abs(self.axis_states[axis_name]):
            self.game_controller_axis1_moving_on = False


class AnnotateBBoxDeprecated(AnnotateBBox):
    def __init__(self,  image_dir, label_dir,
                 class_list_file, user_name=None,
                 music_on=False):
        super().__init__(
            image_dir=image_dir, label_dir=label_dir,
            class_list_file=class_list_file, user_name=user_name,
            music_on=music_on)

    def move_to_specified_image(self, image_id):
        if 0 <= image_id <= self.LAST_IMAGE_INDEX:
            self.set_image_index(image_id)
            # cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME,
            #                    self.active_image_index)
            # self.reset_after_image_change()

    def move_to_pre_or_next_frame(self, prev_or_next):
        if prev_or_next not in ('prev', 'next'):
            print('Please input only "prev" or "next".')
            sys.exit(0)

        if prev_or_next == 'prev':
            self.active_image_index = smrc.utils.decrease_index(
                self.active_image_index, self.LAST_IMAGE_INDEX
            )
        elif prev_or_next == 'next':
            self.active_image_index = smrc.utils.increase_index(
                self.active_image_index, self.LAST_IMAGE_INDEX
            )
        cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME,
                           self.active_image_index)

    def move_to_pre_or_next_class(self, prev_or_next):
        if prev_or_next not in ('prev', 'next'):
            print('Please input only "prev" or "next".')
            sys.exit(0)

        if prev_or_next == 'prev':
            self.active_class_index = smrc.utils.decrease_index(
                self.active_class_index, self.LAST_CLASS_INDEX
            )
        elif prev_or_next == 'next':
            self.active_class_index = smrc.utils.increase_index(
                self.active_class_index, self.LAST_CLASS_INDEX
            )

        cv2.setTrackbarPos(
            self.TRACKBAR_CLASS, self.IMAGE_WINDOW_NAME,
            self.active_class_index
        )

    def Event_MoveToFirstImage(self):
        self.move_to_specified_image(0)

    def Event_MoveToLastImage(self):
        self.move_to_specified_image(self.LAST_IMAGE_INDEX)

    def Event_MoveToNextImage(self):
        self.move_to_pre_or_next_frame('next')
        self.reset_after_image_change()

    def Event_MoveToPrevClass(self):
        prev_or_next = 'prev'
        self.move_to_pre_or_next_class(prev_or_next)

    def Event_MoveToNextClass(self):
        prev_or_next = 'next'
        self.move_to_pre_or_next_class(prev_or_next)

    # def delete_active_object(self, cur_image_id, cur_bbox, max_break_length=3, iou_thd=0.5):
    #     image_id_list_forward = list(range(cur_image_id, len(self.IMAGE_PATH_LIST)))
    #     # push the current bbox to deleted bbox
    #     # record the deleted bbox for recovering purpose
    #
    #     def generate_delete_bbox_dict(image_id_list, deleted_dict=None):
    #         # print(f'cur_image_id = {cur_image_id},cur_bbox = {cur_bbox}')
    #         if deleted_dict is None:
    #             deleted_dict = {}
    #
    #         object_frame_list = [cur_image_id]
    #         object_bbox_list = [cur_bbox]
    #         # start from the first frame so we can delete the current active bbox
    #         for image_id in image_id_list:
    #             object_last_frame = object_frame_list[-1]
    #             if image_id - object_last_frame > max_break_length:
    #                 break
    #
    #             ann_path = self.get_annotation_path(self.IMAGE_PATH_LIST[image_id])
    #             bbox_list = smrc.utils.load_bbox_from_file(ann_path)
    #
    #             max_iou = 0
    #             best_match_bbox_id = None
    #             object_last_bbox = object_bbox_list[-1]
    #             for idx, bbox in enumerate(bbox_list):
    #                 iou = smrc.utils.compute_iou(object_last_bbox[1:5], bbox[1:5])
    #                 if iou > max_iou:
    #                     max_iou = iou
    #                     best_match_bbox_id = idx
    #             if max_iou > iou_thd:
    #                 object_frame_list.append(image_id)
    #
    #                 best_match_bbox = bbox_list[best_match_bbox_id]
    #                 object_bbox_list.append(best_match_bbox)
    #
    #                 # delete the best match bbox
    #                 deleted_dict[ann_path] = [best_match_bbox]
    #                 bbox_list.remove(best_match_bbox)
    #                 smrc.utils.save_bbox_to_file(ann_path=ann_path, bbox_list=bbox_list)
    #
    #                 # display the matched bbox information
    #                 print(f'image_id = {image_id}, bbox = {best_match_bbox}')
    #         return deleted_dict
    #
    #     deleted_dict = generate_delete_bbox_dict(image_id_list_forward)
    #
    #     # checking the current image id is done in the forward round
    #     image_id_list_backward = list(range(cur_image_id-1, -1, -1))
    #     deleted_dict = generate_delete_bbox_dict(image_id_list_backward, deleted_dict)
    #
    #     # print(f'deleted_dict = {deleted_dict}')
    #     self.deleted_bbox_history.append(deleted_dict)

    # def Event_DeleteActiveObject(self):
    #     active_bbox = self.get_active_bbox()
    #     # print(f'active_bbox = {active_bbox}')
    #     if active_bbox is not None:
    #         self.delete_active_object(
    #             cur_image_id=self.active_image_index,
    #             cur_bbox=active_bbox,
    #             max_break_length=5,
    #             iou_thd=0.1
    #         )


    # def curve_fitting_handle_deleting_active_bbox(self, bbox):
    #     assert self.fitting_mode_on
    #     if self.active_image_index in self.curve_fitting_dict and \
    #             self.curve_fitting_dict[self.active_image_index] == \
    #             bbox:
    #
    #         del self.curve_fitting_dict[self.active_image_index]
    #         # self.curve_fit_manually()

    # def curve_fitting_handle_adding_active_bbox(self, bbox):
    #     # this function must be called when self.fitting_mode_on is True
    #     assert self.fitting_mode_on
    #
    #     self.curve_fitting_dict[self.active_image_index] = bbox
    #     # self.curve_fit_manually()
    #     # # self.curve_fitting_dict_updated = True

    def Event_MoveToPrevNoneEmptyImage(self, max_img_bar_move=1000):
        """
        Move to pre image with annotations.
        :param max_img_bar_move: The max number of images allowed to move.
        :return:
        """
        print('Move to previous image that with annotations.')
        if self.active_image_index == 0: return

        for k in range(max_img_bar_move):
            image_index = smrc.utils.decrease_index(
                self.active_image_index, self.LAST_IMAGE_INDEX
            )
            self.set_image_index(image_index)

            ann_path = self.get_annotation_path(
                self.IMAGE_PATH_LIST[image_index]
            )
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            if len(bbox_list) > 0 or image_index == 0:
                # self.set_image_index(image_index)
                break

    def Event_MoveToNextNoneEmptyImage(self, max_img_bar_move=1000):
        """
        Move to next image with annotation.
        :param max_img_bar_move: The max number of images allowed to move.
        :return:
        """
        print(f'Move to next image that with annotation. ')
        if self.active_image_index == self.LAST_IMAGE_INDEX: return

        for k in range(max_img_bar_move):
            image_index = smrc.utils.increase_index(
                self.active_image_index, self.LAST_IMAGE_INDEX
            )
            self.set_image_index(image_index)

            ann_path = self.get_annotation_path(
                self.IMAGE_PATH_LIST[image_index]
            )
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            if len(bbox_list) > 0 or image_index == self.LAST_IMAGE_INDEX:
                # self.set_image_index(image_index)
                break

