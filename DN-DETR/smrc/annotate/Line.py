import cv2
import numpy as np

import smrc.utils
from .Bbox import AnnotateBBox
# from .SparseBbox import AnnotateSparseBBox
# from smrc.lane.line.display import *

# to do work
# automatically change the active lane when mouse is moving


class AnnotateLine(AnnotateBBox):
    def __init__(self, image_dir, label_dir, class_list_file=None,
                 user_name=None, music_on=False
                 ):
        super().__init__(
            image_dir=image_dir, label_dir=label_dir,
            class_list_file=class_list_file, user_name=user_name,
            music_on=music_on
        )

        self.IMAGE_WINDOW_NAME = 'LineAnnotation'
        # self.window_width = 1250  # 1000
        # self.window_height = 750  # 700
        self.move_label = False
        self.BBOX_WIDTH_OR_HEIGHT_THRESHOLD = 5

    def mouse_move_handler_for_dragging(self, x, y):
        if self.point_1 is not None:  # top left corner of the bbox is already decided
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

    def init_annotation_for_active_directory_additional(self):
        ####################################
        # special annotation for this tool
        ######################################

        self.LINE_THICKNESS = 2
        # self.ACTIVE_BBOX_COLOR = self.RED   # self.RED

        # self.CLASS_BGR_COLORS = np.array([self.RED])

        # self.CLASS_BGR_COLORS = unique_colors

    def get_min_rect_for_active_bbox(self):
        if self.is_valid_active_bbox():
            _, x1, y1, x2, y2 = self.active_image_annotated_bboxes[self.active_bbox_idx]
            x_left, y_top, x_right, y_bottom = smrc.utils.get_min_rect(x1, y1, x2, y2)
            return x_left, y_top, x_right, y_bottom
        else:
            return None

    def update_dragging_active_bbox(self):
        if self.is_valid_active_bbox() and self.active_anchor_position is not None:

            # the position of the mouse cursor
            eX, eY = self.mouse_x, self.mouse_y

            # moving the center of the active bbox
            if self.active_anchor_position[0] == "O" or self.active_anchor_position[1] == "O":
                if self.initial_dragging_position is not None:
                    # print('left click, EVENT_LBUTTONDOWN, initial_dragging_position = ',
                    # self.initial_dragging_position)
                    T = (eX - self.initial_dragging_position[0], eY - self.initial_dragging_position[1])
                    # print('T = ', T )
                    self.translate_active_bbox(T)
                    # we need to update the initial_dragging_position when the mouse keeps moving.
                    # otherwise, the T is actually acculating all the translations
                    self.initial_dragging_position = (self.mouse_x, self.mouse_y)
            else:  # moving one of the four boundaries of the active bbox
                ###################################################################
                _, x1, y1, x2, y2 = self.active_image_annotated_bboxes[self.active_bbox_idx]
                x_left, y_top, x_right, y_bottom = smrc.utils.get_min_rect(x1, y1, x2, y2)
                ###################################################################

                anchor_key = self.active_anchor_position[0] + self.active_anchor_position[1]
                # Do not allow the bbox to flip upside down (given a margin)
                margin = 3 * self.BBOX_ANCHOR_THICKNESS
                change_was_made = False
                if self.active_anchor_position[0] == "L":
                    # left anchors (LT, LM, LB)
                    if eX < x_right - margin:
                        x_left = eX
                        change_was_made = True
                elif self.active_anchor_position[0] == "R":
                    # right anchors (RT, RM, RB)
                    if eX > x_left + margin:
                        x_right = eX
                        change_was_made = True
                if self.active_anchor_position[1] == "T":
                    # top anchors (LT, RT, MT)
                    if eY < y_bottom - margin:
                        y_top = eY
                        change_was_made = True
                elif self.active_anchor_position[1] == "B":
                    # bottom anchors (LB, RB, MB)
                    if eY > y_top + margin:
                        y_bottom = eY
                        change_was_made = True

                x_left, y_top, x_right, y_bottom = self.post_process_bbox_coordinate(
                    x_left, y_top, x_right, y_bottom)

                if change_was_made:
                    ################################################################### new
                    if (anchor_key.find('T') > -1 and anchor_key.find('R') > -1) or \
                            (anchor_key.find('B') > -1 and anchor_key.find('L') > -1):
                        # top right, bottom left diagonal
                        self.update_active_bbox_boundaries(x_left, y_bottom, x_right, y_top)
                    else:
                        self.update_active_bbox_boundaries(x_left, y_top, x_right, y_bottom)
                    ###################################################################

    def display_annotation_in_active_image(self):
        # load the class index and class color for plot
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

        ################################################
        # display annotated boxes
        self.draw_bboxes_from_file(tmp_img, self.active_image_annotation_path)  # , image_width, image_height
        ################################################

        self.set_active_bbox_idx_if_NONE()

        # mouse on dragging mode, or self.game_controller_on or self.moving_on
        if self.dragging_on or self.moving_on:
            self.draw_active_bbox(tmp_img)

        else:  # otherwise if mouse is not on dragging mode
            if self.point_1 is not None:  # the mouse is on adding bbox mode
                ################################################
                cv2.line(tmp_img, self.point_1, (self.mouse_x, self.mouse_y), color, self.LINE_THICKNESS)
                # cv2.rectangle(tmp_img, self.point_1, (self.mouse_x, self.mouse_y), color, self.LINE_THICKNESS)
                ################################################
            else:  # the mouse is on wandering mode
                # print('the mouse is in wandering mode......')

                # find the smallest active region of a bbox that the mouse is in and change the active_bbox_idx
                self.set_active_bbox_idx_based_on_mouse_position(allow_none=False)

                # print('self.label_changed_flag = ',self.label_changed_flag)
                if self.active_bbox_idx is not None:
                    # if the label of the active bbox is just changed, then change the color of its 8 anchors
                    # to make the modification more visiable.
                    if self.label_changed_flag:  # == True
                        self.draw_changed_label_bbox(tmp_img)  # no need modification
                        self.label_changed_flag = False  # reinitialize the label_changed_flag
                    else:
                        self.draw_active_bbox(tmp_img)

                # what ever self.active_bbox_idx is None or not, we need the reset the self.active_anchor_position
                # if we only update active_anchor_position when self.active_bbox_idx is None,
                # the self.active_anchor_position
                # will never have a chance to be reset to None and thus it will disable the draw_line function.
                self.set_active_anchor_position()
                if self.active_anchor_position is not None:
                    ################################################ need modification?
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

        ################################################
        self.display_additional_annotation_in_active_image(tmp_img)

        return tmp_img

    # def draw_bboxes_from_file(self, tmp_img, ann_path):
    #     '''
    #     # load the draw bbox from file, and initialize the annotated bbox in this image
    #     for fast accessing (annotation -> self.active_image_annotated_bboxes)
    #         ann_path = labels/489402/0000.txt
    #         print('this ann_path =', ann_path)
    #     '''
    #     self.active_image_annotated_bboxes = []  # initialize the image_annotated_bboxes
    #     if os.path.isfile(ann_path):
    #         with open(ann_path, 'r') as old_file:
    #             lines = old_file.readlines()
    #         old_file.close()
    #
    #         for line in lines:
    #             result = line.split(' ')
    #
    #             # if do the following operation, we will change the displayed
    #             # # the data in txt files are just the coordinates of two points
    #             # # We need to transfer them to its min_rectangle so that we can treat each line as a bbox to
    #             # # use our previous developed annotation tool for bbox operation
    #             # p1, p2 = (int(result[1]), int(result[2])), (int(
    #             #     result[3]), int(result[4]))
    #             # xmin, ymin, xmax, ymax = self.min_rectangle(p1, p2)
    #             # bbox = [int(result[0]), xmin, ymin, xmax, ymax]
    #             bbox = [int(result[0]), int(result[1]), int(result[2]), int(
    #                 result[3]), int(result[4])]
    #             self.active_image_annotated_bboxes.append(bbox)
    #             self.draw_annotated_bbox(tmp_img, bbox)

    # def draw_bboxes_from_file(self, tmp_img, ann_path):
    #     self.active_image_annotated_bboxes = []
    #
    #     lane_list = load_lane_from_txt_file(ann_path=ann_path)
    #     for lane in lane_list:
    #         # generate fake bbox
    #         # bbox = [0] + lane.to_int_list()
    #         bbox = lane.to_int_list()
    #         self.active_image_annotated_bboxes.append(bbox)
    #         self.draw_annotated_bbox(tmp_img, bbox)

    def draw_annotated_bbox(self, tmp_img, bbox):
        # the data format should be int type, class_idx is 0-index.
        class_idx = bbox[0]
        # draw bbox
        class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
        self.draw_single_bbox(tmp_img, bbox, class_color, class_color, class_color)

    def draw_single_bbox(self, tmp_img, bbox, rectangle_color=None,
                         anchor_rect_color=None, text_shadow_color=None,
                         text_color=(0, 0, 0), caption=None):
        class_idx, xmin, ymin, xmax, ymax = bbox
        class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
        if rectangle_color is None: rectangle_color = class_color
        if anchor_rect_color is None: anchor_rect_color = rectangle_color
        if text_shadow_color is None: text_shadow_color = rectangle_color

        cv2.line(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, self.LINE_THICKNESS)
        # cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, 2)
        anchor_rect_color = rectangle_color
        self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, anchor_rect_color, self.LINE_THICKNESS)
        # cv2.addWeighted(image, 0.5, overlay, 0.5, 0, image)

    # def draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color,
    #                      text_shadow_color, text_color=(0, 0, 0)):
    #     class_idx, xmin, ymin, xmax, ymax = bbox
    #
    #     cv2.line(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, self.LINE_THICKNESS)
    #     # cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), rectangle_color, 2)
    #     anchor_rect_color = rectangle_color
    #     self.draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, anchor_rect_color, self.LINE_THICKNESS)

    # draw the 8 bboxes of the anchors around the bbox

    def draw_bbox_anchors(self, tmp_img, xmin, ymin, xmax, ymax, anchor_color, line_thickness=None):
        if line_thickness is None:
            line_thickness = self.LINE_THICKNESS
        anchor_dict = smrc.utils.get_anchors_lane(xmin, ymin, xmax, ymax, line_thickness)

        for anchor_key in anchor_dict:
            x1, y1, x2, y2 = anchor_dict[anchor_key]
            cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), anchor_color, -1)

    def set_active_bbox_idx_based_on_mouse_position(self, allow_none=True):
        """
        With this empty function, we can disable the behaviour of super().set_active_bbox_idx_based_on_mouse_position
        :param allow_none:
        :return:
        """
        pass

    # def set_active_bbox_idx_based_on_mouse_position(self, allow_none=True):
    #     """
    #     Nearest neighbour (ending points from the mouse).
    #     Depreated, because it changes the active line too frequently and takes too much computation resource.
    #     :param allow_none:
    #     :return:
    #     """
    #     # print('self.mouse_x = {}, self.mouse_y = {}'.format(self.mouse_x, self.mouse_y))
    #
    #     if len(self.active_image_annotated_bboxes) == 0:
    #         self.active_bbox_idx = None
    #     elif len(self.active_image_annotated_bboxes) == 1:
    #         self.active_bbox_idx = 0
    #     else:
    #         smallest_dist = 1e+9  # a big value
    #         selected_bbox_inx = 0
    #
    #         for idx, bbox in enumerate(self.active_image_annotated_bboxes):
    #             _, x1, y1, x2, y2 = bbox
    #             # xmin, ymin, xmax, ymax = self.get_bbox_active_region_rectangle(x1, y1, x2, y2)
    #             d1 = smrc.utils.compute_l2_dist([self.mouse_x, self.mouse_y], [x1, y1])
    #             d2 = smrc.utils.compute_l2_dist([self.mouse_x, self.mouse_y], [x2, y2])
    #
    #             # if smrc.utils.point_in_rectangle(self.mouse_x, self.mouse_y, xmin, ymin, xmax, ymax):
    #             #     tmp_area = smrc.utils.get_bbox_area(xmin, ymin, xmax, ymax)
    #             if min(d1, d2) < smallest_dist:
    #                 smallest_dist = min(d1, d2)
    #                 selected_bbox_inx = idx
    #         self.active_bbox_idx = selected_bbox_inx
    #
    #     # if allow_none:
    #     #     self.active_bbox_idx = selected_bbox_inx
    #     # elif not allow_none and selected_bbox_inx is not None:
    #     #     self.active_bbox_idx = selected_bbox_inx

    def set_active_anchor_position(self):
        self.active_anchor_position = None

        # print('self.active_bbox_idx =', self.active_bbox_idx)
        if self.is_valid_active_bbox():
            eX, eY = self.mouse_x, self.mouse_y
            _, x1, y1, x2, y2 = self.active_image_annotated_bboxes[self.active_bbox_idx]
            x_left, y_top, x_right, y_bottom = smrc.utils.get_min_rect(x1, y1, x2, y2)
            # if mouse cursor is inside the inner boundaries of region of the active bbox
            if smrc.utils.point_in_rectangle(eX, eY,
                           x_left - self.BBOX_ANCHOR_THICKNESS,
                           y_top - self.BBOX_ANCHOR_THICKNESS,
                           x_right + self.BBOX_ANCHOR_THICKNESS,
                           y_bottom + self.BBOX_ANCHOR_THICKNESS) \
                    and (not smrc.utils.point_in_rectangle(eX, eY,
                           x_left + self.BBOX_ANCHOR_THICKNESS,
                           y_top + self.BBOX_ANCHOR_THICKNESS,
                           x_right - self.BBOX_ANCHOR_THICKNESS,
                           y_bottom - self.BBOX_ANCHOR_THICKNESS)):

                anchor_dict = smrc.utils.get_anchors_lane(x1, y1, x2, y2, self.BBOX_ANCHOR_THICKNESS)
                valid = False
                for anchor_key in anchor_dict:
                    anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchor_dict[anchor_key]
                    if smrc.utils.point_in_rectangle(eX, eY, anchor_x1, anchor_y1, anchor_x2, anchor_y2):
                        valid = True
                        break
                if valid:  # only if the mouse cursor is in the available anchor bbox (only two for line)
                    # first row: horizontal, second row: vertical, of the center of the 8 anchor rectangles
                    end_points = np.array(
                        [[x_left, x_left, (x_left + x_right) / 2, (x_left + x_right) / 2, x_right],  # horizontal
                         [y_top, y_top, (y_top + y_bottom) / 2, (y_top + y_bottom) / 2, y_bottom]])

                    # print('end_points =', end_points)
                    # left shift, right shift of the center of the anchor rectangle
                    left_right_shift = np.array([-1, 1, -1, 1, -1]) * self.BBOX_ANCHOR_THICKNESS
                    # print('left_right_shift =', left_right_shift)
                    # np.array([[eX], [eY]]) is the mouse_cursor_position

                    end_points = end_points + left_right_shift
                    # print('end_points =', end_points)

                    # print('mouse_curse =', np.array([[eX], [eY]]))
                    position_indicators = np.array([[eX], [eY]]) - end_points

                    # print('position_indicators =', position_indicators)
                    indices = np.sum(position_indicators >= 0, axis=1) - 1  # axis=1, sum over the horizontal direction
                    # print('indices =', indices)

                    self.active_anchor_position = self.ANCHOR_POSITION_ENCODING[0][indices[0]] + \
                                                  self.ANCHOR_POSITION_ENCODING[1][indices[1]]

    def add_bbox(self, ann_path, class_idx, p1, p2):
        """
        adding bbox after a bbox is drew (p1, p2 given)
        :param ann_path: annotation path
        :param class_idx: class index
        :param p1: first vertex of a rectangle
        :param p2: second vertex
        :return:
        """
        # xmin, ymin = min(p1[0], p2[0]), min(p1[1], p2[1])
        # xmax, ymax = max(p1[0], p2[0]), max(p1[1], p2[1])
        xmin, ymin = p1
        xmax, ymax = p2
        bbox = [class_idx, xmin, ymin, xmax, ymax]
        adding_result = self.add_single_bbox(ann_path, bbox)

        # if self.fitting_mode_on and adding_result:
        #     self.curve_fitting_dict[self.active_image_index] = bbox
        #     print(f'{bbox} added into {self.active_image_index} ')

        self.add_bbox_additional()
        return adding_result, bbox

    def Event_MoveBBox(self, direction, option='Line'):
        assert option in ['Line', 'LineUp', 'LineDown']

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
            if option == 'Line':
                self.translate_active_bbox(T)
            elif option == 'LineUp':
                self.translate_line_ending_point(T, UpOrDown='Up')
            elif option == 'LineDown':
                self.translate_line_ending_point(T, UpOrDown='Down')
            self.moving_on = True

    def Event_MoveLineUpPointTowardWest(self):
        self.Event_MoveBBox(direction='west', option='LineUp')

    def Event_MoveLineUpPointTowardNorth(self):
        self.Event_MoveBBox(direction='north', option='LineUp')

    def Event_MoveLineUpPointTowardEast(self):
        self.Event_MoveBBox(direction='east', option='LineUp')

    def Event_MoveLineUpPointTowardSouth(self):
        self.Event_MoveBBox(direction='south', option='LineUp')

    def Event_MoveLineDownPointTowardWest(self):
        self.Event_MoveBBox(direction='west', option='LineDown')

    def Event_MoveLineDownPointTowardNorth(self):
        self.Event_MoveBBox(direction='north', option='LineDown')

    def Event_MoveLineDownPointTowardEast(self):
        self.Event_MoveBBox(direction='east', option='LineDown')

    def Event_MoveLineDownPointTowardSouth(self):
        self.Event_MoveBBox(direction='south', option='LineDown')

    def translate_line_ending_point(self, T, UpOrDown='Up'):
        assert UpOrDown in ['Up', 'Down']

        if self.is_valid_active_bbox():
            active_bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]
            class_idx, x1, y1, x2, y2 = active_bbox

            if y1 < y2:  # point with small y coordinate are on the top of the image
                up_ind, down_ind = 2, 4
            else:
                up_ind, down_ind = 4, 2

            OptDict = {
                'Up': up_ind,
                'Down': down_ind
            }
            print(f'{OptDict}')
            print(f'active_bbox={active_bbox}')
            # do nothing if the translation is illegal (i.e., the bbox is out of the image region
            # [image_width * image_height])
            x, y = active_bbox[OptDict[UpOrDown] - 1], active_bbox[OptDict[UpOrDown]]

            if x + T[0] < 0 or x + T[0] >= self.active_image_width or \
                    y + T[1] < 0 or y + T[1] >= self.active_image_height:
                print('We do nothing because the operation causes the bbox moving out of the image region.')
            else:
                # x1, x2 = x1 + T[0], x2 + T[0]
                # y1, y2 = y1 + T[1], y2 + T[1]
                new_bbox = active_bbox.copy()
                new_bbox[OptDict[UpOrDown] - 1] += T[0]  # x
                new_bbox[OptDict[UpOrDown]] += T[1]  # y

                self.update_active_bbox(new_bbox)

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
            # elif pressed_key == ord('w'): self.Event_MoveToPrevClass()
            # elif pressed_key == ord('s'): self.Event_MoveToNextClass()
            elif pressed_key == ord('w'): self.Event_MoveToPrevOrNextBBox('prev')
            elif pressed_key == ord('s'): self.Event_MoveToPrevOrNextBBox('next')

            # elif pressed_key == ord('h'): self.Event_ShowHelpInfor()
            # elif pressed_key == ord('e'): self.Event_SwitchEdgeShowing()
            elif pressed_key == ord('c'): self.Event_CopyBBox()
            elif pressed_key == ord('v'): self.Event_PasteBBox()
            elif pressed_key == ord('q'): self.Event_CancelDrawingBBox()

            elif pressed_key == ord('r'): self.Event_UndoDeleteSingleDetection()
            elif pressed_key == ord('u'): self.Event_DeleteActiveBBox()
            elif pressed_key == ord('g'): self.Event_DeleteActiveImageBBox()

            elif pressed_key in [ord('k'), ord('l'), ord('j'), ord('h'),
                                 ord('i'), ord('o'), ord('n'), ord('m')]:
                if pressed_key == ord('k'): self.Event_MoveLineUpPointTowardEast()
                # elif pressed_key == ord('l'): T = ['right', self.move_unit]
                elif pressed_key == ord('j'): self.Event_MoveLineUpPointTowardWest()
                # elif pressed_key == ord('h'):  T = ['left', -self.move_unit]
                elif pressed_key == ord('i'): self.Event_MoveLineUpPointTowardNorth()
                # elif pressed_key == ord('o'): T = ['top', self.move_unit]
                # elif pressed_key == ord('n'): T = ['bottom', self.move_unit]
                elif pressed_key == ord('m'): self.Event_MoveLineUpPointTowardSouth()
        elif pressed_key in [ord('/'), ord('\\')]:
            if pressed_key == ord('/'):  self.Event_MoveLineDownPointTowardWest()
            elif pressed_key == ord('\\'): self.Event_MoveLineDownPointTowardEast()

        elif pressed_key == self.keyboard['LEFT']: self.Event_MoveBBoxWest()
        elif pressed_key == self.keyboard['UP']: self.Event_MoveBBoxNorth()
        elif pressed_key == self.keyboard['RIGHT']: self.Event_MoveBBoxEast()
        elif pressed_key == self.keyboard['DOWN']: self.Event_MoveBBoxSouth()

        elif pressed_key == self.keyboard['HOME']: self.Event_MoveToFirstImage()
        elif pressed_key == self.keyboard['END']: self.Event_MoveToLastImage()

        elif pressed_key == ord(','): self.Event_MoveToPrevNoneEmptyImage()
        elif pressed_key == ord('.'): self.Event_MoveToNextNoneEmptyImage()

        elif pressed_key == self.keyboard['-']: self.Event_DeleteAllBBoxForActiveDirectory()
        elif pressed_key == self.keyboard['+']: self.Event_UnDeleteAllBBoxForActiveDirectory()
        # edit key
        # elif pressed_key == self.keyboard['HOME']: self.Event_MoveBBoxNorthWest()
        # elif pressed_key == self.keyboard['PAGEUP']: self.Event_MoveBBoxNorthEast()
        # elif pressed_key == self.keyboard['END']: self.Event_MoveBBoxSouthWest()
        # elif pressed_key == self.keyboard['PAGEDOWN']: self.Event_MoveBBoxSouthEast()

        elif pressed_key == self.keyboard['SPACE']: self.Event_FinishActiveDirectoryAnnotation()

        elif pressed_key == ord(';'): self.Event_MoveToPrevImage()
        elif pressed_key == ord(']'): self.Event_MoveToNextImage()
        # elif pressed_key == ord('w'): self.Event_MoveToPrevClass()
        # elif pressed_key == ord('s'): self.Event_MoveToNextClass()
        elif pressed_key == ord('@'): self.Event_MoveToPrevOrNextBBox('prev')
        elif pressed_key == ord(':'): self.Event_MoveToPrevOrNextBBox('next')

        # elif ord('1') <= pressed_key <= ord('9'):
        #     self.Event_ChangActiveBBoxClassLabel(pressed_key - ord('1')) # use 0-index

        # elif pressed_key & 0xFF == 13:  # Enter key is pressed
        #     print('Enter key is pressed.')
        #     self.Event_SelectFittingKnownData(tmp_img)

        # elif event.key == pygame.K_RETURN: self.Event_MoveBBoxSouthWest()
        # elif event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_SHIFT:
        #     print("pressed: SHIFT + A")
        # elif event.key == pygame.K_a:
        #     print("pressed: SHIFT + A")
        #

        # keyboard listener for additional event
        # self.keyboard_listener_additional(pressed_key, tmp_img)
