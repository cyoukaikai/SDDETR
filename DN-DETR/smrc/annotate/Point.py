#from https://www.daniweb.com/programming/software-development/threads/176391/how-to-move-files-to-another-directory-in-python
# # fro game controller
# for checking the operating system
import cv2
import os
import smrc.utils
from .Bbox import AnnotateBBox
from .SparseBbox import AnnotateSparseBBox
# from smrc.not_used.annotate.lane import *
# from smrc.lane.line.display import *


class AnnotatePoint(AnnotateSparseBBox):
    def __init__(self, image_dir, label_dir, class_list_file=None,
                 user_name=None, music_on=False
                 ):
        super().__init__(
            image_dir=image_dir, label_dir=label_dir,
            class_list_file=class_list_file, user_name=user_name,
            music_on=music_on
        )

        self.IMAGE_WINDOW_NAME = 'PointAnnotation'
        self.point_radius = 5  # the size of the bbox is 2 * self.radius
        self.bbox_radius = 20
        # self.window_width = 1250  # 1000
        # self.window_height = 750  # 700

    def init_annotation_for_active_directory_additional(self):
        ####################################
        # special annotation for this tool
        ######################################
        self.LINE_THICKNESS = self.point_radius
        # self.ACTIVE_BBOX_COLOR = self.RED   # self.RED

        # self.CLASS_BGR_COLORS = np.array([self.RED])

        # self.CLASS_BGR_COLORS = unique_colors

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
                    self.draw_active_bbox(tmp_img)
                    # # if the label of the active bbox is just changed, then change the color of its 8 anchors
                    # # to make the modification more visiable.
                    # if self.label_changed_flag:  # == True
                    #     self.draw_changed_label_bbox(tmp_img)  # no need modification
                    #     self.label_changed_flag = False  # reinitialize the label_changed_flag
                    # else:
                    #     self.draw_active_bbox(tmp_img)

                # what ever self.active_bbox_idx is None or not, we need the reset the self.active_anchor_position
                # if we only update active_anchor_position when self.active_bbox_idx is None,
                # the self.active_anchor_position
                # will never have a chance to be reset to None and thus it will disable the draw_line function.
                self.set_active_anchor_position()
                # if self.active_anchor_position is not None:
                #     ################################################ need modification?
                #     self.draw_active_anchor(tmp_img)

        # we do not draw the mouse_cursor when (dragging_on is True) or (self.active_anchor_position is not None)
        if not self.dragging_on and self.active_anchor_position is None:
            # and (self.game_controller_on is False) and (self.moving_on is False)
            self.draw_line(tmp_img, self.mouse_x, self.mouse_y, self.active_image_height, self.active_image_width,
                           color, 2)  #

        # print('self.display_last_added_bbox_on =', self.display_last_added_bbox_on)
        if self.display_last_added_bbox_on:
            self.draw_last_added_bbox(tmp_img)
            self.display_last_added_bbox_on = False

        ################################################
        self.display_additional_annotation_in_active_image(tmp_img)

        return tmp_img

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

    def draw_line(self, tmp_img, x, y, height, width, color, line_thickness):
        # cv2.line(tmp_img, (x, 0), (x, height), color, line_thickness)
        # cv2.line(tmp_img, (0, y), (width, y), color, line_thickness)
        length = 20
        cv2.line(tmp_img, (x, y - length), (x, y + length), color, line_thickness)
        cv2.line(tmp_img, (x - length, y), (x + length, y), color, line_thickness)

    def draw_annotated_bbox(self, tmp_img, bbox):
        # # the data format should be int type, class_idx is 0-index.
        # class_idx, x, y, w, h = smrc.not_used.bbox_to_xywh(bbox, with_class_index=True)
        # # draw bbox
        # class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
        # radius = self.LINE_THICKNESS
        # cv2.circle(tmp_img, (int(x), int(y)), radius, class_color, thickness=-1)
        self.draw_single_point(tmp_img, bbox)
        # self.draw_single_bbox(tmp_img, bbox, class_color, class_color, class_color)

    def draw_active_bbox(self, tmp_img):
        # self.active_bbox_idx < len(self.active_image_annotated_bboxes) is neccessary, otherwise, it cuases error
        # when we changing the image frame in a very fast speed (dragging trackbar) so that setting acitve bbox
        # is not finished
        if self.is_valid_active_bbox():
            # do not change the class_index here, otherwise every time the active_bbox_idx changed (mouse is wandering),
            # the class_index will change (this is not what we want).
            # the data format should be int type, class_idx is 0-index.
            bbox = self.active_image_annotated_bboxes[self.active_bbox_idx]
            self.draw_single_point(tmp_img, bbox, self.ACTIVE_BBOX_COLOR)
            # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)

    def draw_single_point(self, tmp_img, bbox, color=None):
        class_idx, x, y, w, h = smrc.utils.bbox_to_xywh(bbox, with_class_index=True)
        # draw bbox
        if color is None:
            color = self.CLASS_BGR_COLORS[class_idx].tolist()
        radius = self.LINE_THICKNESS
        x, y = int(round(x)), int(round(y))
        cv2.circle(tmp_img, (x, y), radius, color, thickness=-1)

        width, height = self.active_image_width, self.active_image_height
        x1, y1 = int(round(width / 2)), self.active_image_height-1
        cv2.line(tmp_img, (x1, y1), (x, y), color, 2)

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
                if self.point_1 is None:  # top left corner of the bbox is already decided
                    p1 = (x - self.bbox_radius, y - self.bbox_radius)
                    p2 = (x + self.bbox_radius, y + self.bbox_radius)
                    self.add_bbox(
                        self.active_image_annotation_path,
                        self.active_class_index, p1, p2
                    )
                    # print(f'add_bbox is running ')
                    # print('self.last_added_bbox =', self.last_added_bbox)
                    self.point_1 = None


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

    # def add_bbox(self, ann_path, class_idx, p1, p2):
    #     """
    #     adding bbox after a bbox is drew (p1, p2 given)
    #     :param ann_path: annotation path
    #     :param class_idx: class index
    #     :param p1: first vertex of a rectangle
    #     :param p2: second vertex
    #     :return:
    #     """
    #     # xmin, ymin = min(p1[0], p2[0]), min(p1[1], p2[1])
    #     # xmax, ymax = max(p1[0], p2[0]), max(p1[1], p2[1])
    #     xmin, ymin = p1
    #     xmax, ymax = p2
    #     bbox = [class_idx, xmin, ymin, xmax, ymax]
    #     adding_result = self.add_single_bbox(ann_path, bbox)
    #
    #     if self.fitting_mode_on and adding_result:
    #         self.curve_fitting_dict[self.active_image_index] = bbox
    #         print(f'{bbox} added into {self.active_image_index} ')
    #
    #     self.add_bbox_additional()
    #     return adding_result, bbox

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
            elif pressed_key == ord('w'):
                self.Event_MoveToPrevOrNextBBox('prev')
            elif pressed_key == ord('s'):
                self.Event_MoveToPrevOrNextBBox('next')

            # elif pressed_key == ord('h'): self.Event_ShowHelpInfor()
            elif pressed_key == ord('e'): self.Event_SwitchEdgeShowing()
            elif pressed_key == ord('c'): self.Event_CopyBBox()
            elif pressed_key == ord('v'): self.Event_PasteBBox()
            elif pressed_key == ord('q'): self.Event_CancelDrawingBBox()

            elif pressed_key == ord('r'): self.Event_UndoDeleteSingleDetection()
            # elif pressed_key == ord('f'): self.Event_SwitchFittingMode()
            # elif pressed_key == ord('l'): self.Event_DeleteAndMoveToNextImage(tmp_img)
            # elif pressed_key == ord('j'): self.Event_DeleteAndMoveToPrevImage(tmp_img)
        if pressed_key == self.keyboard['LEFT']: self.Event_MoveBBoxWest()
        elif pressed_key == self.keyboard['UP']: self.Event_MoveBBoxNorth()
        elif pressed_key == self.keyboard['RIGHT']: self.Event_MoveBBoxEast()
        elif pressed_key == self.keyboard['DOWN']: self.Event_MoveBBoxSouth()

        # edit key
        elif pressed_key == self.keyboard['HOME']: self.Event_MoveBBoxNorthWest()
        elif pressed_key == self.keyboard['PAGEUP']: self.Event_MoveBBoxNorthEast()
        elif pressed_key == self.keyboard['END']: self.Event_MoveBBoxSouthWest()
        elif pressed_key == self.keyboard['PAGEDOWN']: self.Event_MoveBBoxSouthEast()

        elif pressed_key == self.keyboard['SPACE']: self.Event_FinishActiveDirectoryAnnotation()
        elif pressed_key == ord(','): self.Event_MoveToPrevDetection(tmp_img)
        elif pressed_key == ord('.'): self.Event_MoveToNextDetection(tmp_img)
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


