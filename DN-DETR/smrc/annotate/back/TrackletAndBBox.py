import cv2
import pygame
import os

import smrc.utils

from smrc.annotate.Bbox import AnnotateBBox
from smrc.object_tracking.visualize import Visualization
from smrc.object_tracking.tracker import TrackerSMRC
from smrc.object_tracking.cnf import IoUTracker, OptFlowPCTracker


# This class has been used for SMRC truck data annotation.
class AnnotateTracklet(AnnotateBBox, TrackerSMRC, Visualization):
    def __init__(self, image_dir, label_dir,
                 class_list_file, user_name=None
                ):
        Visualization.__init__(self)
        TrackerSMRC.__init__(self)
        AnnotateBBox.__init__(
            self, user_name=user_name, image_dir=image_dir,
            label_dir=label_dir, class_list_file=class_list_file
        )

        # recording the cluster id for each bbox for quick access the cluster id
        self.bbox_cluster_IDs = {}
        self.clusters = []
        # the suspicious_bbox_id in self.active_image_annotated_bboxes
        self.suspicious_bbox_id_list = []

        self.cluster_IDs_selected = []
        self.show_only_active_cluster_on = True

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

            # # ==========================================================
            # # this section is new from its parent class
            # # =======================================================
            # if len(self.suspicious_bbox_id_list) > 0:
            #     # as draw_active_bbox may change the color to yellow any way
            #     # we put it here to ensure they have the highest priority
            #     for idx in self.suspicious_bbox_id_list:
            #         bbox_tmp = self.active_image_annotated_bboxes[idx]
            #         self.draw_suspicious_bbox(tmp_img, bbox_tmp)
            # =======================================================
            cv2.imshow(self.IMAGE_WINDOW_NAME, tmp_img)

            if self.WITH_QT:
                # if window gets closed then quit
                if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
                    break

    def keyboard_listener(self, pressed_key, tmp_img=None):
        """
        handle keyboard event
        if pressed_key in [ord('a'), ord('d'), ord('w'), ord('s')]:
        elif pressed_key in [ord('e'), ord('c'), ord('v'), ord('q'), ord('r'), ord('f')]:

        # elif pressed_key in [ord('k'), ord('l'), ord('j'), ord('h'),
            #                      ord('i'), ord('o'), ord('n'), ord('m')]:
            # elif pressed_key in [ord('l'), ord('j'), ord('o'), ord('n')]:
            #     if pressed_key == ord('l'): T = ['right', -self.move_unit]
            #     elif pressed_key == ord('j'): T = ['left', self.move_unit]
            #     elif pressed_key == ord('o'):
            #         T = ['top', self.move_unit]
            #     elif pressed_key == ord('n'):
            #         T = ['bottom',-self.move_unit]
            #     self.translate_active_bbox_boundary(T)
        # elif pressed_key in [ord('k'), ord('l'), ord('j'), ord('h')]:
        #     if pressed_key == ord('j'): self.Event_EnlargeActiveBBox_Horizontal(self.move_unit)
        #     elif pressed_key == ord('h'):
        #         self.Event_EnlargeActiveBBox_Horizontal(-self.move_unit)
        #     elif pressed_key == ord('l'):
        #         self.Event_EnlargeActiveBBox_Vertical(-self.move_unit)
        #     elif pressed_key == ord('k'):
        #         self.Event_EnlargeActiveBBox_Vertical(self.move_unit)

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
            elif pressed_key == ord('s'): self.Event_MoveToNextBBox()

            # elif pressed_key == ord('h'): self.Event_ShowHelpInfor()
            # elif pressed_key == ord('e'): self.Event_SwitchEdgeShowing()
            elif pressed_key == ord('c'): self.Event_CopyBBox()
            elif pressed_key == ord('v'): self.Event_PasteBBox()
            elif pressed_key == ord('q'): self.Event_CancelDrawingBBox()
            elif pressed_key == ord('r'): self.Event_UndoDeleteSingleDetection()
            elif pressed_key == ord('f'): self.Event_SwitchFittingMode()
            elif pressed_key == ord('u'):
                self.Event_DeleteActiveBBox()

            elif pressed_key == ord('x'):
                self.Event_SwitchActiveObjectTrackbar()

            elif pressed_key in [ord('k'), ord('l'), ord('j'), ord('h'),
                                 ord('i'), ord('o'), ord('n'), ord('m')]:
                T = None
                if pressed_key == ord('k'): T = ['right', -self.move_unit]
                elif pressed_key == ord('l'): T = ['right', self.move_unit]
                elif pressed_key == ord('j'): T = ['left', self.move_unit]
                elif pressed_key == ord('h'):  T = ['left', -self.move_unit]
                elif pressed_key == ord('i'): T = ['top', -self.move_unit]
                elif pressed_key == ord('o'): T = ['top', self.move_unit]
                elif pressed_key == ord('n'): T = ['bottom', self.move_unit]
                elif pressed_key == ord('m'): T = ['bottom', -self.move_unit]
                self.translate_active_bbox_boundary(T)

        elif (pressed_key == ord(',') or pressed_key == ord('.')) and \
                len(self.clusters) > 1:  # self.show_only_active_cluster_on and
            if pressed_key == ord(','): self.Event_MoveToPrevObject()
            elif pressed_key == ord('.'): self.Event_MoveToNextObject()
        elif pressed_key == ord('/'): self.Event_EnlargeActiveBBox(self.move_unit)
        elif pressed_key == ord("\\"): self.Event_EnlargeActiveBBox(-self.move_unit)

        elif pressed_key in [self.keyboard['LEFT'], self.keyboard['UP'],
                             self.keyboard['RIGHT'], self.keyboard['DOWN']]:
            if pressed_key == self.keyboard['LEFT']: self.Event_MoveBBoxWest()
            elif pressed_key == self.keyboard['UP']: self.Event_MoveBBoxNorth()
            elif pressed_key == self.keyboard['RIGHT']: self.Event_MoveBBoxEast()
            elif pressed_key == self.keyboard['DOWN']: self.Event_MoveBBoxSouth()

        elif ord('1') <= pressed_key <= ord('9'):
            target_label = pressed_key - ord('1')
            self.Event_ChangActiveBBoxClassLabel(target_label)  # use 0-index

        # elif pressed_key == self.keyboard['HOME']: self.Event_MoveBBoxNorthWest()
        # elif pressed_key == self.keyboard['PAGEUP']: self.Event_MoveBBoxNorthEast()
        # elif pressed_key == self.keyboard['END']: self.Event_MoveBBoxSouthWest()
        # elif pressed_key == self.keyboard['PAGEDOWN']: self.Event_MoveBBoxSouthEast()

        # edit key
        elif pressed_key == self.keyboard['HOME']: self.Event_MoveToActiveObjectStart()
        elif pressed_key == self.keyboard['END']: self.Event_MoveToActiveObjectEnd()
        # elif pressed_key == self.keyboard['PAGEUP']: self.Event_MoveBBoxNorthEast()
        elif pressed_key == self.keyboard['PAGEDOWN']: self.Event_MoveToNextMusic()

        elif pressed_key == self.keyboard['SPACE']: self.Event_FinishActiveDirectoryAnnotation()
        elif pressed_key == ord(';'):
            self.select_or_deselect_one_cluster()
        elif pressed_key & 0xFF == 13:  # Enter key is pressed
            print('Enter key is pressed.')
            self.Event_SelectFittingKnownData(tmp_img)
        elif pressed_key == self.keyboard['DELETE']:
            print(f'Delete pressed, to delete active cluster')
            self.Event_DeleteActiveObject()
        elif pressed_key == self.keyboard['INSERT']:
            print(f'INSERT pressed, to recover deleted active cluster')
            self.Event_UndoDeleteActiveObject()

        elif pressed_key == self.keyboard['-']: self.Event_DeleteAllBBoxForActiveDirectory()

        elif pressed_key == ord('@') or pressed_key == 91 or pressed_key == ord('b'):
            self.select_or_deselect_major_cluster()
        elif pressed_key == self.keyboard['F5']:
            self.Event_RefreshTrackingResult(self.active_image_index)

        self.Event_MoveBBoxAlternativeKeySetting(pressed_key)
        self.keyboard_listener_additional(pressed_key, tmp_img)

    def keyboard_listener_additional(self, pressed_key, tmp_img):
        # print('Entering keyboard_listener_additional in AnnotateTrackletAndBBoxInDirectory.')
        pass
        #
        # if pressed_key == ord(']'):
        #     print('] key is pressed.')
        #     self.Event_SelectActiveObject()
        # elif pressed_key == ord('@'):
        #     self.curve_fitting_active_cluster()
        # elif pressed_key == ord('/'):
        #     self.curve_fitting_all_clusters()

        # elif pressed_key in [ord('k'), ord('l'), ord('j'), ord('h')]:
        # elif pressed_key == ord('l'): self.Event_DeleteAndMoveToNextImage(tmp_img)
        # elif pressed_key == ord('j'): self.Event_DeleteAndMoveToPrevImage(tmp_img)

        # elif pressed_key == ord('m'):
        #     self.Event_MoveToNextMusic()
        # elif ord('1') <= pressed_key <= ord('9'):
        #     self.Event_ChangActiveBBoxClassLabel(pressed_key - ord('1')) # use 0-index
        # elif ord('1') <= pressed_key <= ord('9'):
        #     target_label = pressed_key - ord('1')
        #     # # for generate the training data for driver classification
        #     # self.Event_ChangActiveObjectClassLabel(target_label=target_label)
        #
        #     if self.show_only_active_cluster_on:
        #         self.Event_ChangActiveObjectClassLabel(target_label=target_label)
        #         self.active_cluster_id = None
        #         self.show_only_active_cluster_on = False
        #     else:
        #         self.Event_ChangActiveBBoxClassLabel(target_label)  # use 0-index

        # elif pressed_key == ord('z'): self.blur_bbox = not self.blur_bbox

    def game_controller_listener(self, tmp_img=None):
        """Remained: If I need to modify this function, I should refer to the old
            implementation in the Server.
        # Possible joystick actions:
        # JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        :param tmp_img:
        :return:
        """
        # and self.game_controller_on

        # print('Enter game controller listener')

        if self.game_controller_available and self.game_controller_on:
            # ============================================did not work yet
            if 'x' in self.axis_states:  # move the bbox by axis #and self.axis_states['x'] != 0
                self.AxisEvent_MoveToPrevOrNextImage(axis_name='x', axis_value=self.axis_states['x'])
            # =======================================================
            # fast moving image frame (no matter if any object_detection exists)
            # only for 'SHANWAN IFYOO Gamepad' (name for linux system) and (IFYOO game) for Windows system.
            # if 'B' in self.button_states and self.button_states['B'] == 1:
            #     self.Event_DeleteAndMoveToNextDetection(tmp_img)

            # if 'hat0' in self.hat_states:
            #     # print(f"self.hat_states['hat0']  = {self.hat_states['hat0']}" )
            #     if self.hat_states['hat0'][0] == 1:
            #         self.Event_MoveToNextDetection(tmp_img)
            #     elif self.hat_states['hat0'][0] == -1:
            #         self.Event_MoveToPrevDetection(tmp_img)

            # EVENT PROCESSING STEP
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    break

                if event.type == pygame.USEREVENT:  # A object_tracking has
                    print('Automatically move to next music ...')
                    self.Event_MoveToNextMusic()
                    # sys.exit(0)

                # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
                if event.type == pygame.JOYBUTTONDOWN:
                    # print("Joystick button pressed.")
                    # print('event =', event)
                    # ('event =', < Event(10-JoyButtonDown {'joy': 0, 'button': 2}) >)
                    # print('event["button"] =', event['button'])  # this is wrong
                    # print('event.button =', event.button) #'event.key =', event.key,

                    # get_button
                    btn_id = event.button  # get the id of the button
                    btn_name = self.button_names[btn_id]  # get the name of the button
                    print("%s pressed" % (btn_name))
                    self.button_states[btn_name] = 1  # set the state of the button to 1

                    # # self.set_active_bbox_idx_for_game_controller()
                    if btn_name == 'L1':
                        # return to select directory
                        self.Event_SetAnnotationDone()
                    elif btn_name == 'R1':
                        self.undo_delete_unselected_object()

                    # elif btn_name == 'B': self.Event_DeleteActiveBBox()
                    elif btn_name == 'B':
                        self.select_or_deselect_major_cluster()
                    elif btn_name == 'X':
                        self.cancel_selecting_object()
                    # # elif pressed_key == ord('h'): self.Event_DeleteAndMoveToPrevDetection(tmp_img)

                    elif btn_name == 'Y':
                        # move bbox bottom up
                        # direction in ['top', 'bottom', 'left', 'right']
                        self.Event_MoveActiveBBoxBottom_Upward('bottom', self.move_unit)
                    elif btn_name == 'A':
                        # move bbox bottom down
                        self.Event_MoveActiveBBoxBottom_Downward('bottom', self.move_unit)

                    elif btn_name == 'start':
                        print('start pressed, set annotation done.')
                        self.Event_FinishActiveDirectoryAnnotation()
                    elif self.js_name.find('JC-U3613M') >= 0:  # JC-U3613M game controller
                        if btn_name == 'back':
                            print('back pressed, delete all bbox ...')
                            # cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 1)
                            self.Event_DeleteAllBBoxForActiveDirectory()
                    elif self.js_name.find('IFYOO') >= 0:  # SHANWAN IFYOO Gamepad
                        if btn_name == 'select':
                            print('select pressed, delete all bbox ...')
                            # cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 1)
                            self.Event_DeleteAllBBoxForActiveDirectory()

                #         # elif self.button_states['L2'] == 1:
                #         #     self.Event_MoveToPrevImage()
                #         # elif self.button_states['R2'] == 1:
                #         #     self.Event_MoveToNextImage()
                #
                if event.type == pygame.JOYBUTTONUP:
                    # print("Joystick button released.")
                    btn_id = event.button  # get the id of the button
                    btn_name = self.button_names[btn_id]  # get the name of the button
                    # print("%s released" % (btn_name))
                    self.button_states[btn_name] = 0  # set the state of the button to 0

                # JOYAXISMOTION parameter:  joy, hat, value
                if event.type == pygame.JOYHATMOTION:
                    # print("Joystick hat pressed.")
                    # print('event =', event)

                    hat_id, hat_value = event.hat, event.value  # get the id of the hat
                    hat_name = self.hat_names[hat_id]  # get the name of the hat
                    print("%s pressed, " % (hat_name))
                    print("hat value  : {}".format(hat_value))
                    self.hat_states[hat_name] = hat_value  # set the state of the hat to 1

                    if hat_name == 'hat0':
                        if hat_value[0] == 1:
                            self.Event_MoveToNextImage()  # 'pre' or 'next'
                        elif hat_value[0] == -1:  # leftward, xmin will decrease, hat_value[0] < 0
                            self.Event_MoveToPrevImage()
                        elif hat_value[1] == 1:  # upward, ymin will decrease, but hat_value[1] > 0
                            self.Event_MoveToNextBBox()
                        elif hat_value[1] == -1:
                            # T = ['bottom', int(direction * (-hat_value[1]) * self.increase_decrease_unit)]
                            self.Event_MoveToPrevBBox()

                # # JOYAXISMOTION parameter:  joy, axis, value
                # if event.type == pygame.JOYAXISMOTION:
                #     print("Joystick axis pressed.")
                #     print('event =', event)
                #     # ('event =', < Event(7-JoyAxisMotion {'joy': 0, 'value': 0.0, 'axis': 3}) >)
                #
                #     # get_axis
                #     axis_id = event.axis  # get the id of the axis
                #     print('self.axis_names.keys =', self.axis_names.keys())
                #     print('axis_id =', axis_id)
                #
                #     axis_name = self.axis_names[axis_id]  # get the name of the axis
                #     print("%s axis pressed" % (axis_name))
                #     axis_value = event.value
                #     print("axis value  : {}".format(axis_value))
                #
                #     if axis_name == 'x':
                #         self.AxisEvent_MoveToPrevOrNextBBox(axis_name, axis_value)
                #     elif axis_name == 'rx':  # move the bbox by axis
                #         self.AxisEvent_TranslateActiveBBox(axis_name, 'x', axis_value)
                #     elif axis_name == 'ry':
                #         self.AxisEvent_TranslateActiveBBox(axis_name, 'y', axis_value)
                #
                #     self.axis_states[axis_name] = axis_value

    def load_data_for_tracking(self):
        # dir_name = os.path.join(self.IMAGE_DIR, self.active_directory)
        video_annotation_list = []
        for image_name in self.IMAGE_PATH_LIST:
            # load the labels
            ann_path = smrc.utils.get_image_or_annotation_path(
                image_name, self.IMAGE_DIR, self.LABEL_DIR,
                '.txt'
            )
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            video_annotation_list.append(
                [image_name, bbox_list]
            )
        return video_annotation_list

    def draw_suspicious_bbox(self, tmp_img, bbox):
        class_color = self.GREEN  # red to make it more visible
        self.draw_single_bbox(tmp_img, bbox, class_color, class_color, class_color)
        # draw_single_bbox(self, tmp_img, bbox, rectangle_color, anchor_rect_color, text_shadow_color)

    def Event_SelectActiveObject(self):
        pass
        # if len(self.IMAGE_PATH_LIST) == 0 or self.active_image_index is None \
        #         or self.active_image_index >= len(self.IMAGE_PATH_LIST):
        #     return
        #
        # image_path = self.IMAGE_PATH_LIST[self.active_image_index]
        # if not self.show_only_active_cluster_on and len(image_path) > 0 and \
        #     self.active_bbox_idx is not None and \
        #     self.active_bbox_idx < len(self.active_image_annotated_bboxes):
        #     global_bbox_id = self.get_global_bbox_id_from_bbox(
        #         image_path, self.active_image_annotated_bboxes[self.active_bbox_idx]
        #     )
        #     self.active_cluster_id = self.get_cluster_id_from_bbox_cluster_IDs(global_bbox_id)
        #
        #     if self.active_cluster_id is not None:
        #         self.show_only_active_cluster_on = True
        #
        #         # self.visualize_active_cluster()
        # else:
        #     self.active_cluster_id = None
        #     self.show_only_active_cluster_on = False

    def Event_DeleteActiveObject(self):
        active_cluster_id, active_bbox = self.get_active_cluster_id_from_active_bbox()
        if active_cluster_id is not None:
            self.delete_one_cluster(self.clusters[active_cluster_id])
        else:
            if active_bbox is not None:
                self.Event_DeleteActiveBBox()

    def delete_one_cluster(self, cluster_to_delete):
        deleted_dict = self.delete_one_cluster_bbox_and_generate_deleted_dict(cluster_to_delete)
        self.deleted_bbox_history.append(deleted_dict)

    def delete_one_cluster_bbox_and_generate_deleted_dict(
            self, cluster_to_delete, deleted_dict=None):
        if deleted_dict is None:
            deleted_dict = {}

        for global_bbox_id in cluster_to_delete:
            image_id = self.get_image_id(global_bbox_id)
            bbox = self.get_single_bbox(global_bbox_id)
            image_path = self.IMAGE_PATH_LIST[image_id]

            ann_path = self.get_annotation_path(image_path)
            bbox_list = [bbox]
            assert len(bbox_list) > 0, 'bbox_list should have at least one bbox'
            # delete this bbox from file
            smrc.utils.delete_one_bbox_from_file(ann_path=ann_path, bbox=bbox)
            deleted_dict[ann_path] = bbox_list
        return deleted_dict

    def Event_UndoDeleteActiveObject(self):
        self.undo_delete_single_tracklet()
        print(f'Recover deleted track succeed.')

    def Event_RefreshTrackingResult(self, current_image_index):
        self.cluster_IDs_selected = []
        # self.init_annotation_for_active_directory()
        self.init_annotation_for_active_directory_additional()

        self.set_image_index(current_image_index)
        cv2.setTrackbarPos(
            self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, current_image_index
        )

    def modify_active_cluster_class_label(self, target_label):
        if 0 <= target_label < len(self.CLASS_LIST):
            active_cluster_id, active_bbox = self.get_active_cluster_id_from_active_bbox()
            if active_cluster_id is not None:
                print(f'Changing label for {len(self.clusters[active_cluster_id])} bbox...')
                for global_bbox_id in self.clusters[active_cluster_id]:
                    image_id = self.get_image_id(global_bbox_id)
                    bbox = self.get_single_bbox(global_bbox_id)
                    image_path = self.IMAGE_PATH_LIST[image_id]

                    ann_path = self.get_annotation_path(image_path)
                    bbox_list = smrc.utils.load_bbox_from_file(ann_path)
                    bbox_rect_list = [bbox[1:5] for bbox in bbox_list]
                    try:
                        local_bbox_idx = bbox_rect_list.index(bbox[1:5])
                        bbox_list[local_bbox_idx][0] = target_label
                        smrc.utils.save_bbox_to_file(ann_path, bbox_list)
                        print(f'{ann_path}, succeed changing label from {bbox} to {bbox_list[local_bbox_idx]}')

                        # # update the reference for object_tracking
                        # self.frame_dets[image_id]
                    except IndexError:
                        print(f'bbox {bbox} not found in {ann_path}, bbox_list = {bbox_list}')

                current_image_index = self.active_image_index
                self.Event_RefreshTrackingResult(current_image_index)

            else:
                if active_bbox is not None:
                    self.Event_ChangActiveBBoxClassLabel(target_label)
        else:
            print(f'target label {target_label} not in valid range.')

    def select_or_deselect_one_cluster(self):
        """
        select or deselect the bbox of a cluster by selecting the active bbox or un selecting
        the active bbox
        :return:
        """
        active_cluster_id, active_bbox = self.get_active_cluster_id_from_active_bbox()
        if active_cluster_id is not None:
            for global_bbox_id in self.clusters[active_cluster_id]:
                image_id = self.get_image_id(global_bbox_id)
                bbox = self.get_single_bbox(global_bbox_id)
                if image_id in self.curve_fitting_dict and \
                        bbox == self.curve_fitting_dict[image_id]:
                    del self.curve_fitting_dict[image_id]
                else:
                    self.curve_fitting_dict[image_id] = bbox

            if len(self.curve_fitting_dict) > 0 and not self.fitting_mode_on:
                self.fitting_mode_on = True
        else:
            # only handle the active bbox if it is not included in any cluster
            self.Event_SelectFittingKnownData()

    def get_active_cluster_id_from_active_bbox(self):
        active_cluster_id = None
        active_bbox = self.get_active_bbox()
        if active_bbox is not None:
            if self.active_image_index is not None and \
                    self.active_image_index < len(self.IMAGE_PATH_LIST):
                active_cluster_id = self.get_cluster_id_from_bbox(active_bbox, self.active_image_index)
                # image_path = self.IMAGE_PATH_LIST[self.active_image_index]
                # global_bbox_id = self.get_global_bbox_id_from_bbox(
                #     image_path, active_bbox
                # )
                # if global_bbox_id is None:
                #     return None, None

                # active_cluster_id = self.get_cluster_id_from_bbox_cluster_IDs(global_bbox_id)
        return active_cluster_id, active_bbox

    def get_cluster_id_from_bbox(self, bbox, image_id):
        if 0 <= image_id < len(self.IMAGE_PATH_LIST):
            image_path = self.IMAGE_PATH_LIST[image_id]
            global_bbox_id = self.get_global_bbox_id_from_bbox(
                image_path, bbox
            )
            if global_bbox_id is None:
                return None
            else:
                cluster_id = self.get_cluster_id_from_bbox_cluster_IDs(global_bbox_id)
                return cluster_id
    #################################################
    # functions  modified after new tracking interface
    ###################################################

    def tracking_main(self):
        """
        conduct offline object_tracking
        :return:
        """
        # initialize all the clustering result
        self.bbox_cluster_IDs = {}
        self.clusters = []
        self.cluster_IDs_selected = []

        txt_list = smrc.utils.get_file_list_in_directory(os.path.join(self.LABEL_DIR, self.active_directory))
        if len(txt_list) == 0:
            return
        else:

            my_tracker = IoUTracker()  # OptFlowPCTracker
            video_annotation_list = self.load_data_for_tracking()
            num_bbox = sum([len(x[1]) for x in video_annotation_list])  # x = [image_path, bbox_list]
            if num_bbox > 0:
                # self.clusters, self.video_detected_bbox_all =
                my_tracker.offline_tracking(
                    video_detection_list=video_annotation_list,
                    # max_dist_thd=0.9,  # max_l2_dist
                    max_frame_gap=5,
                    num_pixel_bbox_to_extend=30  # extend each bbox 50 pixels from four directions
                )
                self.from_tracker(my_tracker)
                # clusters, annotated_bbox_all = self.offline_tracking(video_annotation_list)
                # self.recover_frame_dets()
                self.clusters = self.sorted_clusters_based_on_image_id(self.clusters)
                # self.cluster_labels = self.estimate_cluster_label(self.clusters)
                self.estimate_display_object_id(self.CLASS_LIST)
                self._assign_cluster_id_to_global_bbox_idx()

                # # automatically correct the class label
                # This should be used only once when the detections are first imported.
                # ===============================================================
                # deprecated: is the following comment still useful? No
                # we must not automatically correct the class label by (self.correct_class_label_by_majority_voting())
                # if the labels are modified in the txt files, as they are not modified in self.video_detected_bbox_all
                #  delete one cluster will not able to delete it.
                # ===============================================================
                self.correct_class_label()

    def init_annotation_for_active_directory_additional(self):
        self.cancel_selecting_object()
        self.tracking_main()
        self.other_init_annotation_for_active_directory_additional()

    def other_init_annotation_for_active_directory_additional(self):
        pass

    def display_additional_infor_v0(self, tmp_img):
        img_path = self.IMAGE_PATH_LIST[self.active_image_index]
        image_name = img_path.split(os.path.sep)[-1]
        text_content_for_image_name = f'[{self.active_directory}/{image_name}] '

        if self.fitting_mode_on:
            text_content = text_content_for_image_name
            text_content += self.text_infor_for_curve_fitting_status()
            smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)
        else:
            active_cluster_id, active_bbox = self.get_active_cluster_id_from_active_bbox()
            if active_cluster_id is not None:
                # text_content = 'Press Enter to show all BBox.\n'
                # text_content += 'Press Del to delete all the object_tracking result. \n'
                text_content = ''

                if active_bbox is not None:
                    global_bbox_id = self.get_global_bbox_id_from_bbox(
                        self.IMAGE_PATH_LIST[self.active_image_index],
                        self.active_image_annotated_bboxes[self.active_bbox_idx]
                    )
                    if global_bbox_id is None:
                        corrected_class_idx, object_id_to_display = self.object_id_to_display[global_bbox_id]

                        # not assigned to any cluster, (regarded as outlier)
                        if object_id_to_display is not None:
                            text_content += f'[{self.CLASS_LIST[active_bbox[0]]} {object_id_to_display}] ' #
                            text_content += f'{len(self.clusters[active_cluster_id])} detections \n'
                text_content += text_content_for_image_name
            else:
                text_content = text_content_for_image_name
            # smrc.not_used.display_text_on_image_top_middle(tmp_img, 'Press Enter to show a single object.', self.RED)
        smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)

    def display_additional_infor(self, tmp_img):
        img_path = self.IMAGE_PATH_LIST[self.active_image_index]
        image_name = img_path.split(os.path.sep)[-1]
        text_content = f'{self.active_directory}/{image_name} [total objects: {len(self.clusters)}] \n'

        if len(self.cluster_IDs_selected) > 0:
            text_content += self.text_infor_for_select_major_object()
            smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)

        elif self.fitting_mode_on:
            text_content += self.text_infor_for_curve_fitting_status()
            smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)
        else:
            active_cluster_id, active_bbox = self.get_active_cluster_id_from_active_bbox()
            # print(f'active_cluster_id = {active_cluster_id}')
            if active_cluster_id is not None:
                # text_content = 'Press Enter to show all BBox.\n'
                # text_content += 'Press Del to delete all the object_tracking result. \n'
                self.active_cluster_id = active_cluster_id

                if active_bbox is not None:
                    # print(f'active_bbox = {active_bbox}')
                    global_bbox_id = self.get_global_bbox_id_from_bbox(
                        self.IMAGE_PATH_LIST[self.active_image_index],
                        self.active_image_annotated_bboxes[self.active_bbox_idx]
                    )

                    if global_bbox_id is not None:
                        # print(f'global_bbox_id = {global_bbox_id}')
                        corrected_class_idx, object_id_to_display = self.object_id_to_display[global_bbox_id]

                        # not assigned to any cluster, (regarded as outlier)
                        if object_id_to_display is not None:
                            # print(f'object_id_to_display = {object_id_to_display}')
                            text_content += f'{self.CLASS_LIST[active_bbox[0]]} {object_id_to_display}: ' #
                            active_cluster = self.clusters[active_cluster_id]
                            text_content += f'{len(active_cluster)} detections '

                            image_start_path, image_end_path = self.cal_cluster_start_end_images(active_cluster)
                            text_content += f'({image_start_path} - {image_end_path})\n'

                            # num_holes = self.get_number_of_hole_in_cluster(self.clusters[active_cluster_id])
                            # if num_holes > 0:
                            #     text_content += f'[[ {num_holes} holes ]]\n'
                            # else:
                            #     text_content += f'\n'

                # text_content += text_content_for_image_name

            # smrc.not_used.display_text_on_image_top_middle(tmp_img, 'Press Enter to show a single object.', self.RED)
        # text_content += f'\n Total objects {len(self.clusters)}, selected objects: {len(self.cluster_IDs_selected)}'
        smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)

    def draw_bbox_with_tracking_results(self, tmp_img, bbox, global_bbox_id):
        suspicious_bbox = False
        original_class_idx = bbox[0]
        corrected_class_idx, object_id_to_display = self.object_id_to_display[global_bbox_id]

        # not assigned to any cluster, (regarded as outlier)
        if object_id_to_display is not None:
            if original_class_idx != corrected_class_idx:
                suspicious_bbox = True

                # draw here does not make much sense
                # as draw_active_bbox may change the color to yellow any way
                # self.draw_suspicious_bbox(tmp_img, bbox)
            else:
                self.draw_cluster_single_bbox_with_cluster_label(
                    tmp_img, bbox, corrected_class_idx, object_id_to_display
                )
        else:
            self.draw_annotated_bbox(tmp_img, bbox)

        return suspicious_bbox

    def draw_bboxes_from_file(self, tmp_img, ann_path):
        '''
        # load the draw bbox from file, and initialize the annotated bbox in this image
        for fast accessing (annotation -> self.active_image_annotated_bboxes)
            ann_path = labels/489402/0000.txt
            print('this ann_path =', ann_path)
        '''

        bbox_list = smrc.utils.load_bbox_from_file(ann_path)
        self.active_image_annotated_bboxes = []
        self.suspicious_bbox_id_list = []
        if len(bbox_list) == 0:
            return

        # none if no any bbox
        image_path = self.get_image_path_from_annotation_path(ann_path)

        if self.show_only_active_cluster_on:
            if len(self.frame_dets[image_path]) == 0:
                return
            for idx, global_bbox_id in enumerate(self.frame_dets[image_path]):  # 0 0.512208657048 0.455160744501 0.327413984462 0.365482233503
                image_id, bbox = self.get_image_id_and_bbox(global_bbox_id)

                if global_bbox_id in self.clusters[self.active_cluster_id] and \
                        bbox in bbox_list:
                    self.active_image_annotated_bboxes.append(bbox)

                    suspicious_bbox = self.draw_bbox_with_tracking_results(tmp_img, bbox, global_bbox_id)
                    if suspicious_bbox:
                        self.suspicious_bbox_id_list.append(
                            len(self.active_image_annotated_bboxes) - 1
                        )
        else:
            self.active_image_annotated_bboxes = bbox_list[:]
            # if there is no bbox in the first beginning when we conduct object_tracking
            if len(self.frame_dets[image_path]) == 0:
                for bbox in bbox_list:
                    self.draw_annotated_bbox(tmp_img, bbox)
            else:
                cluster_bbox_list = self.get_bbox_list_for_cluster(self.frame_dets[image_path])

                for local_bbox_id, bbox in enumerate(bbox_list):
                    if bbox not in cluster_bbox_list:
                        self.draw_annotated_bbox(tmp_img, bbox)
                    else:
                        idx = cluster_bbox_list.index(bbox)
                        global_bbox_id = self.frame_dets[image_path][idx]
                        image_id, bbox = self.get_image_id_and_bbox(global_bbox_id)

                        suspicious_bbox = self.draw_bbox_with_tracking_results(tmp_img, bbox, global_bbox_id)
                        if suspicious_bbox:
                            self.suspicious_bbox_id_list.append(local_bbox_id)
                        # original_class_idx = bbox[0]
                        # corrected_class_idx, object_id_to_display = self.object_id_to_display[global_bbox_id]
                        #
                        # # not assigned to any cluster, (regarded as outlier)
                        # if object_id_to_display is not None:
                        #     if original_class_idx != corrected_class_idx:
                        #         self.draw_suspicious_bbox(tmp_img, bbox)
                        #     else:
                        #         self.draw_cluster_single_bbox_with_cluster_label(
                        #             tmp_img, bbox, corrected_class_idx, object_id_to_display
                        #         )
                        # else:
                        #     self.draw_annotated_bbox(tmp_img, bbox)

        self.draw_bboxes_from_file_additional_operation(tmp_img)

    def draw_bboxes_from_file_additional_operation(self, tmp_img):
        for bbox in self.active_image_annotated_bboxes:
            x1, y1, x2, y2 = bbox[1:]
            # class_idx = bbox[0]
            cluster_id = self.get_cluster_id_from_bbox(bbox=bbox, image_id=self.active_image_index)

            if cluster_id is not None and (0 <= cluster_id <= len(self.clusters)):
                num_holes = self.get_number_of_hole_in_cluster(self.clusters[cluster_id])
                if num_holes > 0:
                    # class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
                    class_color = smrc.utils.RED
                    text_content = f'{num_holes} FN'
                    LINE_THICKNESS = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
                    font_scale = self.class_name_font_scale
                    margin = 10
                    text_width, text_height = cv2.getTextSize(text_content, font, font_scale, LINE_THICKNESS)[0]
                    # text_content = f'{"%.1f" % distances[0]}, ' \
                    #                f'{"%.1f" % distances[1]}, ' \
                    #                f'{"%.1f" % distances[2]}'
                    self.draw_class_name(tmp_img,
                                         (x1, y2 + text_height + margin),
                                         text_content,
                                         class_color, text_color=(0, 0, 0)
                                         )  #

    def curve_fitting_all_clusters(self):
        pass
        # if len(self.clusters) > 0:
        #     self.fill_in_missed_detection(
        #         class_label_to_fill=None,
        #         save_filled_detection_flag=True
        #     )

    def correct_class_label(self):
        bbox_to_modify = self.class_label_to_correct(self.clusters)
        print(f'correct_class_label: {len(bbox_to_modify)} box will be modified ...')
        for global_bbox_id, bbox_new in bbox_to_modify.items():
            # modify the bbox in the txt files first as we need the original box infor
            image_id, bbox_old = self.get_image_id_and_bbox(global_bbox_id)
            image_path = self.IMAGE_PATH_LIST[image_id]
            ann_path = self.get_annotation_path(image_path)

            print(f'Modify global_bbox_id {global_bbox_id} in {ann_path} '
                  f'and self.video_detected_bbox_all, '
                  f'old_bbox = {self.get_single_bbox(global_bbox_id)}, '
                  f'new_bbox = {bbox_new}')

            smrc.utils.replace_one_bbox(ann_path, bbox_old, bbox_new)

            # modify the class label in detection list
            self.modify_single_bbox(global_bbox_id, bbox_new)

    ##########################################################
    # methods for more advanced applications
    ######################################################

    def display_additional_annotation_in_active_image(self, tmp_img):
        if len(self.cluster_IDs_selected) > 0:
            for cluster_id in self.cluster_IDs_selected:
                cluster = self.clusters[cluster_id]
                image_ids = self.get_image_id_list_for_cluster(cluster)
                if self.active_image_index in image_ids:
                    bbox = self.get_single_bbox(cluster[image_ids.index(self.active_image_index)])

                    # if the bbox is deleted, then we do not show this bbox
                    # Note that, when an bbox that is already used as an known data for curve fitting
                    # is deleted, we do not
                    if bbox in self.active_image_annotated_bboxes:
                        # self.draw_special_bbox(
                        #     tmp_img, bbox=bbox,
                        #     special_color=self.RED
                        # )
                        self.draw_single_bbox(tmp_img, bbox, self.GREEN, self.GREEN, self.GREEN)

        elif self.fitting_mode_on:
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

        self.other_display_in_active_image(tmp_img)

    def other_display_in_active_image(self, tmp_img):
        pass

    def cancel_selecting_object(self):
        self.Event_CancelDrawingBBox()
        self.cluster_IDs_selected = []

    def cal_cluster_start_end_images(self, active_cluster):
        assert len(active_cluster) > 0 and len(self.IMAGE_PATH_LIST) > 0

        image_id_list_sorted = self.get_image_id_list_sorted(active_cluster)
        # print(f'active_cluster = {active_cluster}, {len(active_cluster)}')
        # print(f'image_id_list_sorted = {image_id_list_sorted}, {len(image_id_list_sorted)}')
        image_start_path = self.IMAGE_PATH_LIST[image_id_list_sorted[0]].split(os.path.sep)[-1]
        image_end_path = self.IMAGE_PATH_LIST[image_id_list_sorted[-1]].split(os.path.sep)[-1]

        return image_start_path, image_end_path

    def mouse_listener_additional(self, event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            print('middle button click,  EVENT_MBUTTONDOWN')
            self.delete_unselected_object()

    def delete_unselected_object(self):
        # clusters_remained = []
        if len(self.cluster_IDs_selected) > 0:
            # ========================= slow version
            # record all the deleted bbox
            # deleted_dict = {}
            # for idx, cluster in enumerate(self.clusters):
            #     if idx not in self.cluster_IDs_selected:
            #         deleted_dict = self.delete_one_cluster_bbox_and_generate_deleted_dict(cluster, deleted_dict)
            #         print(f'Deleting object {idx} ...')
            #     # else:
            #     #     clusters_remained.append(cluster)
            #
            # # for recovering all the deleted bbox
            # self.deleted_bbox_history.append(deleted_dict)
            # for recovering all the deleted bbox
            # self.deleted_bbox_history.append(deleted_dict)
            # ========================= slow version over

            self.Event_DeleteAllBBoxForActiveDirectory()
            for idx in self.cluster_IDs_selected:
                cluster = self.clusters[idx]
                for global_bbox_id in cluster:
                    image_path = self.IMAGE_PATH_LIST[
                        self.get_image_id(global_bbox_id)
                    ]
                    ann_path = self.get_annotation_path(image_path)
                    bbox = self.get_single_bbox(global_bbox_id)
                    smrc.utils.save_bbox_to_file_incrementally(ann_path, [bbox])

            self.cluster_IDs_selected = []
            # self.init_annotation_for_active_directory()
            self.init_annotation_for_active_directory_additional()

    def select_or_deselect_major_cluster(self):
        """
        select or deselect the bbox of a cluster by selecting the active bbox or un selecting
        the active bbox
        :return:
        """
        active_cluster_id, active_bbox = self.get_active_cluster_id_from_active_bbox()
        if active_cluster_id is not None:
            # only handle the active bbox if it is not included in any cluster
            if active_cluster_id in self.cluster_IDs_selected:
                self.cluster_IDs_selected.remove(active_cluster_id)
            else:
                # print(f'{active_cluster_id} not in self.cluster_IDs_selected {self.cluster_IDs_selected}')
                self.cluster_IDs_selected.append(active_cluster_id)

            # if len(self.curve_fitting_dict) > 0 and not self.fitting_mode_on:
            #     self.fitting_mode_on = True

    def undo_delete_unselected_object(self):
        self.undo_delete_single_tracklet()
        print(f'Recover deleted track succeed.')

    def text_infor_for_select_major_object(self):
        assert len(self.cluster_IDs_selected) > 0
        text_content = ''

        # text_content += f'Selecting major object mode: On\n'

        for cluster_id in self.cluster_IDs_selected:
            cluster = self.clusters[cluster_id]
            image_ids = self.get_image_id_list_for_cluster(cluster)
            global_bbox_id = cluster[0]
            corrected_class_idx, object_id_to_display = self.object_id_to_display[global_bbox_id]
            if object_id_to_display is not None and corrected_class_idx is not None:
                text_content += f'[{self.CLASS_LIST[corrected_class_idx]} {object_id_to_display}] '  #
                text_content += f'{len(cluster)} detections \n'
                # text_content += f'{cluster_id}, {len(cluster)} bbox {min(image_ids)}-{max(image_ids)}\n'
            else:
                text_content += f'{cluster_id}, {len(cluster)} bbox {min(image_ids)}-{max(image_ids)}\n'
        # text_content += f'selected objects: {len(self.cluster_IDs_selected)}, total objects {len(self.clusters)}\n'
        return text_content

    def Event_ShowActiveObjectTrackbar(self):
        #  -1 if not exist
        checkTrackBarPos = cv2.getTrackbarPos(self.TRACKBAR_CLUSTER, self.IMAGE_WINDOW_NAME)

        # never put this in the while loop, otherwise, error 'tuple object
        # is not callable' (probably multiple createTrackbar generated)
        if not self.show_only_active_cluster_on and len(self.clusters) > 1 and checkTrackBarPos == -1:
            cv2.createTrackbar(self.TRACKBAR_CLUSTER,
                               self.IMAGE_WINDOW_NAME, 0, len(self.clusters) - 1,
                               self.set_cluster_id)
            self.show_only_active_cluster_on = True
        # begin to view the first cluster
        # self.set_cluster_id(0)
        if self.show_only_active_cluster_on:
            if self.active_cluster_id_trackbar is not None:
                self.set_cluster_id(self.active_cluster_id_trackbar)
                cv2.setTrackbarPos(self.TRACKBAR_CLUSTER, self.IMAGE_WINDOW_NAME,
                                   self.active_cluster_id_trackbar)

    def Event_SwitchActiveObjectTrackbar(self):
        if not self.show_only_active_cluster_on:
            self.Event_ShowActiveObjectTrackbar()
        else:
            self.show_only_active_cluster_on = False

            cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
            self.init_image_window_and_mouse_listener()
            cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)

    ##########################################################
    # origin: face and license plate masking
    ######################################################

    def Event_MoveToActiveObjectStart(self):
        if self.active_cluster_id is not None and self.clusters is not None \
                and self.active_cluster_id < len(self.clusters):
            image_id_list_sorted = self.get_image_id_list_sorted(
                self.clusters[self.active_cluster_id]
            )
            self.set_image_index(image_id_list_sorted[0])
            cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)
            self.set_active_bbox_after_set_active_cluster_id()

    def Event_MoveToActiveObjectEnd(self):
        if self.active_cluster_id is not None and self.clusters is not None \
                and self.active_cluster_id < len(self.clusters):
            image_id_list_sorted = self.get_image_id_list_sorted(
                self.clusters[self.active_cluster_id]
            )
            self.set_image_index(image_id_list_sorted[-1])
            cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)
            self.set_active_bbox_after_set_active_cluster_id()

    def set_active_bbox_after_set_active_cluster_id(self):
        if self.active_cluster_id is not None and self.active_cluster_id < len(self.clusters):
            global_bbox_id_list_sorted = self.sort_cluster_based_on_image_id(
                self.clusters[self.active_cluster_id]
            )
            bbox = self.get_single_bbox(global_bbox_id_list_sorted[0])
            # print(bbox)
            self.active_bbox_to_set = bbox[:]
            # if bbox in self.active_image_annotated_bboxes:
            # self.active_bbox_idx = self.active_image_annotated_bboxes.index(bbox)

    def set_cluster_id_additional_operation(self):
        self.set_active_bbox_after_set_active_cluster_id()

    def Event_MoveToPrevOrNextObject(self, prev_or_next):
        if len(self.clusters) == 0:
            return
        elif len(self.clusters) == 1:
            self.set_cluster_id(0)
            return

        # print(f'prev_or_next = {prev_or_next}')
        if prev_or_next in ['prev', 'next']:
            if prev_or_next == 'prev':
                # print('before change value, self.active_bbox_idx =', self.active_bbox_idx)
                self.active_cluster_id_trackbar = smrc.utils.decrease_index(
                    self.active_cluster_id_trackbar,
                    len(self.clusters) - 1
                )
            else:
                self.active_cluster_id_trackbar = smrc.utils.increase_index(
                    self.active_cluster_id_trackbar,
                    len(self.clusters) - 1
                )

            self.set_cluster_id(self.active_cluster_id_trackbar)
            # print(f'self.active_cluster_id_trackbar = {self.active_cluster_id_trackbar}')

            if self.show_only_active_cluster_on:
                cv2.setTrackbarPos(self.TRACKBAR_CLUSTER, self.IMAGE_WINDOW_NAME,
                                   self.active_cluster_id_trackbar)

    def Event_MoveToPrevObject(self):
        self.Event_MoveToPrevOrNextObject('prev')

    def Event_MoveToNextObject(self):
        self.Event_MoveToPrevOrNextObject('next')

