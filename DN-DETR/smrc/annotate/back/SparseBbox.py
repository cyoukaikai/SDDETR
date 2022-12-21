"""
This is specially designed for face and license plate annotation.
"""

import os
import cv2
import shutil
import time
import pygame
import sys
import smrc.utils

from .Bbox import AnnotateBBox


class AnnotateSparseBBox(AnnotateBBox):
    def __init__(self, image_dir, label_dir, class_list_file,
                 user_name=None, blur_bbox=False, music_on=False
                 ):
        AnnotateBBox.__init__(
            self, image_dir=image_dir, label_dir=label_dir,
            class_list_file=class_list_file, user_name=user_name,
            music_on=music_on
        )
        self.IMAGE_WINDOW_NAME = 'FaceLicensePlateMaskingTool'
        # we use line thickness of self.LINE_THICKNESS * 2 for a larger bbox
        self.bbox_area_threshold_for_thick_line = 2500
        # print(f'FaceAnnotation {self.bbox_area_threshold_for_thick_line}')

        # self.window_width = 1250  # 1000
        # self.window_height = 750  # 700
        # indicate if we blur bbox when annotating it
        self.blur_bbox = blur_bbox
        self.blur_option = 'Ellipse'  # 'Ellipse' GaussianBlur
        self.IMAGE_PATH_LIST_WITH_DETECTION_FILE = []

        self.active_image_idx_with_detection = 0
        # record the LicensePlate object_detection information of the active directory
        # key, image id, value, number of detections.

        self.detection_dict = {}
        self.num_detection = 0
        self.show_detection_trackbar_on = True
        self.show_active_image_trackbar_on = True
        self.TRACKBAR_DETECTION = 'Detection'

        # set the threshold small as if the bbox is sparse, very few overlap occur
        self.curve_fitting_overlap_suppression_thd = 0.05
        self.move_label = False
        self.generate_mask = False

    def Event_SwitchFittingMode(self, tmp_img):
        """
        # turn on or off the fitting mode
        :return:
        """
        if self.fitting_mode_on:  # fitting mode already on
            # reinitialize the curve_fitting_dict
            self.curve_fit_manually()
            self.last_fitted_bbox = self.fitted_bbox[:]
            self.curve_fitting_dict = {}
            self.fitted_bbox = []

            self.Event_RefreshDetectionDict(tmp_img, self.active_image_index)

        self.fitting_mode_on = not self.fitting_mode_on

    def Event_DeleteAllDetection(self):
        self.delete_all_detection_for_active_directory()

    def Event_UndoDeleteAllDetection(self):
        self.undo_delete_all_detection_for_active_directory()

    def MoveToPrevOrNextDetection(self, tmp_img, prev_or_next):
        if len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) < 1:
            # self.display_text('No active bbox is selected.', 1000)
            cv2.putText(tmp_img, 'No bbox any more', (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(60, 100, 75), thickness=5)
        else:

            self.move_to_pre_or_next_detection(tmp_img, prev_or_next)
            self.reset_after_image_change()

    def Event_MoveToPrevDetection(self, tmp_img):
        self.MoveToPrevOrNextDetection(tmp_img=tmp_img, prev_or_next='prev')

    def Event_MoveToNextDetection(self, tmp_img):
        self.MoveToPrevOrNextDetection(tmp_img=tmp_img, prev_or_next='next')

    def Event_DeleteAndMoveToPrevDetection(self, tmp_img):
        smrc.utils.display_text_on_image_top_middle(tmp_img, '\n\n\n\ndeleting bbox', self.BLUE)
        self.Event_DeleteActiveBBox()
        # smrc.not_used.display_text_on_image_top_middle(tmp_img, 'deleting bbox', self.BLUE)

        self.Event_MoveToPrevDetection(tmp_img)
        # # time.sleep(0.1)  # wait for 1 second.
        # if self.active_image_index not in self.detection_dict:
        #     self.Event_MoveToPrevDetection(tmp_img)

    def Event_DeleteAndMoveToNextDetection(self, tmp_img):
        smrc.utils.display_text_on_image_top_middle(tmp_img, '\n\n\n\ndeleting bbox', self.BLUE)
        self.Event_DeleteActiveBBox()

        # move to next image no matter how many bbox remains for the current image
        self.Event_MoveToNextDetection(tmp_img)
        # # time.sleep(0.1)  # wait for 1 second.
        # move to next image only when the current image has no bbox to check
        # if self.active_image_index not in self.detection_dict:
        #     self.Event_MoveToNextDetection(tmp_img)

    def Event_RefreshDetectionDict(self, tmp_img, current_image_index):
        # smrc.not_used.display_text_on_image_top_middle(tmp_img, '\n\n\n\n reloading detections', self.BLUE)
        # time.sleep(2)
        self.load_all_detection()
        smrc.utils.display_text_on_image_top_middle(tmp_img, '\n\n\n\n reloading detections', self.BLUE)
        self.reload_additional_trackbar(tmp_img)
        # move the image index
        # print(f'current_image_index = {current_image_index}')
        # print(f'self.IMAGE_PATH_LIST_WITH_DETECTION_FILE = {self.IMAGE_PATH_LIST_WITH_DETECTION_FILE}')

        self.set_image_index(current_image_index)
        cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME,
                           current_image_index)

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

        # pressed_key = cv2.waitKey(self.DELAY)
        print('pressed_key=', pressed_key)  # ('pressed_key=', -1) if no k
        # print('pressed_key & 0xFF =', pressed_key & 0xFF)
        # print('self.platform = ', self.platform)

        if pressed_key == ord(','):
            self.Event_MoveToPrevDetection(tmp_img)
        elif pressed_key == ord('.'): self.Event_MoveToNextDetection(tmp_img)
        elif pressed_key == ord('w'): self.Event_MoveToPrevClass('prev')
        elif pressed_key == ord('s'): self.Event_MoveToPrevOrNextBBox('next')

        # elif pressed_key == ord('h'): self.Event_ShowHelpInfor()
        # elif pressed_key == ord('e'):
        #     self.Event_SwitchEdgeShowing()
        elif pressed_key == ord('c'): self.Event_CopyBBox()
        elif pressed_key == ord('v'): self.Event_PasteBBox()
        elif pressed_key == ord('q'): self.Event_CancelDrawingBBox()

        elif pressed_key == ord('r'): self.Event_UndoDeleteSingleDetection()
        elif pressed_key == ord('f'): self.Event_SwitchFittingMode(tmp_img)
        # elif pressed_key == ord('j'): self.Event_MoveToPrevDetection(tmp_img)
        # elif pressed_key == ord('k'): self.Event_MoveToNextDetection(tmp_img)
        # elif pressed_key == ord('l'): self.Event_DeleteAndMoveToNextDetection(tmp_img)
        # elif pressed_key == ord('j'): self.Event_DeleteAndMoveToPrevDetection(tmp_img)

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

        if pressed_key == self.keyboard['LEFT']: self.Event_MoveBBoxWest()
        elif pressed_key == self.keyboard['UP']: self.Event_MoveBBoxNorth()
        elif pressed_key == self.keyboard['RIGHT']: self.Event_MoveBBoxEast()
        elif pressed_key == self.keyboard['DOWN']: self.Event_MoveBBoxSouth()

        elif pressed_key == ord('u'): self.Event_DeleteActiveBBox()
        # elif pressed_key == ord('y') and self.play_music_on: self.Event_MoveToNextMusic()

        # edit key
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
        # elif pressed_key == self.keyboard['F5'] or pressed_key == ord('t'):
        #     self.Event_RefreshDetectionDict(tmp_img, self.active_image_index)

        elif pressed_key == ord('a') and self.show_active_image_trackbar_on:
            self.Event_MoveToPrevImage()
        elif pressed_key == ord('d') and self.show_active_image_trackbar_on:
            self.Event_MoveToNextImage()
        # print(ord(','),ord('.') ) #print(ord(','),ord('.') ) 44 46 print(ord('>'),ord('<') )  62 60
        elif pressed_key == ord('z'):
            self.blur_bbox = not self.blur_bbox
        elif ord('1') <= pressed_key <= ord('9'):
            self.Event_ChangActiveBBoxClassLabel(pressed_key - ord('1'))  # use 1-index for annotation, but transfer it to 0-index for modifying the class id
        elif pressed_key in [ord('/'), ord('\\'), ord('?'), ord('_')]:
            if pressed_key == ord('/'):  self.Event_EnlargeActiveBBox_Horizontal(self.move_unit)
            elif pressed_key == ord('?'): self.Event_EnlargeActiveBBox_Horizontal(-self.move_unit)
            elif pressed_key == ord('\\'): self.Event_EnlargeActiveBBox_Vertical(self.move_unit)
            elif pressed_key == ord('_'): self.Event_EnlargeActiveBBox_Vertical(-self.move_unit)
        # elif pressed_key == ord('/'): self.Event_EnlargeActiveBBox(self.move_unit)
        # elif pressed_key == ord("\\"): self.Event_EnlargeActiveBBox(-self.move_unit)
        elif pressed_key & 0xFF == 13:  # Enter key is pressed
            print('Enter key is pressed.')
            self.Event_SelectFittingKnownData(tmp_img)

        elif pressed_key == self.keyboard['DELETE']:
            self.Event_DeleteAllDetection()
        elif pressed_key == self.keyboard['INSERT']:
            self.Event_UndoDeleteAllDetection()
        # elif pressed_key == self.keyboard['SHIFT']:
        #     print('SHIFT key pressed ...')
        #     # print(f'{}')
        #     # self.Event_DeleteActiveObject()
        #
        # elif pressed_key == self.keyboard['ALT']:
        #     print('ALT key pressed ...')
        #     # self.Event_UndoDeleteActiveObject()
        #     # if event.key == pygame.K_ESCAPE:
        #     #     self.annotation_done_flag = True

    # def Event_ManuallyCurveFitting(self):
    #     print('Manually curve fitting ...')

    def game_controller_listener(self, tmp_img=None):
        """
        # Possible joystick actions:
        # JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        :param tmp_img:
        :return:
        """
        # and self.game_controller_on

        # print('Enter game controller listener')

        if self.game_controller_available and self.game_controller_on:
            # if self.axis_states['rx'] != 0:  # move the bbox by axis
            #     if self.axis_states['rx'] < 0: self.Event_MoveToPrevImage()
            #     else: self.Event_MoveToNextImage()

            # fast moving image frame (no matter if any object_detection exists)
            # only for 'SHANWAN IFYOO Gamepad' (name for linux system) and (IFYOO game) for Windows system.

            if 'L2' in self.button_states and 'R2' in self.button_states:
                if self.button_states['L2'] == 1:
                    # self.Event_MoveToPrevImage()
                    self.Event_DeleteAndMoveToPrevDetection(tmp_img)
                elif self.button_states['R2'] == 1:
                    self.Event_DeleteAndMoveToNextDetection(tmp_img)
                    # self.Event_MoveToNextImage()

            if 'mode' in self.button_states and self.button_states['mode'] == 1:
                self.Event_DeleteAndMoveToNextDetection(tmp_img)

            if 'tx' in self.axis_states and 'ty' in self.axis_states:
                # print(self.axis_names)
                #  > 0.2 to increase the sensitivity (compared with == 1)
                if self.axis_states['tx'] > 0.2:
                    # self.Event_MoveToPrevImage()
                    self.Event_DeleteAndMoveToPrevDetection(tmp_img)
                elif self.axis_states['ty'] > 0.2:
                    self.Event_DeleteAndMoveToNextDetection(tmp_img)

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

                    # self.set_active_bbox_idx_for_game_controller()
                    if btn_name == 'L1':
                        self.Event_SetAnnotationDone()
                    elif btn_name == 'R1':
                        self.Event_RefreshDetectionDict(tmp_img, self.active_image_index)

                    # elif btn_name == 'B': self.Event_DeleteActiveBBox()
                    elif btn_name == 'X':
                        self.Event_UndoDeleteSingleDetection()
                    elif btn_name == 'B':
                        self.Event_DeleteAndMoveToNextDetection(tmp_img)
                    # # elif pressed_key == ord('h'): self.Event_DeleteAndMoveToPrevDetection(tmp_img)

                    elif btn_name == 'Y':
                        self.Event_DeleteAllDetection()
                    elif btn_name == 'A':
                        self.Event_UndoDeleteAllDetection()

                    elif btn_name == 'start':
                        print('start pressed, set annotation done.')
                        self.Event_FinishActiveDirectoryAnnotation()
                    elif self.js_name.find('JC-U3613M') >= 0:  # JC-U3613M game controller
                        if btn_name == 'back':
                            print('back pressed, delete active bbox ...')
                            # cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 1)
                            self.Event_DeleteActiveBBox()
                    elif self.js_name.find('IFYOO') >= 0:  # SHANWAN IFYOO Gamepad
                        if btn_name == 'select':
                            print('select pressed, delete active bbox ...')
                            # cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 1)
                            self.Event_DeleteActiveBBox()

                        # elif self.button_states['L2'] == 1:
                        #     self.Event_MoveToPrevImage()
                        # elif self.button_states['R2'] == 1:
                        #     self.Event_MoveToNextImage()

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
                            self.Event_MoveToNextDetection(tmp_img)  # 'pre' or 'next'
                        elif hat_value[0] == -1:  # leftward, xmin will decrease, hat_value[0] < 0
                            self.Event_MoveToPrevDetection(tmp_img)
                        elif hat_value[1] == 1:  # upward, ymin will decrease, but hat_value[1] > 0
                            self.Event_MoveToNextBBox()
                        elif hat_value[1] == -1:
                            # T = ['bottom', int(direction * (-hat_value[1]) * self.increase_decrease_unit)]
                            self.Event_MoveToPrevBBox()

                # JOYAXISMOTION parameter:  joy, axis, value
                if event.type == pygame.JOYAXISMOTION:
                    print("Joystick axis pressed.")
                    print('event =', event)
                    # ('event =', < Event(7-JoyAxisMotion {'joy': 0, 'value': 0.0, 'axis': 3}) >)

                    # get_axis
                    axis_id = event.axis  # get the id of the axis
                    print('self.axis_names.keys =', self.axis_names.keys())
                    print('axis_id =', axis_id)

                    axis_name = self.axis_names[axis_id]  # get the name of the axis
                    print("%s axis pressed" % (axis_name))
                    axis_value = event.value
                    print("axis value  : {}".format(axis_value))

                    if axis_name == 'x':
                        self.AxisEvent_MoveToPrevOrNextBBox(axis_name, axis_value)
                    elif axis_name == 'rx':  # move the bbox by axis
                        self.AxisEvent_TranslateActiveBBox(axis_name, 'x', axis_value)
                    elif axis_name == 'ry':
                        self.AxisEvent_TranslateActiveBBox(axis_name, 'y', axis_value)

                    self.axis_states[axis_name] = axis_value

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

        if self.active_image_index in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
            detection_idx = self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.index(self.active_image_index)
            if self.active_image_idx_with_detection != detection_idx:
                self.active_image_idx_with_detection = detection_idx
                cv2.setTrackbarPos(
                    self.TRACKBAR_DETECTION, self.IMAGE_WINDOW_NAME,
                    self.active_image_idx_with_detection
                )

    def set_detection_id(self, idx):
        self.active_image_idx_with_detection = idx
        self.active_image_index = self.IMAGE_PATH_LIST_WITH_DETECTION_FILE[self.active_image_idx_with_detection]
        cv2.setTrackbarPos(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, self.active_image_index)
        self.set_image_index(self.active_image_index)

    def display_additional_trackbar(self, tmp_img):
        checkTrackBarPos = cv2.getTrackbarPos(
            self.TRACKBAR_DETECTION, self.IMAGE_WINDOW_NAME
        )
        # never put this in the while loop, otherwise, error 'tuple object
        # is not callable' (probably multiple createTrackbar generated)
        if self.show_detection_trackbar_on and \
                len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) > 1 \
                and checkTrackBarPos == -1:
            cv2.createTrackbar(
                self.TRACKBAR_DETECTION, self.IMAGE_WINDOW_NAME,
                0, len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - 1,
                self.set_detection_id
            )
            # print(f'===============================================display_additional_trackbar')
            # print(f'self.active_image_idx_with_detection = {self.active_image_idx_with_detection}')

            # if self.active_image_idx_with_detection in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
            #     detection_idx = self.active_image_idx_with_detection
            # else:

            if self.active_image_idx_with_detection < len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE):
                detection_idx = self.active_image_idx_with_detection
            else:
                detection_idx = len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - 1

            self.set_detection_id(detection_idx)
            cv2.setTrackbarPos(self.TRACKBAR_DETECTION, self.IMAGE_WINDOW_NAME,
                               detection_idx)

        if cv2.getTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME) == -1:
            # show the annotation status bar, if annotation done is set (the annotation of this directory is done),
            # then move it to "annotation_finished"
            cv2.createTrackbar(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 0, 1, self.move_annotation_result)

    def reload_additional_trackbar(self, tmp_img):
        cv2.destroyAllWindows()
        self.init_image_window_and_mouse_listener()
        self.display_additional_trackbar(tmp_img)

    def display_additional_infor(self, tmp_img):
        img_path = self.IMAGE_PATH_LIST[self.active_image_index]
        display_image_path = smrc.utils.file_path_last_two_level(img_path)
        if self.fitting_mode_on:
            # text_content = f'[{self.active_directory}/{image_name}] '
            text_content = f'[{display_image_path} '
            text_content += self.text_infor_for_curve_fitting_status()
            smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)
        else:
            if len(self.CLASS_LIST) == 1:
                object_name = self.CLASS_LIST[0]
            else:
                object_name = 'detection'
            # if len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) < 1:
            #     text_content = f'[{self.active_directory}/] No ' + object_name + ' in this video'
            # else:
            #     text_content = f'[{self.active_directory}/] Non empty images: ' + str(self.active_image_idx_with_detection+1)
            #     text_content += '/' + str(len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE)) + '\n'
            #     text_content += object_name + ': ' + str(self.num_detection)
            #     # text_content += '/' + str(len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE))
            #
            if len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) < 1 or \
                    self.num_detection == 0:
                text_content = f'[{display_image_path}] No ' + object_name + ' in this video'
                smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.GREEN)
            else:
                img_path = self.IMAGE_PATH_LIST[self.active_image_index]
                image_name = img_path.split(os.path.sep)[-1]
                text_content = f'[{display_image_path}] Non empty images: ' \
                               + str(self.active_image_idx_with_detection + 1)
                text_content += '/' + str(len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE)) + '\n'
                text_content += 'Remaining ' + object_name + ': ' + str(self.num_detection)
                # text_content += '/' + str(len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE))

                # text_content += f'\n[{display_image_path}]'
                # text_content += f'\n[{img_path}]'
                # self.display_text_on_image(tmp_img, text_content)
            smrc.utils.display_text_on_image_top_middle(tmp_img, text_content, self.RED)

    def move_annotation_result(self, trackBar_ind):
        if not os.path.isdir(self.ANNOTATION_FINAL_DIR):
            os.makedirs(self.ANNOTATION_FINAL_DIR)

        if trackBar_ind == 1 and self.active_directory is not None:  # if ind is not 0
            source_dir = os.path.join(self.LABEL_DIR, self.active_directory)
            target_dir = os.path.join(self.ANNOTATION_FINAL_DIR, self.active_directory)

            if os.path.isdir(target_dir):  # exception check
                print('The annotation directory already exist, please check.')
                smrc.utils.display_text_on_image_top_middle('The annotation directory already exist, please check.',
                                                            3000)
                time.sleep(2)  # wait for 1 second.
                cv2.setTrackbarPos(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 0)
                # sys.exit(0)
            else:  # if no exception, then handle the event
                if self.move_label:
                    shutil.move(source_dir, self.ANNOTATION_FINAL_DIR)
                if self.generate_mask:
                    self.auto_masking_active_directory()

                # self.display_text('[Moving annotation result: done].', 1000)
                self.annotation_done_flag = True
                # update the directory list file
                self.delete_active_directory()
                # self.active_directory = None
                time.sleep(0.5)  # wait for 1 second.

    def move_to_pre_or_next_detection(self, tmp_img, prev_or_next):
        """

        :param tmp_img:
        :param prev_or_next:
        :return:
        """
        if prev_or_next not in ('prev', 'next'):
            print('Please input only "prev" or "next".')
            sys.exit(0)

        if self.IMAGE_PATH_LIST_WITH_DETECTION_FILE is None or \
                len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) == 0:

            if len(self.CLASS_LIST) == 1:
                object_name = self.CLASS_LIST[0]
            else:
                object_name = 'bbox'

            cv2.putText(tmp_img, 'No ' + object_name + ' any more', (300, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color=(60, 100, 75),
                        thickness=5
                        )
            return

        # print('-==============================-')
        # print(f'self.active_image_index = {self.active_image_index}')
        # print(f'In dict, move to next object_detection')

        if self.active_image_index in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
            # print('self.active_image_index in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE ')
            if prev_or_next == 'prev':
                self.active_image_idx_with_detection = smrc.utils.decrease_index(
                    self.active_image_idx_with_detection,
                    len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - 1
                )
            elif prev_or_next == 'next':
                self.active_image_idx_with_detection = smrc.utils.increase_index(
                    self.active_image_idx_with_detection,
                    len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - 1
                )

        # else:
        #     print('not in dict, move to current object_detection id')
        cv2.setTrackbarPos(
            self.TRACKBAR_DETECTION, self.IMAGE_WINDOW_NAME,
            self.active_image_idx_with_detection
        )
        self.set_detection_id(self.active_image_idx_with_detection)

    def load_active_directory_additional_operation(self):
        self.load_all_detection()

    # def keyboard_lister
    def init_image_window_and_mouse_listener(self):
        self.init_window_size_font_size()

        # create window
        cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # cv2.WINDOW_KEEPRATIO cv2.WINDOW_AUTOSIZE
        cv2.resizeWindow(self.IMAGE_WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(self.IMAGE_WINDOW_NAME, self.mouse_listener)

        # show the image index bar, self.set_image_index is defined in AnnotationTool()
        cv2.createTrackbar(self.TRACKBAR_IMG, self.IMAGE_WINDOW_NAME, 0, self.LAST_IMAGE_INDEX, self.set_image_index)

        # # show the class index bar only if we have more than one class
        # if self.LAST_CLASS_INDEX != 0:
        #     cv2.createTrackbar(self.TRACKBAR_CLASS, self.IMAGE_WINDOW_NAME, 0, self.LAST_CLASS_INDEX,
        #                        self.set_class_index)

        # # show the annotation status bar, if annotation done is set (the annotation of this directory is done),
        # # then move it to "annotation_finished"
        # cv2.createTrackbar(self.TRACKBAR_ANNOTATION_DONE, self.IMAGE_WINDOW_NAME, 0, 1, self.move_annotation_result)

        if self.active_image_index is not None:
            self.set_image_index(self.active_image_index)
        else:
            self.set_image_index(0)
        # self.display_text('Welcome!\n Press [h] for help.', 2000)

    def init_annotation_for_active_directory_additional(self):
        self.active_image_idx_with_detection = 0

        # set the object_detection id to the first image
        if len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) > 0:
            # do not use self.set_detection_id(0)
            # as if we refresh the object_detection list, it will always go to 0
            if self.active_image_idx_with_detection is not None:
                self.set_detection_id(self.active_image_idx_with_detection)
            else:
                self.set_detection_id(0)

    def load_all_detection(self):
        # the images that are detected to have some objects
        self.IMAGE_PATH_LIST_WITH_DETECTION_FILE = []
        # self.active_image_idx_with_detection = 0
        self.detection_dict = {}
        self.num_detection = 0
        # create empty annotation files for each image, if it doesn't exist already
        for image_id, img_path in enumerate(self.IMAGE_PATH_LIST):
            ann_path = self.get_annotation_path(img_path)
            # print('ann_path = ', ann_path)

            if not os.path.isfile(ann_path):
                num_bbox = 0
                # open(ann_path, 'a').close()  # no need to generate an empty txt file
            else:
                with open(ann_path) as f:
                    num_bbox = len(list(smrc.utils.non_blank_lines(f)))
                    # sorted(list(self.non_blank_lines(f_directory_list)), key=self.natural_sort_key)
                f.close()  # close the file
            # print('num_bbox = %d for %s' %(num_bbox, ann_path) )
            if num_bbox > 0:
                self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.append(image_id)
                self.detection_dict[image_id] = num_bbox
                self.num_detection += num_bbox
        self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.sort()

    def draw_bboxes_from_file(self, tmp_img, ann_path):
        self.active_image_annotated_bboxes = []  # initialize the image_annotated_bboxes

        # ann_path = output/1000videos/489402/0000.txt
        # print('this ann_path =', ann_path)
        if os.path.isfile(ann_path):
            # edit YOLO file
            with open(ann_path, 'r') as old_file:
                lines = old_file.readlines()
            old_file.close()

            # print('lines = ',lines)
            refined_bboxes = []
            for line in lines:  # 0 0.512208657048 0.455160744501 0.327413984462 0.365482233503
                result = line.split(' ')

                # the data format in line (or txt file) should be int type, 0-index.
                # we transfer them to int again even they are already in int format (just in case they are not)
                bbox = [int(result[0]), int(result[1]), int(result[2]), int(
                    result[3]), int(result[4])]

                self.active_image_annotated_bboxes.append(bbox)
                self.draw_annotated_bbox(tmp_img, bbox)
                refined_bboxes.append([int(result[1]), int(result[2]), int(
                    result[3]), int(result[4])])

            if self.blur_bbox:
                self.overlay_bounding_boxes(
                    raw_img=tmp_img, refined_bboxes=refined_bboxes, lw=3,
                    show_bbox=False, blur_option=self.blur_option,
                    patch_color=self.RED
                )

    def overlay_bounding_boxes(
            self, raw_img, refined_bboxes, lw, show_bbox,
            blur_option, patch_color
    ):
        """Overlay bounding boxes of face on images.
          Args:
            raw_img:
              A target image.
            refined_bboxes:
              Bounding boxes of detected faces.
            lw:
              Line width of bounding boxes. If zero specified,
              this is determined based on confidence of each object_detection.
          Returns:
            None.
        """
        if patch_color is None:
            patch_color = self.RED

        # Overlay bounding boxes on an image with the color based on the confidence.
        for r in refined_bboxes:
            _lw = lw
            _r = [int(x) for x in r[0:4]]
            bw, bh = r[2] - r[0] + 1, r[3] - r[1] + 1
            if blur_option == "Ellipse":
                xc, yx, ew, eh = int(r[0] + bw / 2), int(r[1] + bh / 2), int(bw / 2), int(bh / 2)
                # print(rect_color) #[252, 245, 33] [253, 239, 45]

                # ellipse_color = [211,211,211]
                ellipse_color = patch_color
                cv2.ellipse(raw_img, (xc, yx), (ew, eh), 0, 0, 360, ellipse_color, -1)  # _lw
                # mg = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
            else:  # (blur_option == "GaussianBlur"):
                xmin, ymin, xmax, ymax = _r[0], _r[1], r[2], r[3]
                if ymax - ymin <= 0 or xmax - xmin <= 0:
                    continue
                else:
                    sub_face = raw_img[ymin:ymax, xmin:xmax]
                    # apply a gaussian blur on this new recangle image
                    sub_face = cv2.GaussianBlur(sub_face, (13, 13), 3)  # (23,23),30
                    # merge this blurry rectangle to our final image
                    raw_img[ymin:ymax, xmin:xmax] = sub_face
                    # face_file_name = "./face_" + str(y) + ".jpg"
                    # cv2.imwrite(face_file_name, sub_face)
                    ##img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

                    if show_bbox:
                        # print(rect_color) #[252, 245, 33] [253, 239, 45] self.ACTIVE_BBOX_COLOR
                        rect_color = smrc.utils.complement_bgr(self.ACTIVE_BBOX_COLOR)
                        cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)
                        # cv2.line(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)

    def auto_masking_active_directory(self):
        result_root_dir = os.path.join(self.ANNOTATION_FINAL_DIR, 'masking_results')
        smrc.utils.generate_dir_if_not_exist(
            os.path.join(result_root_dir, self.active_directory)
        )

        # automatically mask the detected area
        # file_list = smrc.not_used.get_file_list_in_directory(
        #     os.path.join(self.LABEL_DIR,   self.active_directory)
        # )

        # create empty annotation files for each image, if it doesn't exist already
        for img_path in self.IMAGE_PATH_LIST:
            raw_img = cv2.imread(img_path)
            ann_path = self.get_annotation_path(img_path)

            refined_bboxes = []  # initialize the image_annotated_bboxes

            # ann_path = output/1000videos/489402/0000.txt
            # print('this ann_path =', ann_path)
            if os.path.isfile(ann_path):
                # edit YOLO file
                with open(ann_path, 'r') as old_file:
                    lines = old_file.readlines()
                old_file.close()

                # print('lines = ',lines)
                for line in lines:  # 0 0.512208657048 0.455160744501 0.327413984462 0.365482233503
                    result = line.split(' ')

                    # do not need the class idx
                    bbox = [int(result[1]), int(result[2]), int(
                        result[3]), int(result[4])]

                    refined_bboxes.append(bbox)

            self.overlay_bounding_boxes(
                raw_img=raw_img, refined_bboxes=refined_bboxes, lw=3,
                show_bbox=False, blur_option=self.blur_option,
                patch_color=[211, 211, 211]
            )

            new_path = img_path.replace(self.IMAGE_DIR, result_root_dir, 1)
            print('Generating masked images ' + new_path)

            # save image with bounding boxes
            # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_path, raw_img)

    def delete_active_bbox_additional(self):

        self.num_detection -= 1
        num_bbox = self.detection_dict[self.active_image_index]
        # print('num_bbox = ', num_bbox)
        if num_bbox > 1:
            self.detection_dict[self.active_image_index] = num_bbox - 1
        else:
            try:
                # print('before delete a object_detection', len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) )
                del self.detection_dict[self.active_image_index]
                # update the IMAGE_PATH_LIST_WITH_DETECTION_FILE
                # index_to_delete = self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.index(self.active_image_index)
                # del self.IMAGE_PATH_LIST_WITH_DETECTION_FILE[index_to_delete]
                # # print('after delete a object_detection', len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) )
                #
                # # # print(f'self.active_image_idx_with_detection = {self.active_image_idx_with_detection}' )
                # # if self.active_image_idx_with_detection != 0:
                # #     self.active_image_idx_with_detection = smrc.not_used.decrease_index(
                # #         self.active_image_idx_with_detection,
                # #         len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - 1
                # #     )

            except KeyError:
                print("Key %d not found, delete failed" % (self.active_image_index,))

            # cv2.setTrackbarPos(self.TRACKBAR_DETECTION, self.IMAGE_WINDOW_NAME, \
            #                    self.active_image_idx_with_detection)
            # self.set_detection_id(self.active_image_idx_with_detection)

        # print(f'after modification, self.active_image_idx_with_detection = {self.active_image_idx_with_detection}' )

        # if len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) < 1:
        #     text_content = f'[{self.active_directory}/] No ' + object_name + ' in this video'
        # else:
        #     text_content = f'[{self.active_directory}/] Non empty images: ' + str(self.active_image_idx_with_detection+1)
        #     text_content += '/' + str(len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE)) + '\n'
        #     text_content += object_name + ': ' + str(self.num_detection)
        #     # text_content += '/' + str(len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE))
        # # img_path = self.IMAGE_PATH_LIST[self.active_image_index]
        # # text_content += '\nShowing {}'.format(img_path)
        # self.display_text_on_image(tmp_img, text_content)

    def add_bbox_additional(self):
        """
        additional operation after adding a bbox
        :return:
        """
        self.num_detection += 1
        if self.active_image_index in self.detection_dict:
            self.detection_dict[self.active_image_index] += 1
        else:
            self.detection_dict[self.active_image_index] = 1
            #
            if self.active_image_index not in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
                self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.append(self.active_image_index)
        self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.sort()

    def delete_next_n_detection(self, N):
        if N == 0: return

        deleted_dict = {}
        remaining_detection_from_here = len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - \
                                        self.active_image_idx_with_detection
        if N > remaining_detection_from_here:
            N = remaining_detection_from_here

        for idx in range(N):
            # for img_id in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
            img_id = self.active_image_idx_with_detection + idx
            image_path = self.IMAGE_PATH_LIST[img_id]
            ann_path = self.get_annotation_path(image_path)
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            # assert len(bbox_list) > 0, 'bbox_list should have at least one bbox'

            if len(bbox_list) > 0:
                deleted_dict[ann_path] = bbox_list
                smrc.utils.empty_annotation_file(ann_path)

            # print('before delete a object_detection', len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) )
            del self.detection_dict[self.active_image_index]
            # update the IMAGE_PATH_LIST_WITH_DETECTION_FILE
            index_to_delete = self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.index(img_id)
            del self.IMAGE_PATH_LIST_WITH_DETECTION_FILE[index_to_delete]
            # print('after delete a object_detection', len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) )

        self.deleted_bbox_history.append(deleted_dict)
        self.num_detection -= N

    def undo_delete_next_n_detection(self, N):
        if N == 0: return

        # self.load_all_detection()
        remaining_detection_from_here = len(self.IMAGE_PATH_LIST_WITH_DETECTION_FILE) - \
                                        self.active_image_idx_with_detection
        if N > remaining_detection_from_here:
            N = remaining_detection_from_here

        self.undo_delete_single_tracklet()
        print(f'Recover deleted {remaining_detection_from_here} bbox succeed.')
        self.load_all_detection()

    def delete_all_detection_for_active_directory(self):
        """
        delete all the annotated bbox
        :return:
        """
        deleted_dict = {}
        for img_id in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
            image_path = self.IMAGE_PATH_LIST[img_id]
            ann_path = self.get_annotation_path(image_path)
            bbox_list = smrc.utils.load_bbox_from_file(ann_path)
            # assert len(bbox_list) > 0, 'bbox_list should have at least one bbox'

            # if we do not update the self.IMAGE_PATH_LIST_WITH_DETECTION_FILE
            if len(bbox_list) > 0:
                deleted_dict[ann_path] = bbox_list
                smrc.utils.empty_annotation_file(ann_path)

        self.deleted_bbox_history.append(deleted_dict)
        self.IMAGE_PATH_LIST_WITH_DETECTION_FILE = []
        self.detection_dict = {}
        self.num_detection = 0

    def undo_delete_all_detection_for_active_directory(self):
        self.undo_delete_single_tracklet()
        print(f'Recover bbox succeed.')
        self.load_all_detection()

    def undo_delete_single_bbox_additional_operation(self):
        self.num_detection += 1
        if self.active_image_index in self.detection_dict:
            self.detection_dict[self.active_image_index] += 1
        else:
            self.detection_dict[self.active_image_index] = 1
            #
            if self.active_image_index not in self.IMAGE_PATH_LIST_WITH_DETECTION_FILE:
                self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.append(self.active_image_index)
        self.IMAGE_PATH_LIST_WITH_DETECTION_FILE.sort()
