import cv2
import os

from .TrackletAndBBox import AnnotateTracklet
import smrc.utils


class AnnotateRelativeDistance(AnnotateTracklet):
    def __init__(self, image_dir, label_dir,
                 class_list_file, user_name=None
                 ):
        # inherit the variables and functions from AnnotationTool for visualization purpose
        # AnnotateTracklet.__init__(
        #     self, user_name=user_name, image_dir=image_dir,
        #     label_dir=label_dir, class_list_file=class_list_file
        # )
        super().__init__(
            user_name=user_name, image_dir=image_dir,
            label_dir=label_dir, class_list_file=class_list_file
        )
        self.frame_distances = []

    def annotate_active_directory(self):
        self.annotation_done_flag = False
        self.init_annotation_for_active_directory()

        result_dir = self.IMAGE_DIR + '_distance'
        smrc.utils.generate_dir_if_not_exist(
            os.path.join(result_dir, self.active_directory)
        )
        for k, image_path in enumerate(self.IMAGE_PATH_LIST):
            self.set_image_index(k)

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
            new_image_file = image_path.replace(self.IMAGE_DIR, result_dir)
            cv2.imwrite(new_image_file, tmp_img)
        # if self.WITH_QT:
        #     # if window gets closed then quit
        #     if cv2.getWindowProperty(self.IMAGE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        #         # cv2.destroyWindow(self.IMAGE_WINDOW_NAME)
        #         break

    @staticmethod
    def load_bbox_from_file(ann_path):
        print(f'ann_path={ann_path} ...')
        annotated_bbox = []
        frame_distances = []
        if os.path.isfile(ann_path):
            with open(ann_path, 'r') as old_file:
                lines = old_file.readlines()
            old_file.close()

            # print('lines = ',lines)
            for line in lines:
                result = line.split(' ')

                # the data format in line (or txt file) should be int type, 0-index.
                # we transfer them to int again even they are already in int format (just in case they are not)
                bbox = [int(result[0]), int(result[1]), int(result[2]), int(
                    result[3]), int(result[4])]
                annotated_bbox.append(bbox)
                if len(result) == 5:
                    frame_distances.append(bbox + [])
                else:
                    feature = [float(result[5]), float(result[6]), float(result[7])]
                    frame_distances.append(bbox + feature)
        return annotated_bbox, frame_distances

    def draw_bboxes_from_file(self, tmp_img, ann_path):
        '''
        # load the draw bbox from file, and initialize the annotated bbox in this image
        for fast accessing (annotation -> self.active_image_annotated_bboxes)
            ann_path = labels/489402/0000.txt
            print('this ann_path =', ann_path)
        '''

        bbox_list, self.frame_distances = self.load_bbox_from_file(ann_path)
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
        for frame_distances in self.frame_distances:
            if len(frame_distances) == 5:
                continue
            bbox, distances = frame_distances[:5], frame_distances[5:]
            x1, y1, x2, y2 = bbox[1:]

            class_idx = bbox[0]
            class_color = self.CLASS_BGR_COLORS[class_idx].tolist()
            text_content = f'{"%.1f" % distances[0]}, ' \
                           f'{"%.1f" % distances[1]}, ' \
                           f'{"%.1f" % distances[2]}'
            LINE_THICKNESS = 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
            font_scale = self.class_name_font_scale
            margin = 5
            # smrc.not_used.put_text_on_image(tmp_img, text_content,
            #                              x0 = x1, y0 = y2, dy = 50,
            # font_color =self.RED, thickness = LINE_THICKNESS, font_scale =font_scale)

            text_width, text_height = cv2.getTextSize(text_content, font, font_scale, LINE_THICKNESS)[0]
            # text_content = f'{"%.1f" % distances[0]}, ' \
            #                f'{"%.1f" % distances[1]}, ' \
            #                f'{"%.1f" % distances[2]}'
            self.draw_class_name(tmp_img,
                                 (x1, y2 + text_height + margin),
                                 f'x:{"%.1f" % distances[0]}',
                                 class_color, text_color=(0, 0, 0)
                                 )  #
            self.draw_class_name(tmp_img,
                                 (x1, y2 + text_height * 2 + margin* 2),
                                 f'y:{"%.1f" % distances[1]}', class_color, text_color=(0, 0, 0)
                                 )  #
            self.draw_class_name(tmp_img,
                                 (x1, y2 + text_height * 3 + margin * 3),
                                 f'dist:{"%.1f" % distances[2]}', class_color, text_color=(0, 0, 0)
                                 )  #
            # text_shadow_color = class_color
            # cv2.rectangle(tmp_img, (x1, y2), (x1 + text_width + margin, y2 - text_height - margin),
            #               text_shadow_color, -1)
            # cv2.putText(tmp_img, text_content, (x1, y2 - text_height -5), font, font_scale, class_color, LINE_THICKNESS,
            #             cv2.LINE_AA)

    def keyboard_listener_additional(self, pressed_key, tmp_img):
        if pressed_key in [ord('k'), ord('l'), ord('j'), ord('h'),
                             ord('i'), ord('o'), ord('n'), ord('m')]:
            T = None
            if pressed_key == ord('k'): T = ['right', -self.move_unit]
            elif pressed_key == ord('l'): T = ['right', self.move_unit]
            elif pressed_key == ord('j'): T = ['left', self.move_unit]
            elif pressed_key == ord('h'): T = ['left', -self.move_unit]
            elif pressed_key == ord('i'): T = ['top', -self.move_unit]
            elif pressed_key == ord('o'): T = ['top', self.move_unit]
            elif pressed_key == ord('n'): T = ['bottom', self.move_unit]
            elif pressed_key == ord('m'): T = ['bottom', -self.move_unit]
            self.translate_active_bbox_boundary(T)


