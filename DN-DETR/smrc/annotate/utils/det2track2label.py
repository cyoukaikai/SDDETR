import os
import cv2
from tqdm import tqdm
from smrc.utils.det.yolo_json import load_yolo_json_det
from smrc.utils.det.detection_process import suppress_noise_det
from smrc.dataset import Dataset

# from smrc.object_tracking.ds import DeepSortOnline
import smrc.utils
import smrc.object_tracking.cnf as cnf
from smrc.object_tracking.cnf import Epsilon
# from smrc.object_tracking.appearance_cnf import AppearanceCNFTracker
# from smrc.object_tracking.ahc_ete.appearance_ahc import AppearanceAHC


class Det2Track2Label(Dataset):
    def __init__(
            self, data_root_dir, image_dir,
            label_dir=None,  # where the result are saved
            yolov3_json_dir=None,
            yolov4_json_dir=None):
        super().__init__(data_root_dir=data_root_dir)

        self.image_dir = image_dir
        smrc.utils.assert_dir_exist(self.image_dir)

        if label_dir is None:
            self.label_dir = os.path.join(self.image_dir, 'det_to_track_to_labels')
        else:
            self.label_dir = label_dir
        smrc.utils.generate_dir_if_not_exist(self.label_dir)
        smrc.utils.assert_dir_exist(self.label_dir)

        assert (yolov3_json_dir is None and yolov4_json_dir is not None) or (
                yolov3_json_dir is not None and yolov4_json_dir is None
        )
        if yolov3_json_dir is not None:
            self.json_dir = yolov3_json_dir
            self.json_format = 'yolov3'
        elif yolov4_json_dir is not None:
            self.json_dir = yolov4_json_dir
            self.json_format = 'yolov4'
        # self.yolov3_json_dir = yolov3_json_dir
        # self.yolov4_json_dir = yolov4_json_dir

    # def object_tracking(self, video_detection_list, **kwargs):
    #     # initialize the tracker
    #     my_tracker = DeepSortOnline()
    #     # conduct online tracking and save the result
    #     tracking_result_dir = self.label_dir
    #     if 'dir_name' in kwargs and kwargs['dir_name'] is not None:
    #         final_result_dir = os.path.join(tracking_result_dir, kwargs['dir_name'])
    #         my_tracker.online_tracking(
    #             video_detection_list=video_detection_list,
    #             resulting_tracks_image_dir=final_result_dir,
    #             resulting_tracks_txt_file=final_result_dir + '.txt'
    #         )
    #     else:
    #         my_tracker.online_tracking(
    #             video_detection_list=video_detection_list
    #         )
    #
    #     return my_tracker

    def object_tracking(self, video_detection_list, **kwargs):
        # initialize the tracker
        my_tracker = cnf.IoUTracker()  # match_metric='ahc' OptFlowPCTracker
        # conduct online tracking and save the result
        tracking_result_dir = self.label_dir
        # distance not similarity
        max_dist_thd_list = [1-Epsilon] * 5 + [0.5] * 5
        frame_gap_list = list(range(1, len(max_dist_thd_list)+1))

        my_tracker.offline_tracking_class_by_class(
            video_detection_list=video_detection_list,
            # max_dist_thd=100,  # max_l2_dist  max_frame_gap=10
            # max_frame_gap=5,
            max_dist_thd_list=max_dist_thd_list,
            frame_gap_list=frame_gap_list,
            num_pixel_bbox_to_extend=30  # extend e
        )

        # if 'dir_name' in kwargs and kwargs['dir_name'] is not None:
        #     final_result_dir = os.path.join(tracking_result_dir, kwargs['dir_name'])
        #     my_tracker.online_tracking(
        #         video_detection_list=video_detection_list,
        #         resulting_tracks_image_dir=final_result_dir,
        #         resulting_tracks_txt_file=final_result_dir + '.txt'
        #     )
        # else:
        #     my_tracker.online_tracking(
        #         video_detection_list=video_detection_list
        #     )

        return my_tracker

    def to_label(self,  # image_dir, json_dir,
                 nms_thd=0.1,  # non maximum suppression threshold
                 score_thd=0.2, noise_det_json_dir=None,
                 dir_list=None
                 ):
        image_dir, json_dir = self.image_dir, self.json_dir
        assert os.path.isdir(json_dir) and os.path.isdir(image_dir)
        # dir_list = ['911962']
        # load the file names of the json detection result
        json_files = smrc.utils.get_json_file_list(json_dir, only_local_name=True)
        if dir_list is not None and isinstance(dir_list, list) and len(dir_list) > 0:
            json_files = [x for x in json_files if smrc.utils.extract_smrc_json_dir(x) in dir_list]
            json_files = sorted(json_files, key=lambda x: dir_list.index(smrc.utils.extract_smrc_json_dir(x)))

        assert len(json_files) > 0

        # # set where to save the tracking result
        # if tracking_result_dir is None:
        #     tracking_result_dir = os.path.join(self.data_root_dir, 'tracking_results')
        #     smrc.not_used.generate_dir_if_not_exist(tracking_result_dir)

        #####################################################
        # conduct tracking for each directory
        #####################################################
        # exclude_dir_name = ['926857', '927050', '927477', '927707']
        pbar = tqdm(enumerate(json_files))
        for k, json_file in pbar:  #
            pbar.set_description(f'Processing {json_file} [{k}/{len(json_files) - 1}] ...')
            test_json_file = os.path.join(self.json_dir, json_file)
            dir_name = smrc.utils.det.yolo_json.extract_smrc_json_dir(
                json_file
            )
            # if dir_name not in exclude_dir_name:
            #     continue

            image_list = smrc.utils.get_file_list_recursively(
                os.path.join(image_dir, dir_name)
            )

            # load detections
            # if self.json_format == 'yolov3':
            #     video_detection_list = smrc.not_used.det.yolo_json.json_det_to_tracking_format(
            #         json_file=test_json_file,
            #         test_image_list=image_list,
            #         nms_thd=nms_thd,  # non maximum suppression threshold
            #         score_thd=score_thd,  # minimum detection score
            #     )
            # detection_dict = load_yolo_json_det(
            #     json_detection_file=test_json_file,
            #     score_thd=score_thd, nms_thd=nms_thd, det_format=self.json_format
            # )  # score_thd= 0.0
            detection_dict = load_yolo_json_det(
                json_detection_file=test_json_file,
                score_thd=0.05, nms_thd=nms_thd, det_format=self.json_format
            )  # score_thd= 0.0

            if noise_det_json_dir is not None:
                noise_json_file = os.path.join(noise_det_json_dir, json_file)
                noise_det_dict = load_yolo_json_det(
                    json_detection_file=noise_json_file,
                    score_thd=0.05, nms_thd=0.1, det_format=self.json_format
                )

                detection_dict, _ = suppress_noise_det(detection_dict, noise_det_dict, nms_thd=0.1)
            det_num = smrc.utils.det.detection_process.count_det_num(detection_dict)
            if det_num == 0:
                continue

            video_detection_list = smrc.utils.det.yolo_json.det_dict_to_smrc_tracking_format(
                detection_dict=detection_dict, image_list=image_list
            )

            print(f'Processing {dir_name} [{k + 1}/{len(json_files)}], '
                  f'{len(image_list)} images ...')

            my_tracker = self.object_tracking(
                video_detection_list=video_detection_list,
                dir_name=dir_name
            )
            # my_tracker.visualize_tracking_results()
            # clusters = my_tracker.clusters

            # post-process the result
            if len(my_tracker.clusters) > 0:
                # change the class labels for majority voting
                # correct label, the data in self.video_detected_bbox_all is modified
                # remove the short tracks
                my_tracker.delete_cluster_with_length_thd(
                    my_tracker.clusters, track_min_length_thd=2
                )
            if len(my_tracker.clusters) > 0:
                my_tracker.delete_low_score_track(
                    my_tracker.clusters, score_thd=score_thd, criteria='mean'
                )
            if len(my_tracker.clusters) > 0:
                # my_tracker.correct_class_label_by_majority_voting(my_tracker.clusters)

                # fill in the holes for long tracks
                # curve fitting modifies self.clusters so we need self.clusters assigned
                # , num_missed_det_to_fill_thd=3 , class_label_to_fill=2  # to make the filled bbox more visible
                my_tracker.fill_in_missed_detection(
                    my_tracker.clusters
                )

            # remove detections with strange sizes
            # tracking_result_dir = os.path.join(self.label_dir, dir_name)
            if len(my_tracker.clusters) > 0:
                smrc.utils.generate_dir_if_not_exist(os.path.join(self.label_dir, dir_name))
                my_tracker.save_tracking_results_as_separate_txt_file(
                    self.label_dir
                )


# class FaceLPDet2Track2Label(Det2Track2Label):
#     def __init__(
#             self, data_root_dir, image_dir,
#             label_dir=None,   # where the result are saved
#             noise_det_json_dir=None,
#             yolov3_json_dir=None,
#             yolov4_json_dir=None):
#         super().__init__(
#             data_root_dir=data_root_dir, image_dir=image_dir,
#             label_dir=label_dir,  # where the result are saved
#             yolov3_json_dir=yolov3_json_dir,
#             yolov4_json_dir=yolov4_json_dir)