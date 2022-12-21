import cv2
import os
import shutil
import time   # time.sleep(1000)


from .TrackletAndBBox import AnnotateTracklet
from smrc.object_tracking.cnf import IoUTracker, _MATCH_METRICS 


class AnnotateMajorObject(AnnotateTracklet):
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
        self.move_label = False
        self.curve_fitting_overlap_suppression_thd = 0.85

        self.max_dist_thd = 0.9  # max_l2_dist
        self.max_frame_gap = 5
        self.with_same_class_constraint = True

    # @staticmethod
    def object_tracking(self, video_annotation_list):
        # self.clusters, self.video_detected_bbox_all =
        my_tracker = IoUTracker()  # OptFlowPCTracker
        my_tracker.match_metric = _MATCH_METRICS['bidirectional']
        my_tracker.with_same_class_constraint = self.with_same_class_constraint

        my_tracker.offline_tracking(  # offline_tracking_class_by_class
            video_detection_list=video_annotation_list,
            max_dist_thd=self.max_dist_thd,  # max_l2_dist
            max_frame_gap=self.max_frame_gap
        )

        return my_tracker



