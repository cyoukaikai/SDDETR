import os
import random
import cv2
import shutil
from tqdm import tqdm
import smrc.utils
from smrc.utils.det.yolo_json import json_yolov4_to_yolov3


class ThreeLevelDir:
    def __init__(self, data_dir, valid_dir_list=None):
        self.data_dir_ = data_dir
        self.dir_list_ = None
        self.valid_dir_list_ = valid_dir_list

        # check if the given valid_dir_list is indeed valid
        if self.valid_dir_list_ is not None:
            assert len(self.valid_dir_list_) > 0
            for dir_name in self.valid_dir_list_:
                smrc.utils.assert_dir_exist(
                    self.get_dir_abs_path(dir_name)
                )

    def get_valid_dir_list(self):
        if self.valid_dir_list_ is not None:
            return self.valid_dir_list_
        else:
            return self.get_complete_dir_list()

    def get_complete_dir_list(self):
        if self.dir_list_ is None:
            self.dir_list_ = smrc.utils.get_dir_list_in_directory(
                self.data_dir_
            )
        return self.dir_list_

    def get_dir_abs_path(self, dir_name):
        return os.path.join(self.data_dir_, dir_name)

    def remove_empty_files_and_dirs(self):
        # if dir_list is None:  # , dir_list=None
        #     dir_list = smrc.not_used.get_dir_list_in_directory(
        #         self.data_root_dir
        #     )
        smrc.utils.delete_empty_file(label_root_dir=self.data_dir_)
        smrc.utils.delete_empty_dirs(self.data_dir_)


class ImageData(ThreeLevelDir):
    def __init__(self, image_root_dir, valid_dir_list=None):
        super().__init__(data_dir=image_root_dir, valid_dir_list=valid_dir_list)

    def get_image_sequence_width(self, dir_name):
        dir_path = self.get_dir_abs_path(dir_name)
        img_h, img_w = smrc.utils.get_image_size_for_image_sequence(
            image_sequence_dir=dir_path
        )
        return img_w

    def sort_dir_list_by_img_w(self, dir_list=None):
        if dir_list is None:
            dir_list = self.get_complete_dir_list().copy()

        dir_list = sorted(dir_list,
               key=lambda dir_name: self.get_image_sequence_width(dir_name),
               reverse=False
        )
        return dir_list

    def sort_dir_list_by_length(self, dir_list=None):
        if dir_list is None:
            dir_list = self.get_complete_dir_list().copy()

        sorted(dir_list,
               key=lambda dir_name: len(
                   smrc.utils.get_file_list_in_directory(self.get_dir_abs_path(dir_name))),
               reverse=False
        )
        return dir_list

    def get_video_inf(self, result_file_name=None, dir_list=None, sort_video_by_img_w=True):
        video_inf = []

        # # sort the video by img_w in default
        if sort_video_by_img_w:
            tmp_dir_list = self.sort_dir_list_by_img_w()
        else:
            tmp_dir_list = self.get_valid_dir_list()

        if dir_list is None:
            dir_list = tmp_dir_list
        else:
            dir_list = [dir_name for dir_name in tmp_dir_list
                        if dir_name in dir_list]

        # assert len(dir_list) > 0
        pbar = tqdm(enumerate(dir_list))
        for dir_idx, dir_name in pbar:
            pbar.set_description(
                f'get_video_inf: To load images from {dir_name} ({dir_idx}/{len(dir_list)}) ...')
            dir_path = self.get_dir_abs_path(dir_name)
            num_img = len(smrc.utils.get_file_list_in_directory(dir_path))
            if num_img > 0:
                img_h, img_w = smrc.utils.get_image_size_for_image_sequence(
                    image_sequence_dir=dir_path
                )
            else:
                img_h, img_w = 0, 0
            video_inf.append([dir_name, num_img, img_h, img_w])

        if result_file_name is not None:
            filed_name_list = ['Video_ID', 'Num_of_Image', 'Image_Height', 'Image_Width']
            smrc.utils.save_excel_file(
                list_2d=video_inf,
                result_file_name=result_file_name,
                field_name_list=filed_name_list
            )
        return video_inf

    def generate_and_save_video_inf(
            self, result_excel_file=None, dir_list=None):
        if result_excel_file is None:
            result_excel_file = self.data_dir_ + '.xlsx'
        self.get_video_inf(result_file_name=result_excel_file, dir_list=dir_list)


class SMRCImageData(ImageData):
    def __init__(self, image_root_dir, valid_dir_list=None):
        """
        :param image_root_dir:
        :param valid_dir_list: only dir_list in this list will be considered
        for operation
        """
        super().__init__(image_root_dir, valid_dir_list=valid_dir_list)

    def get_random_dir_list(self, video_type=None, random_seed=None):
        dir_list = []
        if video_type is None:
            dir_list = self.get_valid_dir_list()
            # self.get_complete_dir_list()
        elif video_type == 'denso':
            dir_list = self.get_denso_videos()
        elif video_type == 'horiba':
            dir_list = self.get_horiba_videos()

        if random_seed is not None:
            random.seed(random_seed)

        random.shuffle(dir_list)
        return dir_list

    def train_test_split_video(self, num_test, video_type=None, random_seed=None):
        dir_list = self.get_random_dir_list(
            video_type=video_type, random_seed=random_seed
        )
        assert num_test < len(dir_list)
        test_videos = dir_list[:num_test]
        train_videos = dir_list[num_test:]
        return train_videos, test_videos

    def split_denso_horiba_test_data(self, num_test_per_camera_type, random_seed=None):
        train_videos1, test_videos1 = self.train_test_split_video(
            num_test=num_test_per_camera_type,  video_type='denso', random_seed=random_seed
        )

        train_videos2, test_videos2 = self.train_test_split_video(
            num_test=num_test_per_camera_type, video_type='horiba', random_seed=random_seed
        )
        train_videos = train_videos1 + train_videos2
        test_videos = test_videos1 + test_videos2
        assert len(test_videos) == 2 * num_test_per_camera_type

        # assert len(train_videos) == len(set(train_videos))
        # assert len(test_videos) == len(set(test_videos))
        # for dir_name in train_videos1 + test_videos1:
        #     assert self.get_image_sequence_width(dir_name) > 1000
        #
        # for dir_name in train_videos2 + test_videos2:
        #     assert self.get_image_sequence_width(dir_name) < 1000

        return train_videos, test_videos

    def get_denso_videos(self):
        dir_list = self.get_valid_dir_list()
        # self.get_complete_dir_list()
        img_w_list = [self.get_image_sequence_width(dir_name)
                      for dir_name in dir_list]
        video_valid = [dir_list[k] for k in range(len(img_w_list))
                       if img_w_list[k] > 1000]
        return video_valid

    def get_horiba_videos(self):
        """
        The image size of the Horiba camera, height * width = 400 * 640
        or 480 * 640.
        :return:
        """
        dir_list = self.get_valid_dir_list()  # self.get_complete_dir_list()
        img_w_list = [self.get_image_sequence_width(dir_name)
                      for dir_name in dir_list]
        video_valid = [dir_list[k] for k in range(len(img_w_list))
                       if img_w_list[k] < 1000]
        return video_valid

    def extract_first_images(self, result_file_name=None):
        dir_list = self.get_valid_dir_list()

        resulting_img_list = []
        pbar = tqdm(enumerate(dir_list))
        for dir_idx, dir_name in pbar:
            pbar.set_description(
                f'extract_first_images: load images from {dir_name} ({dir_idx}/{len(dir_list)}) ...')
            dir_path = self.get_dir_abs_path(dir_name)

            image_path_list = smrc.utils.get_file_list_in_directory(dir_path, ext_str='.jpg')

            if len(image_path_list) > 0:
                if image_path_list[0].find('0000.jpg') < 0:
                    print(f'| Attention, {image_path_list[0]} does not include string 0000.jpg')
                else:
                    resulting_img_list.append(image_path_list[0])
            else:
                print(f'| Attention, {dir_path} has no image in the directory')

        if result_file_name is not None:
            smrc.utils.save_1d_list_to_file(file_path=result_file_name, list_to_save=resulting_img_list)


class LabelData(ThreeLevelDir):
    """
    Only the label directory.
    """

    def __init__(self, label_root_dir, valid_dir_list=None):
        super().__init__(data_dir=label_root_dir, valid_dir_list=valid_dir_list)
        self.data_root_dir = label_root_dir
        self.dir_list_ = None

    def remove_class(self, class_list_to_remove, dir_list=None):
        if dir_list is None:
            dir_list = self.get_valid_dir_list()
        pbar = tqdm(enumerate(dir_list))
        print(f'Remove class {class_list_to_remove} in {self.data_root_dir} ')
        count = 0
        for dir_idx, dir_name in pbar:

            ann_path_list = smrc.utils.get_file_list_in_directory(
                os.path.join(self.data_root_dir, dir_name), ext_str='.txt'
            )
            for ann_path in ann_path_list:
                bbox_list = smrc.utils.load_bbox_from_file(ann_path)
                # print(f'bbox_list = {bbox_list}')

                bbox_list_new = [bbox for bbox in bbox_list if bbox[0] not in class_list_to_remove]
                if len(bbox_list) - len(bbox_list_new) > 0:
                    # print(f'len(bbox_list) = {len(bbox_list)}, len(bbox_list_new) = {len(bbox_list_new)}')
                    smrc.utils.save_bbox_to_file(ann_path=ann_path, bbox_list=bbox_list_new)
                    count += len(bbox_list) - len(bbox_list_new)
            pbar.set_description(
                f'| Processing {dir_name} done, {count} bbox removed [{dir_idx + 1}/{len(dir_list)} ]')

        print(f'| {count} BBox removed in total.')


class Dataset:
    def __init__(self, data_root_dir=None):
        self.data_root_dir = data_root_dir
        if data_root_dir is not None:
            smrc.utils.assert_dir_exist(self.data_root_dir)

    def get_full_path(self, rel_file_path):
        return os.path.join(self.data_root_dir, rel_file_path)

    @staticmethod
    def frame2video(image_dir):
        # , dir_list=None
        smrc.utils.convert_frames_to_video_inside_directory(
            image_dir, fps=30, ext_str='.mp4')

    # def video2frame(video_dir):


class AnnotationData(ImageData, Dataset):
    """
    Image and Label Data.
    """
    def __init__(self, image_root_dir, label_root_dir, class_list_file=None,
                 data_dir=None,
                 valid_dir_list=None):
        ImageData.__init__(self, image_root_dir=image_root_dir, valid_dir_list=valid_dir_list)
        Dataset.__init__(self, data_root_dir=data_dir)
        # super(LabelData).__init__(label_root_dir=label_root_dir, valid_dir_list=valid_dir_list)

        self.image_root_dir = image_root_dir
        self.label_root_dir = label_root_dir
        self.class_list_file = class_list_file
        self.data_root_dir = data_dir  # disable the use of the

        # self.image_data_tool_ = ImageData(image_root_dir=image_root_dir, valid_dir_list=valid_dir_list)
        # self.label_data_tool_ = LabelData(label_root_dir=label_root_dir, valid_dir_list=valid_dir_list)

    def clean_raw_annotation(self, label_dir_to_check, checked_result_dir=None):
        from smrc.utils import AnnotationPostProcess
        image_dir = self.image_root_dir
        # label_dir_to_check = self.label_root_dir
        AnnotationPostProcess(
            image_dir,
            label_dir_to_check,
            checked_result_dir=checked_result_dir,  # default: self.label_dir_to_check + '_tmp_SMRC_FORMAT'
            operation='correct',  # default 'check' 'correct'
            min_bbox_width=5, min_bbox_height=5, min_area=20
        )

    def to_yolo_format(self, smrc_label_dir, yolo_label_dir, class_list=None, dir_list=None):
        if self.class_list_file is not None and os.path.isfile(self.class_list_file):
            class_list = smrc.utils.load_1d_list_from_file(self.class_list_file)

        smrc.utils.annotate.transfer_smrc_label_to_yolo_format(
            image_dir=self.image_root_dir,
            smrc_label_dir=smrc_label_dir,
            yolo_format_dir=yolo_label_dir,
            generate_empty_ann_file_flag=False,
            post_process_bbox_list=False,
            class_list=class_list,
            dir_list=dir_list
        )

    @staticmethod
    def to_yolo_format_for_each_class(yolo_label_dir, class_list=None):
        # yolo_label_dir = os.path.join(data_root_dir, 'YOLO_FORMAT')
        smrc.utils.annotate.generate_single_label_data_from_yolo_format(
            yolo_label_dir=yolo_label_dir,
            class_list=class_list
        )

    # def get_complete_label_dir_list(self):
    #     return smrc.not_used.get_dir_list_in_directory(
    #         self.data_root_dir
    #     )

    def generate_excel_file_for_annotation(self, excel_file_name=None):
        """
        Generate the excel file for managing the annotation work.
        Sort based on the number of bounding boxes for the videos to annotation.
        :return:
        """
        if excel_file_name is not None:
            # make sure the file is not existing
            assert not os.path.isfile(excel_file_name), \
                f'Excel file {excel_file_name} already exists, please' \
                f'change the name for the resulting file.'
        else:
            excel_file_name = self.get_full_path(self.image_root_dir * '_AnnotationFile')

        # get all the image dir list
        image_data_tool = ImageData(image_root_dir=self.image_root_dir, valid_dir_list=self.valid_dir_list_)
        dir_list = image_data_tool.get_valid_dir_list()

        # if there is not dir list to process, then return
        if len(dir_list) == 0:
            return
        # do not sort the video, as we will sort them later based on the bbox num
        video_infor = image_data_tool.get_video_inf(sort_video_by_img_w=False)

        if len(video_infor) > 0:
            pbar = tqdm(enumerate(video_infor))
            for dir_idx, single_video_inf in pbar:
                dir_name, num_img, img_h, img_w = single_video_inf
                pbar.set_description(
                    f'To count bbox from {dir_name} ({dir_idx}/{len(video_infor)}) ...')
                ann_dir_path = os.path.join(self.label_root_dir, dir_name)

                num_bbox = smrc.utils.count_bbox_num_for_single_dir(ann_dir_path)
                single_video_inf.append(num_bbox)
            # sort the video list based on the number of bbox from small to large
            video_infor = sorted(video_infor, key=lambda x: x[-1], reverse=False)

            filed_name_list = ['Video_ID', 'Num_of_Image', 'Image_Height', 'Image_Width', 'Num_BBox']
            smrc.utils.save_excel_file(
                list_2d=video_infor,
                result_file_name=excel_file_name,
                field_name_list=filed_name_list
            )


class YOLOv4DetData:
    """
    The json file is in normalized format.
    """

    def __init__(self, image_dir, v4_json_dir):
        self.image_dir = image_dir
        self.v4_json_dir = v4_json_dir
        smrc.utils.assert_dir_exist(self.image_dir)
        smrc.utils.assert_dir_exist(self.v4_json_dir)

    def json_v4_to_smrc(self, v3_json_dir):
        json_yolov4_to_yolov3(
            json_yolov4_dir=self.v4_json_dir,
            json_yolov3_dir=v3_json_dir,
            image_root_dir=os.path.dirname(self.image_dir)
        )


class SMRCJsonDetData:
    """
    The json file in json_dir should be x1, y1, x2, y2 and not in normalized format
    so that we can directly know the box coordinates.
    """

    def __init__(self, image_dir, json_dir):
        self.image_dir = image_dir
        self.json_dir = json_dir
        smrc.utils.assert_dir_exist(self.image_dir)
        smrc.utils.assert_dir_exist(self.json_dir)

    def json2label(self, score_thd=0.20, nms_thd=0.5, label_dir=None):
        """
        :param label_dir:
        :param score_thd: confidence level threshold
        :param nms_thd: non maximum suppression threshold
            tiny objects (e.g., license plate) should use small thd, e.g., 0.1.
        :return:
        """
        from smrc.utils.annotate import ImportDetection
        json_file_dir = self.json_dir
        if label_dir is None:
            label_dir = json_file_dir + '_labels' + str(score_thd) + '_nms' + str(nms_thd)
        ImportDetection(
            json_file_dir=json_file_dir,
            label_dir=label_dir,
            score_thd=score_thd,
            nms_thd=nms_thd
        )