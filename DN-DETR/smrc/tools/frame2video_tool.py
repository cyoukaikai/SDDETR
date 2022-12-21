import argparse
import smrc.utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image to video')
    parser.add_argument('-i', '--image_dir', default='', type=str, help='Path to image directory')
    args = parser.parse_args()

    smrc.utils.assert_dir_exist(args.image_dir)
    smrc.utils.convert_frames_to_video_inside_directory(
        args.image_dir, fps=2, ext_str='.mp4')

    # python smrc/tools/frame2video_tool.py \
    #   -i /home/sirius/dataset/nuScene/nuscenes/Visualization/2d_dets/mmdet_faster_rcnn