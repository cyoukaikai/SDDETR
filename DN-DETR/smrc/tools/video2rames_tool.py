import argparse
import smrc.utils

if __name__ == '__main__':
    # (mask-rcnn) kai@kai:~/tuat$ python smrc/tools/video2rames_tool.py -i test_data/test_videos

    # change to the directory of this script
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Conver videos to images')
    parser.add_argument('-i', '--input_dir', default='test_videos', type=str, help='Path to input directory')

    args = parser.parse_args()

    # we can specify the active directory by using args_5d_det.input_dir
    smrc.utils.convert_multiple_videos_to_frames(args.input_dir)


