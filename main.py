from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import os

if __name__ == '__main__':
    # parse command line arguments
    parser = ArgumentParser(description='find moving objects of interest in .mp4 files using YOLOv4',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', nargs='?', default='./', type=Path,
                        help='parent directory of video files')
    args = parser.parse_args()
    print(f'Input file path: {args.input}')

    # find all mp4 files
    for path, subdirs, files in os.walk(args.input):
        for name in files:
            print(os.path.join(path, name))
            print(os.path.join(path, name))