from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import os

# find all files is dir and it's subdirectories ending in .ext
# dir = folder path as a string: "\path\to\dir"
# ext = file extension as a string (including the dot): ".mp4"
def findFiles(dir, ext):
    found = []
    for path, subdirs, files in os.walk(dir):  # walk through all files
        for name in files:
            if os.path.splitext(name)[1] == ext:  # check if extension matches
                found.append(os.path.join(path, name))
    return found

if __name__ == '__main__':
    # parse command line arguments
    parser = ArgumentParser(description='find moving objects of interest in .mp4 files using YOLOv4',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', nargs='?', default='./', type=Path,
                        help='parent directory of video files')
    args = parser.parse_args()
    print(f'Input file path: {args.input}')

    # find all mp4 files
    mp4s = findFiles(args.input, '.mp4')
    print(mp4s)


