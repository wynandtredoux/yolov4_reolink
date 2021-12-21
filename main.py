# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import os
import ffmpeg
import sys
import time
import datetime

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# find all files in dir and its subdirectories ending in 'ext'
# dir = folder path as a string: "\path\to\dir"
# ext = file extension as a string (including the dot): ".mp4"
# exclude = a list of files (in the same format as 'found') to exclude
# returns list of files as 'found'
def findFiles(dir, ext, exclude=None):
    found = []
    for path, subdirs, files in os.walk(dir):  # walk through all files
        for name in files:
            if os.path.splitext(name)[-1] == ext:  # check if extension matches
                path_name = os.path.join(path, name)
                # check if file is in exclude list
                do_exclude = False
                for i in exclude:
                    if i == path_name:  # if file is in exclude list, skip
                        do_exclude = True
                        break
                if do_exclude:
                    continue
                found.append(path_name)
    return found


# write a list to a file
# file = file object from open()
# list = any python list
def writeList(file, list):
    for i in list:
        file.write(f'{i}\n')
    return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    # constants
    wf_name = 'working.txt'  # text file that contains paths of video files that have been converted
    ffmpeg_err_name = 'ffmpeg_errors.txt'  # text file that contains any errors encountered by ffmpeg
    ext = '.mp4'  # extension of video files
    fps = 10  # framerate that video files should be reduced to

    # parse command line arguments
    parser = ArgumentParser(description='find moving objects of interest in .mp4 files using YOLOv4'
                            , formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('darknet_location', help='path for the darknet executable')
    parser.add_argument('-i', '--input', nargs='?', default='./', type=Path,
                        help='parent directory of video files')
    args = parser.parse_args()
    print(f'Input file path: {args.input}')

    mp4s_converted = []  # empty list for storing filepaths for .mp4 files that have been converted
    # read working file if it exists
    if os.path.isfile(wf_name):
        # read working file
        with open(wf_name, 'r') as file:
            lines = file.readlines()
            mp4s_converted = ([line.rstrip() for line in lines])

    # find all video files excluding the ones already in the working file
    mp4s = findFiles(args.input, ext, mp4s_converted)

    # convert video files with ffmpeg to reduce framerate (uses NVENC for fast encoding)
    for i in mp4s:
        output_path = os.path.splitext(i)[0] + '.lowfps'  # same filename but different extension
        stream = ffmpeg.input(i).filter('fps', fps=fps) \
            .output(output_path, f='mp4', vcodec='h264_nvenc', crf=23) \
            .overwrite_output().global_args('-an') # setup ffmpeg stream to convert video to framerate set by fps variable and remove audio since we don't need it
        try:  # try ffmpeg conversion
            out, err = stream.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:  # if an error occurs, print error to console and ffmpeg_err_name file and continue with the next video
            err = e.stderr.decode()
            print(err, file=sys.stderr)
            with open(ffmpeg_err_name, 'a') as file:
                sttime = datetime.datetime.fromtimestamp(time.time()).strftime('[%d_%m_%Y_%H:%M:%S]')  # create timestamp
                file.write(f'~~~~~~~~~~~~~~~~~~~~~~{sttime}~~~~~~~~~~~~~~~~~~~~~~\n{err}\n')
        # after ffmpeg is done re-encoding the file (or failed), add file path to working file so it doesn't get converted again
        with open(wf_name, 'a') as file:
            file.write(f'{i}\n')

    # send converted files to darknet YOLOv4
