# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import time
import psutil
import cv2
import ffmpeg
import datetime
from threading import Thread, enumerate
from queue import Queue, Full
import random


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ My Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# pause all processes given by list of pids
def suspendPid(pids):
    if pids is None:
        return
    for pid in pids:
        print(f'suspending PID={pid}')
        try:
            psutil.Process(pid).suspend()
        except psutil.Error as e:
            print(f'error when trying to suspend PID={pid}')
            print(e, file=sys.stderr)
            sys.exit(1)
    return


# resume all processes given by list of pids
def resumePid(pids):
    if pids is None:
        return
    for pid in pids:
        print(f'resuming PID={pid}')
        try:
            psutil.Process(pid).resume()
        except psutil.Error as e:
            print(f'error when trying to resume PID={pid}')
            print(e, file=sys.stderr)
            sys.exit(1)
    return


# read a text file an add each list to a list
# txt = name of text file: "path\to\file.txt"
# returns empty txt_list if file does not exist
def readList(txt):
    txt_list = []
    # read txt file if it exists
    if os.path.isfile(txt):
        # read working file
        with open(txt, 'r') as file:
            lines = file.readlines()
            txt_list = ([line.rstrip() for line in lines])
    return txt_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Modified darknet_video.py Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    print('\tvideo_capture done')
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=darknet_thresh)
        detections_queue.put(detections)
        process_fps = int(1/(time.time() - prev_time))
        try:
            fps_queue.put(process_fps, timeout=1)
        except Full:
            pass
        #print(f'fps: {process_fps}')
        sys.stdout.write("fps: %d   \r" % (process_fps))
        sys.stdout.flush()
        # darknet.print_detections(detections, False)
        darknet.free_image(darknet_image)
    print('\tinference done')
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            int(cap.get(cv2.CAP_PROP_FPS)), (video_width, video_height))

    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fpsq = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            video.write(image)
        else:
            break
    print('\tdrawing done')
    # fps_queue.task_done()
    cap.release()
    video.release()
    cv2.destroyAllWindows()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    # constants
    converted_name = 'converted.txt'  # text file that contains paths of video files that have been run through ffmpeg
    detected_name = 'detected.txt'  # text file that contains paths of video files that have been run through the detector
    ffmpeg_err_name = 'ffmpeg_errors.txt'  # text file that contains any errors encountered by ffmpeg
    ext = '.mp4'  # extension of video files
    fps = 10  # framerate that video files should be reduced to
    darknet_thresh = 0.4  # threshold for darknet detector

    # parse command line arguments
    parser = ArgumentParser(description=f'find moving objects of interest in {ext} files using YOLOv4')
    parser.add_argument('-i', '--input', default='./', type=Path,
                        help='parent directory of video files. Default is "./"')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='Specify which GPU to use in multi-gpu systems. Default is 0')
    parser.add_argument('-d', '--darknet_location', default='./darknet', type=Path,
                        help='path for the darknet folder. Default is "./darknet"')
    parser.add_argument('-p', '--pid', nargs='+', default=None, type=int,
                        help='specify the PID of 1 or more processes that should to be paused. '
                             'This may require elevated privileges.')
    args = parser.parse_args()

    # print all args for debugging
    print(f'Input file path: {args.input}')
    print(f'GPU ID: {args.gpu}')
    print(f'darknet folder: {args.darknet_location}')
    print(f'PID(s): {args.pid}\n')

    # get list of .mp4 files that have been converted already
    mp4s_converted = readList(converted_name)
    # find all video files excluding the ones already in mp4s_converted
    mp4s = findFiles(args.input, ext, mp4s_converted)
    # remove any files ending in _yolo.mp4
    i = 0
    while i < len(mp4s):
        if mp4s[i].endswith('_yolo.mp4'):
            del mp4s[i]
            i -= 1
        i += 1
    del i

    # pause all processes given by -p argument
    suspendPid(args.pid)

    # convert video files with ffmpeg to reduce framerate (uses NVENC for fast encoding)
    for i in mp4s:
        print(f'converting {i} with ffmpeg...')
        output_path = os.path.splitext(i)[0] + '.lowfps'  # same filename but different extension
        # get current video fps
        cap = cv2.VideoCapture(i)
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # check that target fps is not larger than current fps
        output_fps = fps
        if output_fps >= current_fps:
            output_fps = current_fps
        stream = ffmpeg.input(i).filter('fps', fps=output_fps) \
            .output(output_path, f='mp4', vcodec='h264_nvenc', crf=23, gpu=args.gpu) \
            .overwrite_output().global_args('-an')  # setup ffmpeg stream to convert video to framerate set by fps variable and remove audio since we don't need it
        # try ffmpeg conversion
        try:
            out, err = stream.run(capture_stdout=True, capture_stderr=True)
        # if an error occurs, print error to console and ffmpeg_err_name file and continue with the next video
        except ffmpeg.Error as e:
            err = e.stderr.decode()
            print(err, file=sys.stderr)
            with open(ffmpeg_err_name, 'a') as file:
                sttime = datetime.datetime.fromtimestamp(time.time()).strftime('[%d_%m_%Y_%H:%M:%S]')  # create timestamp
                file.write(f'~~~~~~~~~~~~~~~~~~~~~~{sttime}~~~~~~~~~~~~~~~~~~~~~~\n{err}\n')
        finally:
            # after ffmpeg is done re-encoding the file (or failed), add file path to converted_name so it doesn't get converted again
            with open(converted_name, 'a') as file:
                file.write(f'{i}\n')

    # get list of .lowfps files that have been converted already
    mp4s_detected = readList(detected_name)

    # find all .lowfps files excluding the ones in mp4s_detected
    mp4s = findFiles(args.input, '.lowfps', mp4s_detected)

    # if no files need to be processed by darknet, stop here before loading all the darknet stuff
    if len(mp4s) == 0:
        print('no unprocessed .lowfps files found')
        resumePid(args.pid)  # resume processes before exit
        exit(0)

    # import darknet library from -d argument (this must be done *after* getting the path from the user)
    sys.path.insert(0, str(args.darknet_location))  # temporarily add darknet folder to path
    # if in windows, add some paths to dll search path
    if os.name == 'nt':
        os.add_dll_directory(str(args.darknet_location))
        os.add_dll_directory(str(Path(os.getenv('CUDA_PATH')) / 'bin'))
    import darknet
    darknet.set_gpu(args.gpu)  # specify which GPU darknet should use

    # import functions from modified darknet_video file (this must be done after first importing darknet)
    # from darknet_video_modified import video_capture, inference, drawing

    # darknet file paths (relative to darknet executable)
    configPath = "./cfg/yolov4.cfg"  # Path to cfg
    weightPath = "./yolov4.weights"  # Path to weights
    dataPath = "./cfg/coco.data"  # Path to meta data

    # navigate to darknet directory (darknet's default configuration uses paths relative to the executable)
    pwd = os.getcwd()
    os.chdir(args.darknet_location)
    # setup darknet network (only need first 2 return values)
    network, class_names, class_colors = darknet.load_network(configPath, dataPath, weightPath)
    # navigate pack to project dir
    os.chdir(pwd)

    # for each .lowfps video file
    for i in mp4s:
        print(f'Processing {i}...')
        output_video = os.path.splitext(i)[0] + '_yolo.mp4'  # output filename
        # set up multithreading Queues
        # darknet_image_queue and fps_queue are limited to save memory (doesn't seem to affect performance)
        # frame_queue and detections_queue are not limited for maximum performance
        frame_queue = Queue()
        darknet_image_queue = Queue(maxsize=1)
        detections_queue = Queue()
        fps_queue = Queue(maxsize=5)

        # get network dimensions
        darknet_width = darknet.network_width(network)
        darknet_height = darknet.network_height(network)
        # open input video
        cap = cv2.VideoCapture(i)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # set up threads
        threads = []
        t = Thread(target=video_capture, args=(frame_queue, darknet_image_queue))
        threads.append(t)
        t = Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue))
        threads.append(t)
        t = Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue))
        threads.append(t)
        t_start = time.time()
        # start all threads
        for j in threads:
            j.start()
        # Wait for all threads to finish
        for j in threads:
            j.join()

        print(f'\tfinished {i} in {round(time.time() - t_start,1)}s')
        # after darknet is done detecting objects, add file path to detected_name so it doesn't get converted again
        with open(detected_name, 'a') as file:
            file.write(f'{i}\n')

    # resume suspended processes before exit
    resumePid(args.pid)

