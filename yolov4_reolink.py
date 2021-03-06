# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import math
from argparse import ArgumentParser
from pathlib import Path
import time
import cv2
import ffmpeg
import datetime
from threading import Thread
from queue import Queue, Full
import random
from sklearn.cluster import DBSCAN

from functions import *

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
    printIf('\tvideo_capture done', args.verbose)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        # get video frame
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        # detect objects in frame
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=darknet_thresh)
        # find if detection name is in objects_of_interest.txt
        i = 0  # detections index
        while i < len(detections):
            # found = False
            g = -1  # group number index
            for k in range(0, len(OOI_groups)):
                group = OOI_groups[k]
                for j in range(0, len(group)):
                    # if detection label is found in objects_of_interest.txt
                    if detections[i][0] == group[j]:
                        # append group number to detections[i]
                        g = k
                        detections[i] = detections[i] + (g,)
            if g < 0:  # delete detection if not found in objects_of_interest.txt
                del detections[i]
                i -= 1
            i += 1
        detections_queue.put(detections)
        process_fps = int(1/(time.time() - prev_time))
        try:
            fps_queue.put(process_fps, timeout=1)
        except Full:
            pass
        if args.verbose:
            sys.stdout.write("fps: %d   \r" % (process_fps))
            sys.stdout.flush()
        darknet.free_image(darknet_image)
        darknet_image_queue.task_done()
    printIf('\tinference done', args.verbose)
    cap.release()

# modified from darknet.py
# added bsize, fscale, fsize to change the bounding box thickness, font scale, and font line thickness
def draw_boxes(detections, image, colors, bsize, fscale, fsize):
    import cv2
    for detection in detections:
        label, confidence, bbox = detection[0:3]
        left, top, right, bottom = darknet.bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], bsize)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    colors[label], fsize)
    return image


def drawing(frame_queue, detections_queue, fps_queue, bsize, fscale, fsize):
    random.seed(3)  # deterministic bbox colors
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            int(cap.get(cv2.CAP_PROP_FPS)), (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps_queue.get()  # this is just used to clear the fps queue
        fps_queue.task_done()
        detections_adjusted = []
        if frame is not None:
            for detection in detections:
                label, confidence, bbox, group_num = detection
                bbox_adjusted = convert2original(frame, bbox)
                tup = (str(label), confidence, bbox_adjusted, group_num)
                detections_adjusted.append(tup)
                all_detections.append(tup)
            image = draw_boxes(detections_adjusted, frame, class_colors, bsize, fscale, fsize)
            video.write(image)
        frame_queue.task_done()
        detections_queue.task_done()
    printIf('\tdrawing done', args.verbose)
    cap.release()
    video.release()
    cv2.destroyAllWindows()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    # text files
    converted_name = 'converted.txt'  # text file that contains paths of video files that have been run through ffmpeg
    ffmpeg_err_name = 'ffmpeg_errors.txt'  # text file that contains any errors encountered by ffmpeg
    OOI_name = 'objects_of_interest.txt'  # text file containing groups of names of objects of interest
    # ffmpeg options
    ext = '.mp4'  # extension of video files
    fps = 10  # framerate that video files should be reduced to
    output_crf = 45  # constant rate factor of output video
    output_width = 600  # width of output video in pixels (height is calculated based on width)
    # darket options (file paths can be relative to darknet executable)
    darknet_thresh = 0.55  # threshold for darknet detector
    configPath = "./cfg/yolov4.cfg"  # Path to cfg
    weightPath = "./yolov4.weights"  # Path to weights
    dataPath = "./cfg/coco.data"  # Path to meta data
    # DBSCAN options
    dbscan_epsilon = 100  # clustering search radius
    dbscan_minpoints = 3  # minimum number of points to cluster
    # threshold standard deviation in pixels
    # if the 2D (x,y) standard deviation of the points of a detected object
    # is greater than or equal to std_thresh, the object is considered moving
    std_thresh = 60
    # bounding box and text size
    box_thickness = 5
    font_size = 2
    font_thickness = 4

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
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='outputs more information to the console during processing')
    parser.add_argument('-s', '--save_false', action='store_true',
                        help='save videos detected as false positives')
    parser.add_argument('-r', '--remove', action='store_true',
                        help='delete original video files when processing is done')
    parser.add_argument('-o', '--output_detections', action='store_true',
                        help='output bounding box coordinates of detections in a text file in the video directory')
    args = parser.parse_args()

    # print args
    printIf(f'Input file path: {args.input}', args.verbose)
    printIf(f'GPU ID: {args.gpu}', args.verbose)
    printIf(f'darknet folder: {args.darknet_location}', args.verbose)
    printIf(f'PID(s): {args.pid}\n', args.verbose)

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

    # pause all processes given by -p argument
    suspendPid(args.pid)

    # convert video files with ffmpeg to reduce framerate (uses NVENC for fast encoding)
    for i in mp4s:
        printIf(f'converting {i} with ffmpeg...', args.verbose)
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
            # delete original file if flag is set
            if args.remove:
                if Path(i).exists():  # check that the file still exists
                    try:
                        os.remove(i)
                    except:
                        warn(f'Could not delete {i}. Check file permissions')

    # find all .lowfps files (they are encoded as h264 mp4 files, just with a different extension
    mp4s = findFiles(args.input, '.lowfps')

    # if no files need to be processed by darknet, stop here before loading all the darknet stuff
    if len(mp4s) == 0:
        print('no unprocessed .lowfps files found')
        resumePid(args.pid)  # resume processes before exit
        exit(0)

    # read in names of objects of interest from OOI_name
    OOI_groups, group_names = getOOI(OOI_name)

    # import darknet library from -d argument (this must be done *after* getting the path from the user)
    sys.path.insert(0, str(args.darknet_location))  # temporarily add darknet folder to path
    # if in windows, add some paths to dll search path
    if os.name == 'nt':
        os.add_dll_directory(str(args.darknet_location))
        os.add_dll_directory(str(Path(os.getenv('CUDA_PATH')) / 'bin'))
    import darknet
    darknet.set_gpu(args.gpu)  # specify which GPU darknet should use

    # navigate to darknet directory (darknet's default configuration uses paths relative to the executable)
    pwd = os.getcwd()
    os.chdir(args.darknet_location)
    # setup darknet network (only need first 2 return values)
    network, class_names, class_colors = darknet.load_network(configPath, dataPath, weightPath)
    # navigate pack to project dir
    os.chdir(pwd)

    # for each .lowfps video file
    for i in mp4s:
        printIf(f'Processing {i}...', args.verbose)
        output_video = os.path.splitext(i)[0] + '_yolo.mp4'  # output filename
        # set up multithreading Queues
        # darknet_image_queue and fps_queue are limited to save memory (doesn't seem to affect performance)
        # frame_queue and detections_queue are not limited for maximum performance
        frame_queue = Queue()
        darknet_image_queue = Queue(maxsize=1)
        detections_queue = Queue()
        fps_queue = Queue(maxsize=5)
        all_detections = []

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
        t = Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, box_thickness, font_size, font_thickness))
        threads.append(t)
        t_start = time.time()
        # start all threads
        for j in threads:
            j.start()
        # Wait for all threads to finish
        for j in threads:
            j.join()

        # define array to keep track of motion in each object group
        group_motion = np.zeros(shape=(len(group_names), 1), dtype=bool)
        # for each object group defined in OOI_name
        for group_name, groupid in zip(group_names, range(len(group_names))):
            # get bounding box coordinates of all detections from the current group
            bboxes = detectionsToArray(all_detections, groupid)
            printIf(f'\t{bboxes.shape[0]} detections found for group {group_name}', args.verbose)
            # if there are no detections for current group, skip
            if bboxes.shape[0] == 0:
                continue

            # preform DBSCAN clustering on bounding box coordinates in group
            # eps = search radius
            # min_samples = minimum number of points
            clusters = DBSCAN(eps=dbscan_epsilon, min_samples=dbscan_minpoints).fit(bboxes[:, 0:2])
            cluster_ids = set(clusters.labels_)
            # for each cluster
            for id in cluster_ids:
                # ignore outlier points
                if id == -1:
                    continue
                # get points of the current cluster
                cluster_points = bboxes[clusters.labels_ == id, 0:2]
                printIf(f'\t\t{len(cluster_points)} cluster points found in cluster {id}', args.verbose)
                # calculate standard deviations of the coordinates
                std_x = np.std(cluster_points[:, 0])
                std_y = np.std(cluster_points[:, 1])
                std_xy = math.sqrt(std_x**2 + std_y**2)
                printIf(f'\t\t\tstdx: {std_x}, stdy: {std_y}, std: {std_xy}', args.verbose)
                if std_xy >= std_thresh:
                    printIf(f'\t\t!!Detected a moving object of type {group_name}!!', args.verbose)
                    group_motion[groupid] = True

        # write detections to text file if flag is set
        if args.output_detections:
            detect_file = os.path.splitext(i)[0] + '_detect.txt'
            with open(detect_file, 'w') as file:
                for j in all_detections:
                    for k in j:
                        file.write(f'{k}\n')

        printIf(f'\tfinished {i} in {round(time.time() - t_start,1)}s', args.verbose)

        # remove file.mp4 from converted list
        removeLine(converted_name, os.path.splitext(i)[0] + '.mp4')
        # delete .lowfps file after detection is complete
        if Path(i).exists():  # check that the file still exists
            try:
                os.remove(i)
            except:  # just send warning if delete fails instead of stopping the program
                warn(f'Could not delete {i}. Check file permissions')

        # make new folders (if they don't already exist) in the video directory
        vid_folder = Path(os.path.dirname(i))
        true_path = Path(vid_folder) / 'true_positive'
        false_path = Path(vid_folder) / 'false_positive'
        if not true_path.exists():
            os.mkdir(str(true_path))
        if not false_path.exists() and args.save_false:
            os.mkdir(str(false_path))

        # move _yolo.mp4 file to the corresponding folder (re-encoding with ffmpeg to save space)
        output_path = ''
        if any(group_motion):
            output_path = str(true_path / os.path.basename(output_video))
        elif args.save_false:
            output_path = str(false_path / os.path.basename(output_video))
        if len(output_path) > 0:
            stream = ffmpeg.input(output_video).filter('scale', output_width, -1)\
                .output(output_path, f='mp4', vcodec='h264_nvenc', crf=output_crf, gpu=args.gpu).overwrite_output()
            try:
                out, err = stream.run(capture_stdout=True, capture_stderr=True)
                # if an error occurs, print error to console and ffmpeg_err_name file and continue with the next video
            except ffmpeg.Error as e:
                err = e.stderr.decode()
                print(err, file=sys.stderr)
                with open(ffmpeg_err_name, 'a') as file:
                    sttime = datetime.datetime.fromtimestamp(time.time()).strftime('[%d_%m_%Y_%H:%M:%S]')  # create timestamp
                    file.write(f'~~~~~~~~~~~~~~~~~~~~~~{sttime}~~~~~~~~~~~~~~~~~~~~~~\n{err}\n')
            if Path(output_video).exists():  # check that the file still exists
                # try to delete video file
                try:
                    os.remove(output_video)
                except:
                    warn(f'Could not delete {output_video}. Check file permissions')

    # resume suspended processes before exit
    resumePid(args.pid)

