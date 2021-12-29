# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os
import sys
from warnings import warn
import psutil


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
                if exclude != None:
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
# encoding = encoding type: "utf8" by defualt
# returns empty txt_list if file does not exist
def readList(txt, encoding="utf8"):
    txt_list = []
    # read txt file if it exists
    if os.path.isfile(txt):
        # read working file
        with open(txt, 'r', encoding=encoding) as file:
            lines = file.readlines()
            txt_list = ([line.rstrip() for line in lines])
    return txt_list


# get groups of objects of interest from text file
# txt = name of text file: "path\to\file.txt"
# returns object names in each group in group_elements, and each group name in groups_names
def getOOI(txt):
    group_elements = []
    elements = []
    groups_names = []
    with open(txt, 'r') as file:
        lines = file.read().splitlines()
    for line in lines:
        # if there is an empty line, a new group has started
        if not line:
            # add current list of elements to group_elements if not empty
            if len(elements) > 0:
                group_elements.append(elements)
                elements = []  # reset elements list
            continue  # go to next line
        # if the line starts with '%', there is a new group name
        if line[0] == '%':
            groups_names.append(line[1:])
            continue  # next line
        # add new object name to elements list
        elements.append(line)
    # add last group of object names
    group_elements.append(elements)
    # check that each group has a name
    if len(group_elements) > len(groups_names):
        warn(f'group_elements (size {len(group_elements)}) is not the same length as groups_names (size {len(groups_names)}).\n'
             f'missing group names will be set to their group IDs')
        for i in range(len(groups_names), len(group_elements)):
            groups_names.append(f'{i}')
    return group_elements, groups_names


# write python list to file:
# file = file object
# list = python list object
def writeList(file, list):
    for i in list:
        file.write(f'{i}\n')
    return


# separate detections by group ID number
# detections = list of lists in the format [[object name, confidence, bounding box, groupid]]
#             where bounding box is a tuple: (x, y, width, height)
# groupid = integer
def detectionsToArray(detections, groupid):
    # get all indices where the groupID matches
    group_idx = [idx for idx in range(len(detections)) if detections[idx][3] == groupid]
    n = len(group_idx) # number of detections found
    # preallocate array = [x, y, width, height] from bounding box in detections
    array = np.zeros(shape=(n, 4), dtype=int)
    for i in range(0, n):
        # get bounding box
        bbox = detections[group_idx[i]][2]
        # save bounding box coordinates and size in array
        for j in range(0,4):
            array[i, j] = bbox[j]
    return array


# print to console only if verbose argument is true
def printIf(msg, arg):
    if arg:
        print(msg)
    return

# remove matching line from file
# filename = path to text file: "path\to\file.txt"
# match = string to match to a line
def removeLine(filename, match):
    # get all lines from file
    lines = readList(filename)
    # write all lines back to file except when line == match
    with open(filename, 'w') as file:
        for line in lines:
            if line == match:
                continue
            file.write(f'{line}\n')
    return