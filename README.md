# yolov4_reolink

TODO: proper readme file for usage. This is still a work in progress
 
The situation:

We have a new fancy security camera system (a Reolink NVR) that sends email alerts when it detects motion.

The problem:

There are far too many false positives for the email notifications to be useful. Most false positives are from snow falling near the camera or insects flying too close to the camera (insects can be attracted to the IR floodlight at night time). 

The Solution:

The NVR has the functionality to send the video clips over FTP to a server when it detects motion instead of in an email. I want this program to be able to find all video files in a folder, determine if it is a false positive (no *moving* people, animals, or vehicles found by YOLO), and then only the true positives get send as email notifications.

Issues I encountered so far:

you need to compile darknet yolov4 with CUDA enabled. Go through https://github.com/AlexeyAB/darknet/ and test to make sure that the darknet executable is working. This can be a very long process.

CUDA_PATH environment variable needs to be set.
