CARD - Crop, Annotate, Record Dataset

What is CARD? 
CARD is an annotation engine and synthetic dataset pipeline for objects that can be found easily with contours. This was designed for game cards, and is most useful for batch cropping or batch annotating anything that is flat and visually consistant like cards, box faces, or shipping labels. 

How to use:
To run the program, 

1: Load images or an image directory through the menu. 
2: Check a few images to make sure objects are being isolated properly. If they are not, disabling one or two color channels may help isolate the object from the background. Up and down arrow moves up and down the file list.
3: Click "File/Save all crops". For each image on the file list, CARD will create a new image containing only the what is within the bounding box with a transparent background. CARD will apply image adjustments to each image before extracting. Files will be placed in <dir>/output by default, where dir is the directory of the last file opened.
4: Click "Create dataset." CARD will superimpose your cropped images onto random backgrounds all subdirectories of the chosen folder to create a full dataset of annotated images.

Alternatively, to create a label file in the chosen format for each image in the file list, click "File/Save all labels"

To go through images one at a time, Spacebar will save the crop and move to the next image in the list.

Currently only YOLO bounding boxes and oriented bounding boxes are supported.