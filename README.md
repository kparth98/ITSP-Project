# ITSP-Project
The goal of this ITSP Project is to create a system for detecting Offside in the game of Football using Computer Vision algorithms

To run this code, you require python 2.7 and opencv 3

USAGE
To detect offside in a pre-recorded video, type the following in terminal:
python Offside_detection.py -v 'name of video file'

To detect offside from live camera feed:
python Offside_detection.py

While running the program, press 'i' to input and 'q' to quit

The input to be given are(in the following order):-
	1) Input a region of jersey of any player of first team by left clicking and dragging for detecting players using colour 	detection
	2) In the same way, input region of jersey of second team
	3) Input region of ball in similar manner
	4) Input the field by selecting 2 points along each edge. The edges to be selected in the following way: Top edge, Left 	edge, Bottom edge, Right edge

		NOTE:-If edges are not input in this exact manner or if the regions selected include background then program will give wrong output

