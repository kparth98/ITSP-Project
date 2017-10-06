## Offside Detection System for Football

### Abstract

Here, we present our attempt towards the creation of a mechanism by which the problem of incorrect offside decisions made by referees in football can be addressed. The offside detection problem is simplified by classifying the attacking and defending teams based on the half in which the forward ball is played. Further the assumption that team members wear the same coloured jerseys works towards simplifying the problem.
The implementation involves two separate modules that track the ball and the players respectively. The successful integration of the modules leads to the desired goal of offside detection. 
The model works well with numerous sample situations of a single defender against two attacking players.

### Motivation
Being avid football followers we commonly notice unfair decisions being called due to an error of judgment on the part of the referees which is a limitation which cannot be overcome in the present system. The off-side rule is by-far the rule that is the most abused. Ideating and creating a system which can handle the rule in question in a fair manner will go a long way in the game of football.



### Usage of Program:
* To detect offside in a pre-recorded video, go to the folder in which code and video are stored and type the following in terminal:

```javascript
$ python Offside_detection.py -v 'path/name of video file'
```
* To detect offside from live camera feed:
```javascript
$ python Offside_detection.py
```
* While running the program, press ***'i'*** to input and ***'q'*** to quit
	1. Input patch of jersey of any player of team A by clicking and dragging with the mouse and making sure that background does not get included. The size of patch doesn’t matter much though it is better to have a bigger patch.
	2.  Input patch of jersey of any player of team B in similar manner
	3.  Input patch of ball
	4.  Input two points along each side of the field in the exact order as shown i.e. Top edge, Left edge, Bottom edge then Right edge.

### Approach to problem:
* The problem was broken down into a **ball tracking module** and a  **player tracking module** then combining the two to detect Offside. 
* The **ball tracking module** would take care of detecting the ball, tracking it and detecting whether a ball pass has occurred. 
* The **player tracking module** would detect the players of each team, attacking and defending, and get an approximate location of the foot of the players. 
* Finally the two would be integrated into one program. The ball pass is detected only when it is passed from one player to different player. 
* If the player of the attacking team receiving the pass(when he receives the pass) is behind the last player of the defending team then offside is called.
* The offside region is shown by a line passing through the position of the last defender.
* Note that offside is NOT called if the attacking player is behind the offside line but doesn’t receive the ball.

### Coding Technicality
We used Python 2.7 and OpenCV 3.0 and did the coding on PyCharm IDE. The installation procedure was made easy by using Anaconda Python. It is recommended to create virtual environment using Anaconda and install OpenCV on it rather than using the in-built Python(in case of MacOS and Linux).
We chose python as it is easy to code in and it was fast enough for our application.
### Initial Learning:
We initially started learning OpenCV and Python by writing basic codes and studying examples given in OpenCV library and on internet. We found the following links to be very helpful along with the documentation and stackoverflow :
* [http://www.pyimagesearch.com/](http://www.pyimagesearch.com/)
* [http://docs.opencv.org/trunk/d6/d00/tutorial_py_root.html](http://docs.opencv.org/trunk/d6/d00/tutorial_py_root.html)
* [https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)

### Ball Tracking:
* We tried multiple techniques for ball detection namely colour thresholding, histogram backprojection, Hough circle transform and dlib library’s correlation tracker. 
* Out of these, colour thresholding seemed to work the best. Its simple and efficient (Occam’s Razor :P ). Although it will give noise if colour of ball matches the background and thus we used it along with background subtraction.
* Histogram backprojection is another good technique and using a 2D histogram of hue and value gives us some robustness in terms of lighting condition.
* Hough circle transform gave too much noise.
* Dlib library’s correlation tracker is great but it failed at tracking small and fast moving object like football. It also slows down immensely if number of objects to be tracked exceeds 3.
#### Detection and Tracking of Ball on the basis of colour using Thresholding
* Get a sample patch of the ball as input
* Convert from the default RGB space to the HSV colour space and get the range of Hue, Saturation and Value
* Apply background subtraction to get a mask.
* Apply the mask to the frame
* Threshold the frames of the video to get contour of the ball which would be the contour with largest area.
* Generate a minimum enclosing circle and get center of the contour to detect the ball.
* For tracking, do this for every frame and store points of previous 10-20 frames
* The Pass is detected by change in trajectory which in turn is detected by a large enough change in direction of vector joining the position of center 10 frames before to the current position. This is done so as to eliminate false detection due to noise in the position of center of ball. 
### Player Tracking:
* We first thought of using dlib library’s correlation tracker but we realised that it became very slow if we tracked more than 3 players. Hence it could not be used for real time application.
* We even tried HOG descriptor with SVM which is a general method of detecting humans using machine learning and comes pre-trained in /supOpenCV but it gave very disappointing results and was very slow.
* In this case we used Histogram Backprojection to detect players from the colour of their jersey. We take a patch of jersey as input and calculate its histogram with axes hue and saturation and normalize it. 
* Then we used the calcBackProject function of OpenCV to generate a grayscale image where the magnitude of pixel value is proportional to its similarity to the patch given as input
* We then threshold this image and apply erosion and dilation to remove noise and we obtain contours of the jersey of the players. This is separated from the noise on the basis of contour area and the fact that height of minimum enclosing square is greater than the width.
* The location of the player’s feet is approximated to be 1.5 times the height of the contour below the center of the contour
### Generating the top view
* The user first inputs the endpoints of the field. In our case, the endpoints were not visible, hence we take 2 points along each sideline and solve the equation of line to get the 4 intersection points.
* Using the getPerspectiveTransform function of OpenCV we obtain the homography matrix which maps source points to destination points
* The top view location of the feet of players and the ball is obtained by following formula: x' = Hx
* Where His the homography matrix (3x3 matrix given by getPerspectiveTransform function), x=[x<sub>1</sub> y<sub>1</sub> z<sub>1</sub>] where (x<sub>1</sub>,y<sub>1</sub>) is the location of the foot in camera view and x'=[x'<sub>1</sub> y'<sub>1</sub> z'<sub>1</sub>]<sup>T</sup>
* The top view co	ordinate are (x'<sub>1</sub>/z'<sub>1</sub> , y'<sub>1</sub>/z'<sub>1</sub> )
* The offside line is drawn by inverse mapping the endpoints of the line in the top view

### Offside Detection
* At every pass detected, we check its validity by checking whether it was passed between different players.
* The list index of the previous passer is stored in and checked whether the pass was received by a different player
* The player in possession of the ball is identified measuring distance between ball’s center and location of feet of players. This can give false positive if the ball is bouncing and the nearest player detected as passer.
* To overcome this, a change in trajectory was identified as pass and offside checked only if the distance was less than some maximum (100 in our case)
* Players of the defending team are sorted based on x-coordinate of their location in top view and offside line is drawn.
* If the ball is passed between players of defending team(in this case team B) then offside is not detected
* If the ball is passed between players of attacking team (team A) then we check whether the x-coordinate of the receiving player is less than the last player of team B. If that’s True then Offside is called.

### Result:
### Future Prospects of Improvement
* The player’s and ball’s location can be detected with much greater accuracy by combining data from multiple cameras.
* In our application, there is still the issue of occlusion i.e. when one player blocks the view of another player. This can be resolved by using multiple viewpoints.
* Using multiple cameras to cover entire field.
* Using better algorithms to detect players and ball as which does not depend on colour of jersey.
![Image](/Downloads/screenshot.png)











