#Paper read
## Action Recognition with Improved Trajectories
#### Terms and notions
* To estimate camera motion, we **match** feature points between frames using
    * SURF descriptor
    * dense optical flow
* Human motion is different from camera motion, and generates inconsistency in matches. A **human descriptor** is employed to remove inconsistent matches. 
* Motion based descriptors(Optical flow based motion descriptor):
    * HOF: Histogras of optical flow
    * MBH: Motion Boundary Histograms
* Space-time features are successful on some datasets. Since they avoid non-trivial pre-processing like tracking or segmentation. 
* space-time feature + bag of words => good action classification.
* Dense trajectory has been shown to perform best on a variety of datasets. (Other image features: 3D-SIFT, extended SURF, HOG3D and local itinary patterns.)
* What is optical flow?
    * a efficient way to suppress camera motion
* **Motion boundary histograms(MBH)** give the best results to compute the trajectories of feature points. 
* Homography: General projective transformation
* 


## Thumos 14 Action Recognition challenge
Two tasks: 
* Action recognition: predict presence
* Temporal action detection: predict presence/absence
### Data 
* Training data: based on UCF101, **temporally trimmed**
* Validation data, **untrimmed**
* Background data, **untrimmed**
* Test data, **untrimmed**

#### Action recognition data
* Training data: based on UCF101, **temporally trimmed**, 13320 videos of 101 human actions. Each category has more than 100 videos.
* Validation data, **untrimmed**, 1000 videos as validation data
* Background data, **untrimmed**
* Test data, **untrimmed**

#### Temporal Action Detection Data Set
* Training data: a subset of UCF101, 20 actions, 
* Validation data, 200 videos from 20 action classes
* Background data, 
* Test data, 1574 temporally untrimmed videos

#### Pre-Computed Low-level Video Features
IDTF


[, , , , Traj_index, HOG_index, HOF_index, MBH_index]

### Action recognition task
#### defin
* similar to conventional action recogmitnion.
* System should output a confidence value for the predicted presence
* The test data is untrimmed, and there might be part of the video containing no action, or multiple action might occur at different timestamps.
* THere will also be empty videos, that does not contain any actions.

#### Submission Format
submit the result of at most five runs. 

confidence score between 1 and 0 
#### Evaluation Metric
Interpolated Average Precision(AP). 

### Temporal action detection task
#### Definition
A system should output:
* a real-value score indicating the confidence of the prediction
* starting and ending time for the given action
* Test data : 20 action classes of 101 

#### Submission Format

[video_name] [starting_time] [ending_time] [class_label] [confidence_score]


The time should be in seconds with one decimal point precision. The confidence score shall be between 0 and 1. A larger confidence value indicates greater confidence in detecting the action of interests in a test video.

#### Evaluation
Interpolated Average Precision (AP) and its mean value (mAP) are the official metrics used to measure the performance of each action class and each run, respectively.**Just like action detection** A detection is marked as true or false positive based on the time period of overlap with the ground truth time range. The overlap \\(o\\)between the predicted time range and ground truth time range is computed as:
$$o = \frac{R_p\cap R_{gt}}{R_p\cup R_{gt}}

### Development kit
a software package composed of the code of the evalua- tion metrics for the recognition and detection tasks, 



## Challenge result from University of Amsterdam - THUMOS 2014
They 
* investigate and exploit the action-object relationship by capturing both motion and related objects
* local descriptors: HOG,HOF, MBH and improved dense trajectories
* video encoding: fisher vector
* also use deep net features to capture action context. 
* Actions are classified with *one vs rest* linear SVM














