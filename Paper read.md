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



## Dense Trajectories and motion boundary descriptors for action recognition

**Abstract** 
* Dense trajectory:
    - Trajectories: capture local motion information of the video
    - A dense representation guarentees a good coverage of foreground motion as well as of the surrounding context. 
    - A state-of-the-art optical flow algorithm enables a robust and efficient extraction of dense trajectories.
* As descriptors
    - features aligned with the trajectories to characterize
        + shape： point coordinates
        + appearance: histograms of oriented gradients
        + motions: histogram of optical flow
    - Introduce a descriptor based on motion boundary histograms (MBH) which rely on differential optical flow
        + The MBH descriptor shows to consistently outperform other state-of-the-art descriptors, in particular on real-world videos that contain a **significant amount of camera motion**.

### Intro
* Local space-time features are a successful representation for action recognition. 
* **However, the 2D space domain and 1D time domain in videos show different characteristics.** It is, therefore, more intuitive to handle them in a different manner than to detect interest points in a joint 3D space. **Tracking interest points through video sequences is a straightforward choice.** Existing methods:
    - Either, tracking techniques based on KLT tracker(sparse tracking)
    - Or, SIFT descriptors between consecutive frames are matched.
    - Or, combined both approaches and added random trajectories in low density regions of both trackers in order to increase density.
* **Dense sampling** has shown to improve results over sparse interest points for image classification. In action recognition, dense sampling at regular positions in space and time also outperforms state-of-the-art spatio-temporal interest point detectors. *In this work, we propose to sample feature points on a dense grid in each frame and track them using a state-of-the-art dense optical flow algorithm.*
* To reduce the influence of **camera motion** on **action recognition**, we introduce a descriptor based on motion boundaries, initially developed in the context of human detection. 
    - Motion boundaries are computed by a **derivative operation on the optical flow field**. **Thus, motion due to locally translational camera movement is canceled out and relative motion is captured** (see Figure 5). We show that motion boundaries provide a robust descriptor for action recognition that significantly outperforms existing state- of-the-art descriptors.










