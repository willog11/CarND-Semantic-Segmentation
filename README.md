# Udacity Self-Driving Car Engineer Nanodegree
## Semantic Segmentation
### Introduction

The goal of this project is to construct a fully convolutional neural network (FCN) based on the VGG-16 image classifier architecture for performing semantic segmentation to identify drivable road area (trained and tested on the KITTI data set).

### Implementation
#### Architecture
A pre-trained VGG-16 network was converted to a FCN by taking different layer outputs (3, 4 and 7) and:

1. Encoding them using a convultional 2D layer. This makes sure the shapes are the same
2. Performing skip layer connections with the use of element wise and operations
3. Finally the results are upsampled by performing a transpose convolutional layer.

Note, each convolution and transpose convolution layer includes a kernel initializer and regularizer. Additionally the images are augemented in the training section by applying normalization to the images.

For further reading on the architecture, please see the [publication](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Berkley University.

#### Training

For training of the network the following hyperparameters were used:

~~~
keep_prob:0.5
learning_rate: 0.0001
epochs = 30
batch_size = 5
~~~

As mentioned above the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) was used during the traiing and evaluation stages.

#### Results

The following log is a breakdown of loss per epoch during testing. Its clear that there is a smooth drop in loss over each epoch and it seems to find a global minimum at 28 epochs. This is a very good results:

~~~
EPOCH 1 ...
Loss per epoch: = 0.598
EPOCH 2 ...
Loss per epoch: = 0.179
EPOCH 3 ...
Loss per epoch: = 0.132
EPOCH 4 ...
Loss per epoch: = 0.111
EPOCH 5 ...
Loss per epoch: = 0.099
EPOCH 6 ...
Loss per epoch: = 0.086
EPOCH 7 ...
Loss per epoch: = 0.073
EPOCH 8 ...
Loss per epoch: = 0.079
EPOCH 9 ...
Loss per epoch: = 0.061
EPOCH 10 ...
Loss per epoch: = 0.056
EPOCH 11 ...
Loss per epoch: = 0.052
EPOCH 12 ...
Loss per epoch: = 0.048
EPOCH 13 ...
Loss per epoch: = 0.045
EPOCH 14 ...
Loss per epoch: = 0.050
EPOCH 15 ...
Loss per epoch: = 0.045
EPOCH 16 ...
Loss per epoch: = 0.040
EPOCH 17 ...
Loss per epoch: = 0.037
EPOCH 18 ...
Loss per epoch: = 0.036
EPOCH 19 ...
Loss per epoch: = 0.035
EPOCH 20 ...
Loss per epoch: = 0.034
EPOCH 21 ...
Loss per epoch: = 0.031
EPOCH 22 ...
Loss per epoch: = 0.029
EPOCH 23 ...
Loss per epoch: = 0.028
EPOCH 24 ...
Loss per epoch: = 0.028
EPOCH 25 ...
Loss per epoch: = 0.029
EPOCH 26 ...
Loss per epoch: = 0.026
EPOCH 27 ...
Loss per epoch: = 0.026
EPOCH 28 ...
Loss per epoch: = 0.024
EPOCH 29 ...
Loss per epoch: = 0.024
EPOCH 30 ...
Loss per epoch: = 0.024
~~~

### Conclusion

Overall the network performs extremely well in nearly all cases. Rarely it doesn't perform as well in some areas with strong lighting and shadows. However this is to be expected with the amount of data that has been used.

The following demo shows this performance:
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/toTTx14s8jA/0.jpg)](https://www.youtube.com/watch?v=toTTx14s8jA)

# CarND-Semantic-Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
