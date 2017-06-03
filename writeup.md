# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/extra_samples.png "Extra samples"
[image3]: ./examples/softmax_predictions.png "Softmax prediction"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the core python library to get the summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a random image from the training dataset. It label id and corresponding meaning is also shown.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

First of all, the rgb image is converted into gray-scale using the magic number  
`0.2989 * r + 0.5870 * g + 0.1140 * b`  

Convert the image from RGB to grayscale reduced the input space by 2/3. This because the gray-scaled traffic sign still has one to one relationship with the class label, i.e., traffic sign are color insensitive.

After converting the image, I applied normalization so that the data has zero mean and equal variance.  
To avoid the NN memorize the training data, random shuffling is also applied.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|----------------------:|:----------------------------------------------| 
| Input                 | 32x32x1 Gray-scale image                       | 
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x6     |
| ReLU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x6                     |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16      |
| ReLU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x16                     |
| Flatten layer         | outputs 400                                   |
| Fully connected        | outputs 400x120                                |
| ReLU                    |                                                |
| Dropout               | `keep_prob` 0.25                                |
| Fully connected        | outputs ?x84                                    |
| ReLU                    |                                                |
| Fully connected        | outputs 84x43                                    |
| ReLU                    |                                                |
|                        |                                                |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The hyperparameters in the model are:
- `EPOCHS`: the number of iterations until the network stops learning or start overfitting
- `BATCH_SIZE`: the highest number that your machine has a memory for.
- `KEEP_PROBABILITY`: the probability of keeping a node using dropout
- `LEARNING_RATE`: the learning rate of the optimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well-known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The original LeNet model gives 88% accuracy. Adding the dropout layer improves the system performance by preventing overfitting. The achieve 93% goal, I first tried to make the NN deeper by adding another 1x1 convolution net. The training becomes slower, and it can just reach the 93% goal. Later, I found that without the extra 1x1 convolution net gives better results. Therefore, the final model keeps similar with the LeNet model with an extra dropout layer applied.  
To tune the hyperparameter, I tried the different combinations of `EPOCHS`, `KEEP_PROBABILITY` and `LEARNING_RATE`, and measure the performance (accuracy). The final tuned hyperparameters are `EPOCHS=100`, `KEEP_PROBABILITY=0.25`, and `LEARNING_RATE=0.001`.

My final model results were:
* training loss of 0.010
* validation set accuracy of 96.3% 
* test set accuracy of 94.1%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image2]

Figures 1, 2, 3, 8 should be easy for the classifier. Figures 4 and 7 have very low resolution, which might be a problem to the classifier. Figures 5 and 6 are very similar as both are the speed limit sign. Also, number 7 and 3 are very similar as well. These might be the most difficult samples in this data set. The sign in Figure 9 has an irregular shape, which might be a problem.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Priority road          | Priority road                                   | 
| Turn right ahead         | Turn right ahead                                 |
| Yield                    | Yield                                            |
| No entry                  | No entry                                         |
| Speed limit (70km/h)    | Speed limit (70km/h)                             |
| Speed limit (30km/h)    | Speed limit (30km/h)                             |
| Children crossing        | Children crossing                             |
| Ahead only            | Ahead only                                    |
| Vehicles over 3.5 metric tons prohibited    | Vehicles over 3.5 metric tons prohibited    |
|                       |                                               |


The model was able to correctly predict 9 of the 9 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Surprisingly, the classifier has no difficulties for most of the images. Notably, it got a little confused on the Speed limit (30km/h) sign. 

![alt text][image3]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The features map is not available in this project. This is because the convolution network was generated using the function `conv2d_maxpool`. This function returns a convolution network after ReLU and maximum pooling.
