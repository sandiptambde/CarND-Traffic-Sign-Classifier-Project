# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sandiptambde/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the basic python built-ins to calculate summary statistics of the traffic
signs data set:

* The size of training set is :34799
* The size of the validation set is :4410
* The size of test set is :12630
* The shape of a traffic sign image is :32x32x3
* The number of unique classes/labels in the data set is :43


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have not done data preprocesing for now other than training dataset shuffling. I mainly work on network architechure to achieve validation accuracy >93%.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution1 5x5     	| 1x1 stride, same padding, outputs:28x28x15 	|
| RELU1					| outputs:28x28x15								|
| Max pooling1	      	| 2x2 stride,  outputs:14x14x15 				|
| Convolution2 3x3     	| 3x3 stride, same padding, outputs:12x12x15 	|
| RELU2					| outputs:12x12x15								|
| Max pooling2	      	| 2x2 stride,  outputs:6x6x15 					|
| Convolution3 3x3     	| 3x3 stride, same padding, outputs:4x4x15	 	|
| RELU3					| outputs:4x4x15								|
| Max pooling2	      	| 2x2 stride,  outputs:2x2x15 					|
| FC0					| shape : 540 									|
| FC1			      	| shape:540x120 								|
| FC2			      	| shape:120x84									|
| FC3			      	| shape:84x43	 								|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an following hyperparameter:
EPOCHS = 30
BATCH_SIZE = 64
mu = 0
sigma = 0.1
learning_rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of : 0.992
* validation set accuracy of :0.932
* test set accuracy of : 0.909

I initially started with LeNet acrchitechure used for digit recognition.
Problem faced:
- Adjusting hyperparameters like learning rate(0.1-0.0001),epoch(10-50) wasn't improving the validation accuracy >0.90%

So, I decided to modify LeNet architecture. I done following modifications:
1. Increased number of output neurons in first conv1 layer to 15 from 6
2. Decreased conv filter size from 5x5 to 3x3 to retain more info in second conv2 layer
3. Added 3rd conv3 layer with same properties as conv2 
4. Kept output neuron size of all conv layer same to 15
5. Tried with diffent epoch:10,20,30 
6. Reduce Batch size to 64 from 128 for stability

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

Available under test_images folder


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) 	| Speed limit (30km/h)							| 
| Pedestrians  			| Right-of-way at the next intersection			|
| Roadwork				| Roadwork										|
| Speed limit (60km/h)	| Speed limit (50km/h)			 				|
| no_entry				| no_entry		      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 145th cell of the Ipython notebook.

For the first image, the model is 100% sure that this is a speed_limit30 (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)							| 
| 9.75613078e-17		| Speed limit (80km/h)							|
| 1.75238187e-19		| Speed limit (50km/h)							|
| 4.98145406e-35		| Speed limit (20km/h)			 				|
| 0					    | Speed limit (60km/h)							|


For the second image, the model is uncertain that this is a Pedestrians (probability of 2.98733803e-16), It things it as Right-of-way at the next intersection. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection			| 
| 1.50338075e-12		| Traffic signals								|
| 2.98733803e-16		| Pedestrians									|
| 2.09060446e-23		| Road narrows on the right		 				|
| 2.44842503e-26		| Beware of ice/snow							|

For the third image, the model is 100% sure that this is a Road work (probability of 1.0), and the image does contain a Road work. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work										| 
| 2.59604906e-08		| Dangerous curve to the right					|
| 1.88901250e-09		| Right-of-way at the next intersection			|
| 3.95308676e-22		| Beware of ice/snow			 				|
| 4.35396788e-24		| Pedestrians									|

For the fourth image, the model is not able to predict it is a Speed limit (60km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99328017e-01		| Speed limit (80km/h)							| 
| 6.71963615e-04		| Keep left										|
| 2.08859985e-10		| Speed limit (30km/h)							|
| 9.14461757e-13		| Speed limit (70km/h)			 				|
| 1.10790053e-13		| Speed limit (80km/h)							|

For the fifth image, the model is 100% sure that this is a no_entry (probability of 1.0), and the image does contain a no_entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| no_entry										| 
| 1.66997560e-09		| Stop											|
| 2.02275694e-20		| Priority road									|
| 9.35290529e-23		| Yield							 				|
| 2.63863866e-25		| Traffic Signals								|
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Not yet Done.

