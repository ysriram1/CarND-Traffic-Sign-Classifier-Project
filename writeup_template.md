# **Traffic Sign Recognition**

[//]: # (Image References)

[image1]: ./writeup_images/sample1.png "ex1"
[image2]: ./writeup_images/sample2.png "ex2"
[image3]: ./writeup_images/data_dist.png "class distribution"
[image4]: ./images_from_web/9.jpg "web1"
[image5]: ./images_from_web/14.jpg "web2"
[image6]: ./images_from_web/17.jpg "wb3"
[image7]: ./images_from_web/28.jpg "web4"
[image8]: ./images_from_web/32.jpg "web5"


## Overview

In this project we have been tasked with constructing a traffic signs classifier using tensor flow. We were expected to build a multilayer Convolutional Neural Network (CNN) that would train on a provided dataset with traffic signs. In addition to testing on a random split of the original data, we were also asked to find traffic sign images from the web and run them through the our classifier to see how well it performs on such data.

We first explored and pre-processed the data. Following this the data was split into training and validation (testing data was provided). After the data was split, we trained it using a CNN build on Tensorflow. The CNN was run for 50 Epochs and at the end of each Epoch the model was tested on the validation data.

The validation accuracy towards the final epoch was 97.1%. And the accuracy on the testing dataset was 95.2%.


The specific goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here is a link to my [project code](https://github.com/ysriram1/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Code
The code is in the *Traffic_Sign_Classifier.ipnb* file. The main sections in the code have been marked with titles and comments have been used to better explain different functions and methods used. All the mandatory sections of the project have been filled out.

** Dependencies: **
- Tensorflow
- numpy
- matplotlib
- pickle
- time
- os

** System configuration: **
- Intel i7 4.20GHz
- Nvidia GTX 1070 (Tensorflow GPU was used)
- 32 GB of RAM
- Windows 10

## Data
Each value in the dataset is a image and is assigned a class (0 to 42) based on the traffic sign it represents. Each image has a resolution of 32 x 32 with 3 color channels (R, G, B).

Examples of images from dataset:

![alt text][image1]

![alt text][image2]

***Here are the number of cases in each dataset used***:

Training: 34,799 cases
Validation: 4,410 cases
Testing: 12,630 cases

## Data Exploration and Pre-processing

The pixel intensities of each image were normalized to be between 0 and 1 from there originally scale of 0 to 255. This was done to make meaningful comparisons between different pixels.

We also explored to ascertain the class distribution:

![alt text][image3]

As we can see there is a large variation in the number images each class has. Hence, the data is quite unevenly distributed. In order to avoid unevenly splitting between train and validation sets, we performed stratified splitting.


## Convolutional Neural Network

My final model consisted of the following layers:

Please note that this model was generated after multiple iterations of different CNN designs. The validation accuracy (and not the testing accuracy) was used to fine-tune and change model.

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x10 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x30 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 16x16x64 		|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 13x13x60 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 6x6x60 		|
| Flatten					|												|
| Fully Connected				|  outputs 120			|
| RELU					|												|
| Fully Connected				|  outputs 43 (# of classes)			|
| Softmax				|   	<logits>								|


***hyperparameters used:***

* mean and standard deviation of normal distribution for initializing weights: 0 and 0.1
* EPOCHS: 50
* batch size: 128
* learning rate: 0.001

**Train Time**: 112.1 seconds

We used 50 epochs in training the model as the initial attempts to train using fewer epochs had shown that the model was still learning and didn't plateau yet. The entire

## Results
* training set accuracy of **95.8%** (at 50th Epoch)
* validation set accuracy of **95.4%** (at 50th Epoch)
* test set accuracy of **94.3%**
* of the 5 chosen images from the internet **4** were classified correctly.

## Data from Web
The neural network as used on 5 images obtained from the internet. These 5 images (displayed below) were chosen via a google image search by typing a traffic sign description.

| Image | Class | Description|
|:---------------------:|:---------------:|:--------------:|
|![][image4] | 9 |  No Passing|
|![][image5] |  14 |  Stop|
|![][image6] | 17 |  No Entry|
|![][image7] | 28 |  Yield|
|![][image8] | 32 |  End of all speed and passing limits|

All these images were run through the same pipeline as the training images. They were resized into 32x32 images and then normalized using a 0-1 normalization.

Here were the results obtained the corresponding softmax probability (in the same order as above):

| Actual Class | Predicted Class | Softmax Prob. of Prediction|
|:---------------------:|:---------------:|:--------------:|
| No Passing| No Passing | 1.0|
| Stop| Stop | 0.997|
| No Entry| Priority road | 0.867 |
| Yield| Yield | 0.785 |
| End of all speed and passing limits|  End of all speed and passing limits | 1.0 |

Only the "No entry" sign was misclassified. The misclassification was most likely due to the confusing background that all the other web images lacked. All the classifications had a very high confidence (including the misclassification). Hence, the accuracy of our model on images from web is **80%**.
