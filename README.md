# **Traffic Sign Recognition** 
---

[//]: # (Image References)

[image1]: ./examples/DataSetSample.jpg "DataSetSample"
[image2]: ./examples/TrainDataDist.jpg "TrainDataDist"
[image3]: ./examples/ValidDataDist.jpg "ValidDataDist"
[image4]: ./examples/TestDataDist.jpg "TestDataDist"
[image5]: ./examples/TrainDataSynthDist.jpg "TrainDataSynthDist"
[image6]: ./examples/grayscaleSample.jpg "grayscaleSample"
[image7]: ./examples/lenet.png "lenet"
[image8]: ./examples/TrainVsValid.jpg "TrainVsValid"
[image9]: ./examples/misclassfiedTrainSamples.jpg "misclassfiedTrainSamples"
[image10]: ./examples/misclassfiedPercentageTrainSamples.jpg "misclassfiedPercentageTrainSamples"
[image11]: ./examples/FeatureMapsVis.jpg "FeatureMapsVis"
[image12]: ./examples/weightsVis.jpg "weightsVis"
[image13]: ./examples/activationsVis.jpg "activationsVis"
[image14]: ./examples/misclassfiedTestSamples.jpg "misclassfiedTestSamples"
[image15]: ./examples/misclassfiedPercentageTestSamples.jpg "misclassfiedPercentageTestSamples"
[image16]: ./examples/S1.jpg "S1"
[image17]: ./examples/S2.jpg "S2"
[image18]: ./examples/S3.jpg "S3"
[image19]: ./examples/S4.jpg "S4"
[image20]: ./examples/S5.jpg "S5"
[image21]: ./examples/im1.jpg "im1"
[image22]: ./examples/im2.jpg "im2"
[image23]: ./examples/im3.jpg "im3"
[image24]: ./examples/im4.jpg "im4"
[image25]: ./examples/im5.jpg "im5"


[//]: # (Links)
[link1]: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
[link2]: https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

## Overview

The goal of this project is to build a traffic sign recognition pipeline based on deep neural nets which is trained to classify German traffic signs.

## Project Analysis

The pipeline in 4 main sections which are:

1. Dataset Summary and Exploration
2. Model Design and Training
3. Model Analysis
4. Model Testing

---

### 1. Data set Summary and Exploration

In this section, the dataset is loaded and analyzed.The data is divided into three chunks which are testing, validation and testing as follows:

* **Training Set:   34799 samples**
* **Validation Set: 4410 samples**
* **Test Set:       12630 samples**


Each sample is an RGB colored image with **size (32,32,3)** and the number of classification classes are **43 classes**.

![alt text][image1]


The data is further analyzed to show the samples distribution(y-axis) per each class(x-axis) as follows:

Training Data
![alt text][image2]

Validation Data
![alt text][image3]

Testing Data
![alt text][image4]

---

### 2. Model Design and Training

In this section the model architecture is designed and trained using processed training data.

#### Training data PreProcessing

First, looking at the training data analysis we can see that the samples are biased where some classes have more samples than others which affects the training performance.Therefore, synthesized data is added by rotating, translating and blurring the original training dataset in order to have nearly unbiased distribution of the training data keeping the validation data and testing data unchanged.

* **Training Set:   63478 samples**
* **Validation Set: 4410 samples**
* **Test Set:       12630 samples**

Training Data Synthesized
![alt text][image5]

Finally, the training data is preprocessed by transforming the images from RGB to GRAYSCALE which would help in decreasing the complication of the training process and make the network learn better.

![alt text][image6]

Then the data is normalized to have around zero mean and equal standard deviation making the dataset well conditioned which will help the gradient descent to minimized the error efficiently.

#### Model Architecture

The Model Architecture is a modified version of the Lenet architecture. 

Lenet Architecture
![alt text][image7]

The Modification is inspired from the paper [*(Traffic Sign Recognition with Multi-Scale Convolutional Networks by Pierre Sermanet and Yann LeCun)*][link1]  which is applied by taking the output of the subsampled result **S2** (pooled layer) of the first convolution layer **C1** and passing it to the fully connected layer **C5** changing its input size from *16@5x5 = 400* to *16@5x5 + 6@14x14 = 1576*.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| outputs 120        							|
| Fully connected		| outputs 84        							|
| Fully connected		| outputs 43        							|
| Softmax				| outputs 43        							|



#### Training HyperParameters

The training HyperParameters are configured as follows:

* Learning Rate : 0.001
* Epochs : 15
* Batch Size : 80
* Dropout: 0.5
* Optimizer : Adam Optimizer

The Learning Rate (0.001) is chosen a low numerical value in order to avoid overshooting.

The number of Epochs (15) is chosen based on experimenting where this number of epochs yields the saturation accuracy and beyond it the accuracy starts to oscillate.

The Batch size (80) is chosen to have at least two samples for each label per batch.

The Optimizer (Adam Optimizer) is used as it performs learning rate decay to prevent overshooting and momentum to average the gradient from all the batches yielding a good gradient descent.

#### Training Results

The Training process yielded **Training Data Accuarcy of 0.993** and **Validation data Accuarcy 0.963**

The graph below shows the Training Accuracy (red line) and Validation Accuracy (blue line)
![alt text][image8]

---
### 3. Model Analysis

In this section the trained model performance is analyzed by means of data analysis and visualizations.The Analysis gives insight about what the model actually learned which provides more insight than the validation accuracy metric.

#### Training Performance

The trained network is further analyzed to see if there a class in the training data that is not well learned by the model.This is achieved by computing the misclassified training data samples for each label then getting their percentage from the whole data samples for each label.

MisClassfied Training data Samples
![alt text][image9]

MisClassfied Training Data Samples Percentage
![alt text][image10]

#### Visualize FeatureMaps

Visualizing the feature maps  get the output of a weight layer by passing as input a stimuli image through a trained layer and visualize it.

The output shows if the network learned well or not, some training trials could yield high validation accuracy but half of the feature maps didnt learn anything which in turn yields bad classification in the test set.Therefore, the visualized output of the feature maps can be used as an indicator of the training performance.

The visualization is done for the first convolution layer and the output for a stimuli image is shown below.
![alt text][image11]

The result shows that the network learns to detect curves and lines and the various combinations of those detections classify the image.

#### Visualize Weights

Visualizing the weights shows the normalized weights of a certain layer as a an image which highlights the weights which has the highest values hence highest effect also the ones which has the least effect.

Having redundancy and randomness in the weights visualization indicates that network is not well trained.
 
The visualization is done for the first convolution layer's weights.
![alt text][image12]

#### Visualize Activations

Inspired by [(Visualizing and Understanding Convolutional Networks by Matthew D. Zeiler and Rob Fergus)][link2] the first layer convolution network is visualized by passing a stimuli image and getting the output feature maps, then the feature maps are DeConvlouded where for each feature map the weights of all others is set to zero except for this feature map resulting in an image having the section of the image which this feature map detects.

The result of this visualization shows what each feature map is good in classifying and provides more insight of how the convlutional neural networks work.

Result for the first convloution layer
![alt text][image13]

---
### 4. Model Testing

In this section the trained model performance is tested on the test dataset showing how the train works with new data.

#### Full TestData Evaluation

Evaluating the whole test dataset by the trained network yields a **test accuary of 0.932**

By doing further analysis to see the second best accuracy results on the test data for each label the results are as follows.

MisClassfied Test data Samples
![alt text][image14]

MisClassfied Test Data Samples Percentage
![alt text][image15]

#### Sample TestData Evaluation

5 German signs where randomly chosen from the test dataset and their results are deeply analyzed

The signs are shown below

![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20]

The Results for Predections are:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory  | Roundabout mandatory   						| 
| Speed limit (120 km/h)| Speed limit (120 km/h)						|
| Traffic Signals		| Traffic Signals								|
| Yield	      			| Yield					 						|
| No Passing			| No Passing	      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

For the 5 images, the model is relatively sure of its results.This is shown in the results below for each of the images.

Image 1 (Roundabout mandatory)
![alt text][image21]

Image 2 (Speed limit (120 km/h)
![alt text][image22]

Image 3 (Traffic Signals)
![alt text][image23]

Image 4 (Yield)
![alt text][image24]

Image 5 (No Passing)
![alt text][image25]"
