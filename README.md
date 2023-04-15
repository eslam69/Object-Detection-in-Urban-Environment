# Object Detection in Urban Environment
 ![EfficientNet](media/animation.gif)
## Table Of Contents
- [Object Detection in Urban Environment](#object-detection-in-urban-environment)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Methodology](#methodology)
    - [Training \& Deployment Process with AWS](#training--deployment-process-with-aws)
    - [Model Selection](#model-selection)
    - [Results](#results)
  - [Future Work \& Possible Improvement](#future-work--possible-improvement)

## Introduction
Utilizing transfer learning using TensorFlow object detection API and AWS Sagemaker to train models to detect and classify objects using data from the Waymo Open Dataset.

## Dataset
Front Camera Images from [Waymo Open Dataset](https://waymo.com/open/). 
Data are in **TFRecord** Format, the TFRecord format is a simple format for storing a sequence of binary records, which helps in data reading and processing efficiency.

## Methodology 
### Training & Deployment Process with AWS
- AWS Sagemaker for running Jupyter notebooks, training and deploying the model, and inference.
- AWS Elastic Container Registry (ECR) to build the Docker image and create the container required for running this project.
- AWS Simple Storage Service (S3) to save logs for creating visualizations. Also, the data for this project was stored in a public S3 bucket.

### Model Selection
For this project, I tested several object detection models using the Tensorflow Object Detection API. The models tested were:

|         **Model**        	|  **Config**  	|
|:------------------------:	|:------------:	|
| EfficientNet D1          	| [file](model1-effecientNet\pipeline.config) 	|
| SSD MobileNet V2 FPNLite 	| [file](model2-mobileNet\pipeline.config) 	|
| SSD ResNet50 V1 FPN       | [file](model3-Resnet\pipeline.config) 	|

These pre-trained models are available in the TensorFlow 2 Object Detection Model Zoo, and they were trained on the COCO 2017 dataset. 
So,  their `pipeline.config` files need to be adjusted so TensorFlow 2 can find the `TFRecord` and `label_map.pbtxt` files when they are loaded inside the container from Amazon S3.
 
Since the Waymo dataset has only 3 classes, Cars, Pedestrians, and Cyclists, the `pipeline.config` was adjusted to our problem instead of the 90 classes that were there for the COCO dataset.

For the 3 Models I used a fixed number of training steps which is **2000**, this was due to my limited AWS budget.

Also, I used Momentum Optimizer with the same batch size of **8** in the 3 experiments, for the same reason.

### Results
Each model was evaluated using the mAP metric, which measures the accuracy of the model in detecting objects. The mAP is calculated based on the precision and recall of the model at different IoU (Intersection over Union) thresholds.

Tensorboard was used to visualize the training loss and validation mAP for each model. From the Tensorboard graphs, we observed that the models showed similar patterns in terms of training loss, but differed in their ability to generalize to the test data.

|                                       	|                        EfficientNet D1                       	|                 SSD MobileNet V2 FPNLite               	|                   SSD ResNet50 V1 FPN                 	|
|:-------------------------------------:	|:------------------------------------------------------------:	|:------------------------------------------------------:	|:-----------------------------------------------------:	|
|        **mAP@ (0.5:0.95) IOU**        	|                             0.0938                           	|                         **0.09543**                    	|                         0.05755                       	|
|             **mAP@.50IOU**            	|                             **0.2253**                        |                          0.2234                        	|                         0.1248                        	|
|             **mAP@.75IOU**            	|                             0.0668                           	|                          **0.071**                        |                         0.04505                       	|
|        **mAP (small objects)**        	|                            0.01484                           	|                          **0.0392**                       |                         0.02317                       	|
|        **mAP (medium objects)**       	|                             **0.364**                         |                          0.3383                        	|                         0.2107                        	|
|        **mAP (large objects)**        	|                             **0.839**                         |                          0.4531                        	|                         0.1917                        	|
| **Predicted Vs Ground Truth Sample** 	|     ![EfficientNet](media/Efficient_side_by_side_png.png)    	|     ![MobileNet](media/Mobile_side_by_side_png.png)    	|     ![ResNet50](media/RESNET_side_by_side_png.png)    	|
|               **Video**               	|           ![EfficientNet](media/output1.gif)               	|            ![MobileNet](media/output2.gif)           	|         ![ResNet50](media/output3.gif)              	|

Based on the results of the three models evaluated for object detection in an urban environment, the SSD MobileNet V2 FPNLite model performed the best with an mAP@(0.5:0.05:0.95) IOU of 0.09543, outperforming both the EfficientNet D1 and SSD ResNet50 V1 FPN models.

In terms of detecting small objects like cyclists and pedestrians, the SSD MobileNet V2 FPNLite also had the highest mAP, indicating its ability to detect smaller objects better than the other models. However, the EfficientNet D1 had the highest mAP for large objects, suggesting that it may perform better in detecting larger objects like e.g nearby cars.

The three models had poor performance in detecting cyclists. This may be a result due to the skewness of the dataset, where cars are the dominant class in the dataset, and the cyclists class is the least abundant.

Overall, the model selection process showed that different models have different strengths and weaknesses in object detection, and choosing the right model for a specific application requires careful consideration of the type and size of the objects to be detected. Additionally, the results suggest that the ResNet50 model may not be the best choice for object detection in an urban environment, at least not without further optimization and tuning.



Here are the training losses of the 3 experiments:
![](media/loss.png)

The plots show that the 3 models could achieve better loss if we increased the n. of training steps because there is room for convergence.

## Future Work & Possible Improvement

We identified several potential avenues for improving performance, but they would require additional resources and a higher computing budget. These include:
- Increase the training steps: Each model was trained for only 2000 steps, which is relatively low for such kinds of data and complex architectures. So, increasing the number of training steps till the loss reaches the plateau can further improve performance.
- Data augmentation: techniques such as flipping, scaling, and random cropping. More advanced techniques such as color jittering, rotation, and translation can also be used to improve the model's accuracy.
- Hyperparameter tuning: Fine-tuning the hyperparameters can potentially improve the model's performance.
- Handling occlusion and partial object detection: In this project, we focused on detecting complete objects. However, in an urban environment, objects are often partially occluded or obstructed. Developing techniques to handle partial object detection can further improve the model's performance.
