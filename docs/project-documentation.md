## Problem statement

### Waste classification (WS)

The objective of this project is to design, train, and evaluate an efficient and accurate image based waste classification model that can serve as a foundation for automated recycling and smart waste management systems.

The project focuses on the task of multi class garbage image classification using deep learning techniques. Given an input image containing a waste object, the model aims to automatically assign the object to one of several predefined waste categories such as plastic, paper, glass, or metal.

### Problem summary

* **What are we trying to do:** identify and classify waste objects from images.
* **What's the objective:** design, train, and evaluate an accurate and efficient image based classification model.
* **How are we evaluating this:** validation accuracy and validation loss.  
* **Why are we solving this problem:** automated waste classification systems based on computer vision can improve recycling efficiency and reduce environmental impact.

### Exploration

A detailed exploratory notebook describing the experiments and modeling decisions can be found at [here](../notebook.ipynb).

## Dataset description

The dataset used in this project is a publicly available garbage image classification dataset obtained from Kaggle. It consists of labeled images of waste objects belonging to multiple material categories including cardboard, glass, metal, paper, plastic, and trash. Each image contains a single primary waste object captured under certain conditions such as lighting, backgrounds, and viewpoints.

The dataset is organized into class specific directories, where each directory corresponds to a waste category. The dataset is divided into three separate subsets: training, validation, and testing. Each subset contains 2,527 images, resulting in a total of 7,581 images across all splits.

Data source: https://www.kaggle.com/datasets/hassnainzaidi/garbage-classification
  

## Dataset

### Dataset Overview

Each of the training, validation, and test subsets contains 2,527 images distributed across six waste categories. The class distribution within each subset is shown below.

Class | Image count
---|---
Cardboard | 403
Glass | 501
Metal | 410
Paper | 594
Plastic | 482
Trash | 137
Total | 2,527

### Class imbalance

The dataset exhibits class imbalance, with certain categories containing significantly fewer images than others. In particular, the Trash class is under represented compared to classes such as Paper and Glass.

Since the same class distribution is present in each data split, the imbalance is consistent across training, validation, and testing subsets. This imbalance may lead to biased learning, where the model achieves high overall accuracy while performing less effectively on minority classes. As a result, overall accuracy alone may not fully reflect performance across all waste categories.


## Modeling approach & metrics

The proposed approach leverages transfer learning using a MobileNetV2 convolutional neural network pre trained on ImageNet. The backbone network is used as a feature extractor, with most layers frozen during training. A compact task specific classification head is trained on top of the extracted features to adapt the model to the waste classification domain.

Data augmentation techniques are applied during training to improve robustness and generalization. Model performance is primarily evaluated using validation accuracy and validation loss.

### Model Evaluation Summary

The final model configuration employs a MobileNetV2 backbone with a lightweight custom classifier head of size ten. This configuration provided the best balance between model capacity and generalization performance.

Evaluation results are summarized as follows.

Training accuracy 99.0 percent
Validation accuracy 99.8 percent
Validation loss 0.0072

These results indicate that the model achieves strong classification performance while maintaining a low validation loss, suggesting effective learning without significant overfitting.

Observations

A compact classifier head significantly outperformed larger configurations, indicating that the extracted MobileNetV2 features are highly discriminative for the waste classification task.

Transfer learning and data augmentation played a key role in improving generalization, particularly given the limited size of the dataset.

Dropout was evaluated as a regularization technique but was excluded from the final model due to underfitting and reduced validation accuracy.


## Known limitations / next steps

Although the final model achieved high validation accuracy, this performance raises an important question regarding potential dataset bias. Specifically, the model may be learning contextual features related to image acquisition conditions rather than purely object specific characteristics.

High performance on a validation set drawn from the same data distribution does not necessarily guarantee robustness in real world scenarios. Factors such as consistent backgrounds, lighting conditions, and class specific capture environments may influence the model predictions.

These limitations do not invalidate the results of the project. Instead, they highlight the need for future work, including evaluation on more diverse real world images, analysis of class specific performance, and the use of model explainability techniques to better understand the decision making process.
