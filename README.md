## Introduction
This repository contains a comprehensive project for detecting fake news using machine learning techniques and various natural language processing techniques. The project includes data analysis, model training, and a web application for real-time fake news detection. The machine learning model is designed to classify news articles as either real or fake based on their content.

## Problem Definition
We aim to develop a machine learning program to identify when a news source may be producing fake news. The model will focus on identifying fake news sources, based on multiple articles originating from a source. Once a source is labeled as a producer of fake news, we can predict with high confidence that any future articles from that source will also be fake news. Focusing on sources widens our article misclassification tolerance, because we will have multiple data points coming from each source.

The intended application of the project is for use in applying visibility weights in social media. Using weights produced by this model, social networks can make stories that are highly likely to be fake news less visible.

<img width="2514" height="1360" alt="image" src="https://github.com/user-attachments/assets/75ed895b-95e2-4873-a134-29e00e3198eb" />



## Model Name
The machine learning model used for fake news detection in this project is the **Passive Aggressive Classifier**.

### Model Description
The Passive Aggressive Classifier (PAC) is a type of online learning algorithm for binary classification tasks. It is well-suited for applications like fake news detection. The PAC algorithm updates its model continuously as new data arrives, making it efficient for real-time classification.

<img width="2534" height="1352" alt="image" src="https://github.com/user-attachments/assets/b60f0dd2-5033-4452-8105-d0a9fd307372" />



### Model Accuracy
The Passive Aggressive Classifier achieved an impressive accuracy of **96%** during evaluation. This high accuracy indicates its effectiveness in classifying news articles as reliable or unreliable.

<img width="2532" height="1358" alt="image" src="https://github.com/user-attachments/assets/9e752fc4-e9bd-47fd-9855-d9f51e014f5f" />


The model is pre-trained and available as `model.pkl` in this repository, allowing you to use it for making predictions.

Feel free to explore the Jupyter Notebook (`Fake_News_Detector-PA.ipynb`) for more details about the model's training and performance.


