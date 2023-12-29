# Machine Learning Project for Enterprise Risk Prediction
Welcome to the repository for the machine learning project dedicated to predicting enterprise risk. This project was developed as part of a course to apply fundamental concepts of machine learning and data analysis.

## Repository Structure
This folder contains all the source code and resources related to the enterprise risk prediction project. You will find Jupyter notebooks, Python scripts, datasets, and other relevant files.

## Project Objective
The main goal of this project is to develop a machine learning model capable of predicting the risk associated with an enterprise. Using financial and operational data, the model aims to assess the probability that the company poses an increased risk.

## Required Packages
To run the file you need those specific packages.
To install the necessary packages from the [requirements.txt](https://github.com/Marius739/Machine-Learning/blob/main/Project_Risk/requirements.txt) file you need this command line

if you are using venv from Anaconda :
```bash
$ conda install --file requirements.txt
```
if you are using pip :
```bash
$ pip install -r requirements.txt
```
## Script and Notebook
To run the .py file : 
```bash
$ python predict_risk.py --output predictions.csv
```
You can replace "predictions.csv" with another name. Note that the file will be created in the same file as the script
For the notebook 

You will also find the [training](https://github.com/Marius739/Machine-Learning/blob/main/Project_Risk/training_dataset.csv) and the [final test](https://github.com/Marius739/Machine-Learning/blob/main/Project_Risk/final_test.csv) datasets for the model. 
