# Animal Species Detection

This repository contains code for an animal species detection project. 
The project uses a pre-trained deep learning model, specifically ResNet50, 
to predict the species of an animal from an input image.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)

## Introduction

The Animal Species Detection project aims to classify images of animals into their respective species using deep learning techniques. It utilizes the ResNet50 model, which is pre-trained on the ImageNet dataset, to make accurate predictions. By providing an input image, the code will preprocess the image, pass it through the ResNet50 model, and output the top three predictions along with their respective probabilities. Additionally, it displays the input image with matplotlib.

## Installation

To run the animal species detection code, follow these steps:

1. Clone the repository:
git clone https://github.com/akashdipsaha/animal-species-detection.git


2. Install the required dependencies:
pip install -r requirements.txt


## Usage

1. Place the image you want to classify in the repository's root directory.

2. In the code, update the `image_path` variable with the file path of the image you want to classify:
```python
image_path = r'path/to/your/image.jpg'

Run the predict_animal_species function:
python animal_species_detection.py

The code will display the top three predictions for the animal species along with their probabilities, and it will also show the input image.

Examples
Here are a few examples of how to use the animal species detection code:
example1:
image_path = r'path/to/your/image1.jpg'
predict_animal_species(image_path)
example2:
image_path = r'path/to/your/image2.jpg'
predict_animal_species(image_path)

