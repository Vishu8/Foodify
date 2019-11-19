# Foodify Image Classifier

## Overview

This is the Machine Learning Image Classification files from which 6 classes of food packets are trained.

## Requirements

To execute machine learning model install all in requirements.txt

    `pip install -r requirements.txt`

## Dataset

The `packed_food_dataset` is the dataset of this project. It consists of 6 classes:

1. balaji_waffers
2. krackjack_biscuit
3. lays_waffers
4. maggie_noodles
5. parleg_biscuit
6. pepsi_can

## Run ML Files

To train the model run the `train.py` with the following command:

    `python train.py --image_dir packed_food_dataset --how_many_training_steps NO_OF_TRAINING_STEPS`

To train the model run the `test.py` with the following command:

    `python test.py test_food.jpg`

To see the training logs in Tensorboard run the following command:

    `tensorboard --logdir train_files/train_logs`
