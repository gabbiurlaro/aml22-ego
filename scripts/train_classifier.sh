#!/bin/bash

for feature_type in FT PT
  do
    for sampling in D U 
    do
      for frames in 8 16
      do
          python train_classifier_LSTM.py action=train config=configs/train_classifier.yaml \
            dataset.shift=D1-D1 models.RGB.model=ActionLSTM num_clips=5 \
            name=${feature_type}_${sampling}_D1_${frames}f_5c dataset.RGB.data_path="../ek_data/frames/" \
            wandb_name=LSTM wandb_dir=LSTM_dir dataset.RGB.features_name=EPIC/${feature_type}_${sampling}_D1_${frames}f_5c
      done
    done
  done