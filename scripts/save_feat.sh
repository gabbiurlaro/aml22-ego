#!/bin/bash

# this is how to obtain the features of the model finetuned on EPIC-Kitchens
for d in D1
do
  for split in "train" "test"
    do
      python save_feat.py config=configs/I3D_save_feat.yaml dataset.shift=${d}-${d} \
      split=${split} name=save_feat_I3D
    done
done

# this is how to obtain the features of the model pretrained on Kinetics
for d in D1
do
  for split in "train" "test"
    do
      python save_feat.py config=configs/I3D_save_feat.yaml dataset.shift=${d}-${d} \
      split=${split} name=save_feat_I3D
    done
done
