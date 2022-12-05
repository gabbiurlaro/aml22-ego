#!/bin/bash

for d in D1
do
  python train_classifier.py name=action_classifier${d} dataset.shift=${d}-${d} gpus=0
done

for d in D1
do
  for s in D1
    do
      n=action_classifier${d}
      python train_classifier.py action=validate name=${n} dataset.shift=${d}-${s} gpus=0
    done
done