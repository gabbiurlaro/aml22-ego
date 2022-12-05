#!/bin/bash

for d in D1
do
  python train_classifier_scratch.py name=I3D_SourceOnly${d} dataset.shift=${d}-${d} gpus=0
done

for d in D1
do
  for s in D1
    do
      n=I3D_SourceOnly${d}
      python train_classifier_scratch.py action=test name=${n} dataset.shift=${d}-${s} gpus=1
    done
done