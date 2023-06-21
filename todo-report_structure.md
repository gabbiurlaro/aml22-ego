# TODO
 - [ ] Classification RGB-EK (report results and considerations, save pictures and test)
  - [ ] Final Classifier (use train_classifier_multimodal)
  - [ ] TRN
  - [ ] LSTM
  - [ ] late fusion
  - [ ] early fusion
- [ ] Classification MM-EK ()
  - [ ] Sweep 
- [ ] Classification per clip
  - [ ] Final classifier MM-EK
  - [ ] Final classifier EMG-EK
  - [ ] Final classifier RGB-EK
- [ ] Report
  - [ ] Run everything for the last time
- [ ] Clean Code
  - [ ] Comments and svarioni
  - [ ] Remove uneseful and Merge
  - [ ] Merge into main


# Report structure
- [ ] Abstract
1. Introduzione
   - [] Metodo
   - [] Dataset(generalities)
2. Related Works
   - [x] Temporal aggregation 
   - [x] Multimodal Classification
   - [ ] ActionNet e TRN
   - [x] VAE  
3. Methods
   - [x] Classifications RGB (TRN, LSTM, early, late)
     - [x] Sampling strategies
     - [x] Clips and video levels
   - [x] Pre-processing EMG 
     - [x] Spectrogram
     - [x] Feature Extraction (citare svarioni con Augmentations e vari tentativi fatti)
   - [x] VAE
     - [x] VAE-RGB, reconstruction loss to pretrain encoder
     - [x] VAE-EMG, reconstruction loss to pretrain decoder
     - [x] RGB->VAE->EMG traslation
4. Experimental results
   - [x] Datasets(classes, number of samples, etc)
     - [x] Shapes
     - [x] Contents
   - [x] Classification RGB / FeatsEX analysis
   - [x] Classification EMG / FeatsEX analysis
   - [x] VAE RGB | VAE EMG reconstructions (features plots, post-reconstruction classification results)
   - [x] Translation + Multimodal Classification
5. Conclusions
