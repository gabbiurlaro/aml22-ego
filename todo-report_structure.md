# TODO
 <font size='100'> Classification RGB-EK (report results and considerations, save pictures and test)
  * Final Classifier (use train_classifier_multimodal)
  * TRN
  * LSTM
  * late fusion
  * early fusion
* Classification MM-EK ()
  * Sweep 
* Classification per clip
  * Final classifier MM-EK
  * Final classifier EMG-EK
  * Final classifier RGB-EK
* Report
  * Run everything for the last time
* Clean Code
  * Comments and svarioni
  * Remove uneseful and Merge
  * Merge into main


# Report structure
* Abstract
1. Introduzione
   * Metodo
   * Dataset
2. Related Works
   * Temporal aggregation 
   * Multimodal Classification
   * ActionNet
   * VAE  
3. Methods
   * Classifications RGB (TRN, LSTM, early, late)
     * Sampling strategies
     * Clips and video levels
   * Pre-processing EMG 
     * Spectrogram
     * Feature Extraction (citare svarioni con Augmentations e vari tentativi fatti)
   * VAE
     * VAE-RGB, reconstruction loss to pretrain encoder
     * VAE-EMG, reconstruction loss to pretrain decoder
     * RGB->VAE->EMG traslation
4. Experimental results
   * Datasets
     * Shapes
     * Contents
   * Classification RGB / FeatsEX analysis
   * Classification EMG / FeatsEX analysis
   * VAE RGB | VAE EMG reconstructions (features plots, post-reconstruction classification results)
   * Translation + Multimodal Classification
5. Conclusions
