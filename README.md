# Starting code for course project of Advanced Machine Learning (AML) 2022
"Multimodal Egocentric Vision" exam project.


## Getting started

You can play around with the code on your local machine, and use Google Colab for training on GPUs. 
In all the cases it is necessary to have the reduced version of the dataset where you are running the code. For simplicity, we inserted just the necessary frames at [this link](https://drive.google.com/drive/folders/1dJOtZ07WovP3YSCRAnU0E4gsfqDzpMVo?usp=share_link).

Before starting to implement your own code, make sure to:
1. read and study the material provided
2. understand how all the scripts are working and interacting
3. get familiar with the structure of the [EPIC-KITCHENS dataset](https://epic-kitchens.github.io/2022), what a sample, a clip and a frame are
4. play around with the code in the template to familiarize with all the tools.

Some scripts do not need to be run (i.e., [train_classifier_scratch.py](./train_classifier_scratch.py)) but are still inserted in the template in order to make the students understand how the baseline models are obtained.

### 1. Local

You can work on your local machine directly, the code which needs to be run does not require heavy computations. 
In order to do so a file with all the requirements for the python environment is provided [here](requirements.yaml), it contains even more packages than the strictly needed ones so if you want to install everything step-by-step just be careful to use pytorch 1.12 and torchvision 0.13. 

### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/).

- Upload all the scripts in this repo.
- Prepare a proper notebook structured as the `train_classifier.py` script.

As a reference, `colab_runner.ipynb` provides an example of how to set up a working environment in Google Colab.

NOTE: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.

### 3. How to recreate the results

In the paper, we have discussed first feature extraction on RGB videos using I3D, later we have investigated the possibility to use another modality(i.e. EMG), and try to use it to make multimodal action recognition. Later, we have discussed the possibility to translate from one modality to another using the Variational Autoencoder framework.

#### 3.1 Feature extraction
All the analysis are present in the paper.

##### 3.1.1 RGB features extraction

In order to extract features, we have used the I3D model based on the inception module. In our analysis we have included different frame numbers, clip numbers and sampling, in order to find the best configuration, both qualitatively and quantitatively. To extract the RGB features on Epic Kitchen Dataset, we have used the following command:


```bash
python save_feat_epic.py name="rgb_features" config=configs/save_feat/epic.yaml split="train"
python save_feat_epic.py name="rgb_features" config=configs/save_feat/epic.yaml split="test"
```

While, to extract RGB features on ActionSense Dataset, we have used the following command:

```bash
python save_feat_actionnet.py
```

It's also possible to extract all the features together by using the following script:

```bash
./scripts/save_epic.sh
```

```bash
./scripts/save_actionnet.sh
```


##### 3.1.2 EMG features extraction
Also for this task we have used different configurations, in order to find the best parameters, in order to achieve good qualitative and quantitative results. We later used this configuration:

```bash
python train_classifier_EMG.py action="job_feature_extraction" \
      name="job_feature_extraction" \
      config=configs/emg/emg_classifier_1.yaml
```

features are validated  both qualitatively and quantitatively(by a classifier). In `plot_utils.py` are present some useful functions to plot the features, we have used it do conduct our analysis.

##### 3.1.3 Use the feature for an action recognition task

In order to train a classifer on the extracted features, we have to prepare a config file in the `configs` folder.

#### 3.2 Train a variational autoencoder

Variational autoencoder is a powerful framework that allow to learn a latent representation of the data. In our case, we have used it to learn a latent representation of the RGB data, and then use it to translate from RGB to EMG. In order to train the VAE, we have used the following command:

```bash
python train_VAE_features_clip.py action="train_and_save"  name="VAE_FT_D_16f"   config=configs/vae/rgb_vae.yaml```
```

```bash
python train_VAE_EMG_features.py action="train_and_save"  name="VAE_FT_D_16f" \
  config=configs/VAE_save_feat.yaml dataset.shift=D1-D1 wandb_name='vae' wandb_dir='Experiment_logs'  \
  dataset.RGB.data_path=../ek_data/frames dataset.RGB.features_name='EPIC/FT_D_D1_16f_5c' models.RGB.model='VAE' split=train
```

Finally, we can train the final VAE, that translate from RGB to EMG, using the following command:

```bash
python RGB_sEMG.py action="train_and_save" name="RGB_sEMG" config=configs/vae/RGB-sEMG.yaml
```

#### 3.2.1 Translate and perform multimodal action recognition

```bash
python RGB_sEMG.py action="simulate" name="RGB_sEMG" config=configs/vae/EMG-epic.yaml
```

Now, we finally perfom multimodal action recognition, using:

```bash
python train_clasifier_multimodal.py config=configs/classifier_multimodal_EPIC.yaml
```

