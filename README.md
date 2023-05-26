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

### 3. Recreate our experiments

#### 3.2 Use reconstructed feature to train a classifier
First, we need to train the vae and save the model

```bash
python /home/gabb/egovision_project/aml22-ego/train_VAE_features_clip.py action="train"  name="VAE_FT_D_16f" \
  config=configs/VAE_save_feat.yaml dataset.shift=D1-D1 wandb_name='vae' wandb_dir='Experiment_logs'  \
  dataset.RGB.data_path=../ek_data/frames dataset.RGB.features_name='EPIC/FT_D_D1_16f_5c' models.RGB.model='VAE'
```

```bash
python train_VAE_features_clip.py action="save" name="VAE_FT_D_16f"   config=configs/VAE_save_feat.yaml   dataset.shift=D1-D1   wandb_name='vae'  wandb_dir='Experiment_logs'  dataset.RGB.data_path=../ek_data/frames    dataset.RGB.features_name='EPIC/FT_D_D1_16f_5c'  models.RGB.model='VAE' resume_from='saved_models/VAE_RGB/VAE_FT_D_16f_lr0.0001_1.pth'
```

Or, using the same command:

```bash
python train_VAE_features_clip.py action="train_and_save" name="VAE_RGB" config=configs/VAE_save_feat.yaml   dataset.shift=D1-D1   wandb_name='vae-rgb'  wandb_dir='Experiment_logs'  dataset.RGB.data_path=../ek_data/frames    dataset.RGB.features_name='EPIC/FT_D_D1_16f_5c'  models.RGB.model='VAE'
```

The second option is preferable, because we don't need to provide the path of the model to resume from.


```bash

Once we have the reconstructed features, we can train the classifier using them as input.

```bash

python train_classifier_TRN.py action=train config=configs/train_classifier.yaml \
            dataset.shift=D1-D1 models.RGB.model=action_TRN num_clips=5 \
            name='test_reconstructed_feature' dataset.RGB.data_path="../ek_data/frames/" \
            wandb_name=TRN_reconstructed dataset.RGB.features_name='reconstructed/VAE_0.001_2023-05-26 10:41:49.665145'
```

The performace of the model are comparable with the one that use the original features.

ALERT: Per qualche strano motivo, funziona solo lr = 0.001, non so perché, nonostante i loss siano abbastanza simili.

lr = 10^-4: brutti risultati
lr = 10^-3: buoni risultati
lr = 10^-2: bruttissimi risultati

Bisogna capire il perchè solo con lr = 10^-3 funziona. Reduction = sum, non funziona con reduction = mean.
