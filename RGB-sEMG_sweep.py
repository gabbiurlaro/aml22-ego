"""
RGB to sEMG translation(feature level) -> sweep
"""

from utils.logger import logger
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch
from utils.loaders import ActionNetDataset, EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import numpy as np
import os
import models as model_list
import wandb
from utils.utils import costant_scheduler, frange_cycle_linear, frange_cycle_sigmoid
import matplotlib.pyplot as plt
from  sklearn.manifold import TSNE
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)

def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    
    if args.wandb_name is not None:
        WANDB_KEY = "c87fa53083814af2a9d0ed46e5a562b9a5f8b3ec" # Salvatore's key
        if os.getenv('WANDB_KEY') is not None:
            WANDB_KEY = os.environ['WANDB_KEY']
            logger.info("Using key retrieved from enviroment.")
        wandb.login(key=WANDB_KEY)
        run = wandb.init(project="FC-VAE(RGB-sEMG)", entity="egovision-aml22")
        wandb.run.name = f'RGB-sEMG_lr{wandb.config.lr}_b{wandb.config.beta}'

def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    args.models.vae.lr = wandb.config.lr
    args.models.vae.beta = wandb.config.beta
    args.models.lr_steps = wandb.config.lr_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    model = getattr(model_list, args.models.vae.model)(args.train.in_feature_size, 
                                                              args.train.bottleneck_size, 
                                                              args.train.out_feature_size,
                                                              resume_from=args.last_model)

    
    if args.action == "train":
        # Using both modalities
        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], 
                                                                    modalities,
                                                                    'train',
                                                                    args.dataset, 
                                                                    args.train.num_frames_per_clip, 
                                                                    args.train.num_clips,
                                                                    args.train.dense_sampling,
                                                                    transform = None, 
                                                                    load_feat=True, 
                                                                    require_spectrogram=True),
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], 
                                                                  modalities,
                                                                  'test',
                                                                   args.dataset, 
                                                                   args.train.num_frames_per_clip,
                                                                   args.train.num_clips,
                                                                   args.train.dense_sampling,
                                                                   transform = None,
                                                                   load_feat=True),
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        autoencoder = train(model, train_loader, val_loader, device, args.models.vae)
        timestamp = datetime.now()
        model_filename = f"{args.name}_lr{args.models.vae.lr}_{timestamp}.pth"
        save_model(autoencoder, model_filename)
        logger.info(f"Model saved in {model_filename}")
    elif args.action == "save":
        if args.last_model is None:
            raise ValueError("You must specify a model to load from")
        print(args.last_model)
        model.load_last_model()
        model.load_on(device)
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], 
                                                                modalities,
                                                                'train', 
                                                                args.dataset,
                                                                args.save.num_frames_per_clip, 
                                                                args.save.num_clips, 
                                                                args.save.dense_sampling,
                                                                load_feat=True, 
                                                                additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        
        loader_test = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], 
                                                                    modalities,
                                                                    'test',
                                                                    args.dataset,
                                                                    args.save.num_frames_per_clip, 
                                                                    args.save.num_clips, 
                                                                    args.save.dense_sampling,
                                                                    load_feat=True, 
                                                                    additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        timestamp = datetime.now()
        logger.info(f"Reconstructing features...")
        filename = f"features_lr{args.models.EMG.lr}_b{args.models.EMG.beta}_{timestamp}"
        reconstructed_features, output = reconstruct(model, loader, device, "train", save = True, filename=filename, debug=True)
        logger.debug(f"Train Output {output}")
        reconstructed_features, output = reconstruct(model, loader_test, device, "test", save = True, filename=filename, debug=True)
        logger.debug(f"Test Output {output}")
    elif args.action == "train_and_save":
        model.load_last_model()
        model.load_on(device)
        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                                'train', args.dataset,  None, args.train.num_clips, None,
                                                                                load_feat=True, require_spectrogram=True),
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                                'test', args.dataset,  None, args.train.num_clips, None,
                                                                                load_feat=True, require_spectrogram=True),
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=args.dataset.workers, pin_memory=True, drop_last=False)    
        
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset,  None, args.train.num_clips, None,
                                                                       load_feat=True, additional_info=True, require_spectrogram=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        
        loader_test = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'test', args.dataset,  None, args.train.num_clips, None,
                                                                       load_feat=True, additional_info=True,require_spectrogram=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        timestamp = datetime.now()

        ae = train(model, train_loader, val_loader, device, args.models.vae)
        model_filename = f"{args.name}_lr{args.models.vae.lr}_b{args.models.vae.beta}_{timestamp}.pth"
        save_model(ae, model_filename)
        logger.info(f"Model saved in {model_filename}")
        logger.info(f"TRAINING VAE FINISHED, RECONSTUCTING FEATURES...")
        filename = f"features_lr{args.models.vae.lr}_b{args.models.vae.beta}_{timestamp}"
        reconstructed_features = reconstruct(model, loader, device, "train", save = True, filename=filename, debug = True)
        reconstructed_features = reconstruct(model, loader_test, device, "test", save = True, filename=filename)
    else:
        raise NotImplementedError(f"Action {args.action} not implemented")

def train(autoencoder, train_dataloader, val_dataloader, device, model_args):
    logger.info(f"Start VAE training.")
    autoencoder.train(True)

    opt = build_optimizer(autoencoder, "adam", model_args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=model_args.lr_steps, gamma=model_args.lr_gamma)

    reconstruction_loss = nn.MSELoss(reduction='mean')

    beta = costant_scheduler(model_args.beta, model_args.epochs)
    for epoch in range(model_args.epochs):
        # train_loop
        total_loss = 0 # total loss for the epoch
        for i, (data, _) in enumerate(train_dataloader):
            opt.zero_grad()                                                                 #  reset the gradients    
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)                                              #  Data is now in the form (clip, batch, features)            
            
            for i_c in range(args.test.num_clips):
                clip_level_loss = 0                                                         #  loss for the clip             
                # extract the clip related to the modality
                rgb_clip = data['RGB'][i_c].to(device)
                emg_clip = data['EMG'][i_c].to(device)

                x_hat, _, mean, log_var = autoencoder(rgb_clip)
                mse_loss = reconstruction_loss(x_hat, emg_clip)                              #  compute the reconstruction loss
                kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  #  compute the KLD loss
                loss = mse_loss + beta[epoch] * kld_loss
                # generate an error if loss is nan
                if loss.isnan():
                    raise ValueError("Loss is NaN.")
                clip_level_loss += loss
                loss.backward()
                wandb.log({"Beta": beta[epoch], "MSE LOSS": mse_loss, 'KLD_loss': kld_loss, 'loss': loss, 'lr': scheduler.get_last_lr()[0]})
            # update the weights at the end of the batch
            opt.step()

            total_loss += clip_level_loss.item()
        if epoch % 10 == 0:
            wandb.log({"validation_loss": validate(autoencoder, val_dataloader, device, reconstruction_loss)})
        logger.info(f"[{epoch+1}/{model_args.epochs}] - Total loss: {total_loss}")
        wandb.log({"train_loss": total_loss})
        scheduler.step()
    logger.info(f"VAE training finished.")
    return autoencoder

def reconstruct(autoencoder, dataloader, device, split=None, **kwargs):
    """
    Reconstruct the features using the trained autoencoder
    - autoencoder: the trained autoencoder
    - dataloader: the dataloader to use
    - device: the device to use
    - split: the split to use
    - kwargs: additional arguments
    """
    filename = kwargs.get('filename', "reconstructed_features_EMG")
    result = {'features': []}
    # for debugging purpose, I introduce also a loss in reconstruction
    reconstruction_loss = nn.MSELoss()
    avg_video_level_loss = 0
    with torch.no_grad():
        for i, (data, label, video_name, uid) in enumerate(dataloader):
            autoencoder.train(False)
            for m in modalities:
                data[m] = data[m].squeeze(1).permute(1, 0, 2)     #  clip level
            
            clips = []
            clip_loss = 0
            for i_c in range(args.test.num_clips): #  iterate over the clips
                rgb_clip = data['RGB'][i_c].to(device)     #  retrieve the clip
                x_hat, _, _, _ = autoencoder(rgb_clip)     
                clips.append(x_hat)
            clips = torch.stack(clips, dim = 0)
            clips = clips.permute(1, 0, 2)
            clips = clips.squeeze(0)
            result['features'].append({
                    'features_EMG': clips.cpu().numpy(), 
                    'label': label.item(), 
                    'uid': uid.item(), 
                    'video_name': video_name
    })
    try:
        date = str(datetime.now().date())
        if not os.path.isdir(os.path.join('./saved_features/translated/', date, 'sweep')):
            os.mkdir(os.path.join('./saved_features/translated/', date, 'sweep'))
        with open(os.path.join('./saved_features/translated/', date, 'sweep',f"{filename}_{args.dataset.shift.split('-')[0]}_{split}.pkl"), "wb") as file:
            pickle.dump(result, file)
        logger.info(f"Saved {filename}_{args.dataset.shift.split('-')[0]}_{split}.pkl")
    except Exception as e:
        logger.warning(f"Error while saving the file: {e}")
    
    return result

def validate(autoencoder, val_dataloader, device, reconstruction_loss):
    total_loss = 0
    autoencoder.train(False)
    for i, (data, labels) in enumerate(val_dataloader):
        for m in modalities:
            # logger.info(f"Data size: {data[m].squeeze(1).shape}")
            data[m] = data[m].squeeze(1).permute(1, 0, 2).to(device)
            # print(f"Data after permutation: {data[m].size()}")
        for i_c in range(args.test.num_clips):
            # extract the clip related to the modality
            rgb_clip = data['RGB'][i_c].to(device)
            emg_clip = data['EMG'][i_c].to(device)
            x_hat, _, mean, log_var = autoencoder(rgb_clip)
            mse_loss = reconstruction_loss(x_hat, emg_clip)
            kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = mse_loss + kld_loss
            total_loss += loss
    return total_loss/len(val_dataloader)

def save_model(model, filename):
    try:
        date = str(datetime.now().date())
        if not os.path.isdir(os.path.join('./saved_models/RGB_sEMG', date)):
            os.mkdir(os.path.join('./saved_models/RGB_sEMG', date))
        torch.save({'encoder': model.encoder.state_dict(), 'decoder': model.decoder.state_dict()}, 
                   os.path.join('./saved_models/RGB_sEMG', date, filename))
    except Exception as e:
        logger.info("An error occurred while saving the checkpoint:")
        logger.info(e)

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

if __name__ == '__main__':
    main()
