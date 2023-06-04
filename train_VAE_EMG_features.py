from multiprocessing import reduction
from wsgiref import validate
from utils.logger import logger
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch
from utils.loaders import ActionNetDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import wandb
import torchvision.transforms as transforms
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

# with this script we trained and tested FC_VAE.VariationalAutoencoder to reconstruct features from the EMG modality
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
        run = wandb.init(project="FC-VAE(EMG)", entity="egovision-aml22")
        wandb.run.name = f'{args.name}_{args.models.EMG.model}'


def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    # device where everything is run
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        # notice that here, the first parameter passed is the input dimension
        # In our case it represents the feature dimensionality which is equivalent to 1024 for I3D
        #print(getattr(model_list, args.models[m].model)())
        models[m] = getattr(model_list, args.models[m].model)(1024, 256, 1024)

    if args.action == "train":
        # TODO: fiX dataset_config passing during multimodal training
        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, {'EMG': args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': args.train.dense_sampling.EMG},
                                                                       None, load_feat=True, require_spectrogram=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'test', args.dataset, {'EMG': args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': args.train.dense_sampling.EMG},
                                                                       None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        ae = train(models, train_loader, val_loader, device, args.models.EMG)
        logger.info(f"TRAINING VAE FINISHED, SAVING THE MODELS...")
        save_model(ae['EMG'], f"{args.name}_lr{args.models.EMG.lr}_{datetime.now()}.pth")
        logger.info(f"Model saved in {args.name}_lr{args.models.EMG.lr}_{datetime.now()}.pth")

    elif args.action == "save":
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, {'EMG': 32}, 5, {'EMG': False},
                                                                        load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        
        loader_test = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'test', args.dataset, {'EMG': 32}, 5, {'EMG': False},
                                                                       load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        last_model = args.resume_from
        logger.info(f"Loading last model from {last_model}")
        load_model(models['EMG'], last_model)
        logger.info(f"Reconstructing features...")

        filename = f"../drive/MyDrive/reconstructed/AUG_VAE_2050_{args.models.EMG.lr}"
        reconstructed_features, output = reconstruct(models, loader, device, "train", save = True, filename=filename, debug=True)
        logger.debug(f"Train Output {output}")
        reconstructed_features, output = reconstruct(models, loader_test, device, "test", save = True, filename=filename, debug=True)
        logger.debug(f"Test Output {output}")

    elif args.action == "train_and_save":
        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                                'train', args.dataset, {'EMG': 32}, 5, {'EMG': False},
                                                                                load_feat=True, require_spectrogram=True),
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                                'test', args.dataset, {'EMG': 32}, 5, {'EMG': False},
                                                                                load_feat=True, require_spectrogram=True),
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=args.dataset.workers, pin_memory=True, drop_last=False)    
        
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, {'EMG': 32}, 5, {'EMG': False},
                                                                       load_feat=True, additional_info=True, require_spectrogram=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        
        loader_test = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'test', args.dataset, {'EMG': 32}, 5, {'EMG': False},
                                                                       load_feat=True, additional_info=True,require_spectrogram=True),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        timestamp = datetime.now()

        ae = train(models, train_loader, val_loader, device, args.models.EMG)

        save_model(ae['EMG'], f"{args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
        logger.info(f"Model saved in {args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
        logger.info(f"TRAINING VAE FINISHED, RECONSTUCTING FEATURES...")
        filename = f"reconstructed_features"
        reconstructed_features, results = reconstruct(models, loader, device, "train", save = True, filename=filename, debug = True)
        logger.debug(f"Results on train: {results}")
        reconstructed_features = reconstruct(models, loader_test, device, "test", save = True, filename=filename)
    

def reconstruct(autoencoder, dataloader, device, split=None, save = False, filename = None, debug = False):
    result = {'features': []}
    # for debugging purpose, I introduce also a loss in reconstruction
    reconstruction_loss = nn.MSELoss()
    avg_video_level_loss = 0
    with torch.no_grad():
        for i, (data, label, video_name, uid) in enumerate(dataloader):
            for m in modalities:
                autoencoder[m].train(False)
                # logger.debug(f"Data shape(before squeeze): {data[m].shape}")
                data[m] = data[m].squeeze(1).permute(1, 0, 2)     #  clip level
                # logger.debug(f"Data shape(after squeeze): {data[m].shape}")
                clips = []
                clip_loss = 0
                for i_c in range(args.test.num_clips): #  iterate over the clips
                    clip = data[m][i_c].to(device)     #  retrieve the clip
                    x_hat, _, _, _ = autoencoder[m](clip)      
                    x_hat = x_hat.to(device).detach()
                    # logger.debug(f"Clip: {clip.shape}, x_hat: {x_hat.shape}")
                    # logger.debug(f"Reconstruction loss: {reconstruction_loss(clip, x_hat)}")
                    clip_loss += reconstruction_loss(clip, x_hat)
                    clips.append(x_hat)
                # avg_video_level_loss += clip_loss
                # logger.debug(f"clips è un array({type(clips)}, di dimensione 5({len(clips)})")
                clips = torch.stack(clips, dim = 0)
                # logger.debug(f"clips è un TENSORE({type(clips)}, che rappresenta il video {clips.shape})")
                clips = clips.permute(1, 0, 2)
                # logger.debug(f"clips è un TENSORE({type(clips)}, che rappresenta il video ({clips.shape})[ho eliminato la dimensione inutile]")
                avg_video_level_loss += reconstruction_loss(data[m].permute(1, 0, 2), clips)
                clips = clips.squeeze(0)
                # logger.debug(f"Reconstruction loss: {reconstruction_loss(data[m], clips)}")
                result['features'].append({'features_EMG': clips.numpy(), 'label': label.item(), 'uid': uid.item(), 'video_name': video_name})
    if save:    
        with open(f"{filename}_{split}.pkl", "wb") as file:
            pickle.dump(result, file)
    if debug:
        return result, {'total_loss': avg_video_level_loss, 'avg_loss': avg_video_level_loss/len(dataloader)}
    else:
        return result

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L  

def validate(autoencoder, val_dataloader, device, reconstruction_loss):
    total_loss = 0
    autoencoder.train(False)
    for i, (data, labels) in enumerate(val_dataloader):
        for m in modalities:
            logger.info(f"Data size: {data[m].squeeze(1).shape}")
            data[m] = data[m].squeeze(1).permute(1, 0, 2)
            # print(f"Data after permutation: {data[m].size()}")
        for i_c in range(args.test.num_clips):
            for m in modalities:
                # extract the clip related to the modality
                clip = data[m][i_c].to(device)
                x_hat, _, mean, log_var = autoencoder(clip)
                mse_loss = reconstruction_loss(x_hat, clip)
                kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = mse_loss + kld_loss
                total_loss += loss
    return total_loss/len(val_dataloader)

def costant_scheduler(value = 1, n_epoch = 200):
    return np.ones(n_epoch) * value

def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    


def train(autoencoder, train_dataloader, val_dataloader, device, model_args):
    logger.info(f"Start VAE training.")

    for m in modalities:
        autoencoder[m].load_on(device)

    opt = build_optimizer(autoencoder['EMG'], "adam", model_args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=model_args.lr_steps, gamma=model_args.lr_gamma)

    reconstruction_loss = nn.MSELoss(reduction='mean')

    for m in modalities:
        autoencoder[m].train(True)
    # beta = np.concatenate((costant_scheduler(1/(100*1024), model_args.epochs//2), frange_cycle_sigmoid(0, 1.0, model_args.epochs//2, n_cycle=1)))
    # beta = np.ones(model_args.epochs) - frange_cycle_sigmoid(1/(100*1024), 1, model_args.epochs, n_cycle=10, ratio=.001)
    beta = costant_scheduler(model_args.beta, model_args.epochs)
    # beta = np.concatenate((costant_scheduler(1/(100 * 1024), (model_args.epochs//5)*4), frange_cycle_linear(1/(100 * 1024), .5, (model_args.epochs//5)*1, n_cycle=1, ratio=.001)))
    for epoch in range(model_args.epochs):
        # train_loop
        total_loss = 0 # total loss for the epoch
        for i, (data, _) in enumerate(train_dataloader):
            opt.zero_grad()                                                                 #  reset the gradients    
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)                                          #  Data is now in the form (clip, batch, features)            
            for i_c in range(args.test.num_clips):
                clip_level_loss = 0                                                         #  loss for the clip             
                for m in modalities:
                    # extract the clip related to the modality
                    clip = data[m][i_c].to(device)

                    x_hat, _, mean, log_var = autoencoder[m](clip)

                    mse_loss = reconstruction_loss(x_hat, clip)                              #  compute the reconstruction loss
                    kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  #  compute the KLD loss
                    loss = mse_loss + beta[epoch] * kld_loss
                    # generate an error if loss is nan
                    if loss.isnan():
                        raise ValueError("Loss is NaN.")
                    clip_level_loss += loss
                    loss.backward()
                    opt.step()
                    wandb.log({"Beta": beta[epoch], "MSE LOSS": mse_loss, 'KLD_loss': kld_loss, 'loss': loss, 'lr': scheduler.get_last_lr()[0]})
            total_loss += clip_level_loss.item()
        if epoch % 10 == 0:
            wandb.log({"validation_loss": validate(autoencoder['EMG'], val_dataloader, device, reconstruction_loss)})
        print(f"[{epoch+1}/{model_args.epochs}] - Total loss: {total_loss}")
        scheduler.step()
    return autoencoder


def save_model(model, filename):
        try:
            date = str(datetime.now().date())
            if not os.path.isdir(os.path.join('./saved_models/VAE_EMG', date)):
                os.mkdir(os.path.join('./saved_models/VAE_EMG', date))
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('./saved_models/VAE_EMG', date, filename))
        except Exception as e:
            logger.info("An error occurred while saving the checkpoint:")
            logger.info(e)

def plot_latent(autoencoder, dataloader, device, split = 'train'):
    """
    encodes EMG features, saves them in a latent_split.pkl file and plots them ina img_VAE_split.png file
    """

    output = []
    labels = []
    final_latents = []
    with torch.no_grad():
        #print(len(dataloader))
        for i, (data, label) in enumerate(dataloader):
            output = []
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)
                #print(len(data[m]))
                for i_c in range(args.test.num_clips):
                    clip = data[m][i_c].to(device)
                    z = autoencoder[m].encoder.encode(clip)
                    z = z.to(device).detach()
                    output.append(z)
                output = torch.stack(output)
                output = output.permute(1, 0, 2)
                #print(f'[DEBUG], Batch finito, output: {output.size()}')
                for j in range(len(output)):
                    final_latents.append(output[j])
                    for _ in range(5):
                        labels.append(label[j].item())
    final_latents = torch.stack(final_latents).reshape(-1,512)
    reduced = TSNE().fit_transform(final_latents)
    x_l = reduced[:, 0]
    y_l = reduced[:, 1]
    with open(f"./latent_{split}.pkl", "wb") as file:
        pickle.dump({'x': x_l, 'y': y_l, 'labels': labels}, file)
    
    d = pd.read_pickle(f'./aml22-ego/latent_{split}.pkl')

    colors= ['green', 'red', 'yellow', 'grey', 'green', 'blue', 'black', 'purple']
    for x, y, l in zip(d['x'], d['y'], d['labels']):
        plt.scatter(x, y, c=colors[l])
    plt.savefig(f"./img_VAE_{split}.png")
    plt.show()
  # colors= ['green', 'red', 'yellow', 'grey', 'green', 'blu', 'black', 'purple']
    # # for x, y, l in zip(x_l, y_l, labels):
    # #     print(colors[l])
    # plt.scatter(x_l, y_l, c=colors, label=labels)    
    # plt.legend()
    # plt.savefig("./img_VAE.png")
    # plt.show()

def load_model(ae, path):
    state_dict = torch.load(path)["model_state_dict"]
    #print([x for x in state_dict.keys()])
    ae.load_state_dict(state_dict, strict=False)

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
