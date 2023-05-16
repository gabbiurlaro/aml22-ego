from wsgiref import validate
from utils.logger import logger
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import wandb
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

# with this script we trained and tested FC_VAE.VariationalAutoencoder to reconstruct features from the RGB modality
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
        wandb.login(key='c87fa53083814af2a9d0ed46e5a562b9a5f8b3ec')
        run = wandb.init(project="FC-VAE(rgb)", entity="egovision-aml22")
        wandb.run.name = f'{args.name}_{args.models.RGB.model}'

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
        # resume_from argument is adopted in case of restoring from a checkpoint
        # if args.resume_from is not None:
        #     action_classifier.load_last_model(args.resume_from)
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        # training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here
        train_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'test', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        ae = train(models, train_loader, val_loader, device, args.models.RGB)
        logger.info(f"TRAINING VAE FINISHED, SAVING THE MODELS...")
        save_model(ae['RGB'], f"{args.name}_lr{args.models.RGB.lr}_{datetime.now()}.pth")
        logger.info(f"Model saved in {args.name}_lr{args.models.RGB.lr}_{datetime.now()}.pth")

    elif args.action == "save":
        loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       args.split , args.dataset, None, None, None,
                                                                       None, load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        loader_test = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       "test", args.dataset, None, None, None,
                                                                       None, load_feat=True, additional_info=True),
                                                   batch_size=1, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        last_model = args.resume_from
        logger.info(f"Loading last model from {last_model}")
        load_model(models['RGB'], last_model)
        logger.info(f"Reconstructing features...")
        filename = f"./saved_features/reconstructed/{datetime.now()}"
        reconstructed_features = reconstruct(models, loader, device, "train", save = True, filename=filename)
        reconstructed_features = reconstruct(models, loader_test, device, "test", save = True, filename=filename)

def reconstruct(autoencoder, dataloader, device, split=None, save = False, filename = None):
    result = {'features': []}

    with torch.no_grad():
        for i, (data, label, video_name, uid) in enumerate(dataloader):
            for m in modalities:
                autoencoder[m].train(False)
                data[m] = data[m].permute(1, 0, 2)     #  clip level
                # print(f'[DEBUG]: data[m] ha come primo elemento la dimensione delle clip: {data[m].size()}')
                clips = []
                for i_c in range(args.test.num_clips): #  iterate over the clips
                    clip = data[m][i_c].to(device)     #  retrieve the clip
                    z, _, _, _ = autoencoder[m](clip)      
                    z = z.to(device).detach()
                    clips.append(z)
                # print(f"[DEBUG] clips è un array({type(clips)}, di dimensione 5({len(clips)})")
                clips = torch.stack(clips, dim = 0)
                # print(f"[DEBUG] clips è un TENSORE({type(clips)}, che rappresenta il video {clips.shape})")
                clips = clips.permute(1, 0, 2).squeeze(0)
                # print(f"[DEBUG] clips è un TENSORE({type(clips)}, che rappresenta il video ({clips.shape})[ho eliminato la dimensione inutile]")
                result['features'].append({'features_RGB': clips.numpy(), 'label': label.item(), 'uid': uid.item(), 'video_name': video_name})
                
    if save:
        with open(f"{filename}_D1_{split}.pkl", "wb") as file:
            pickle.dump(result, file)
        
    return result
    # reduced = TSNE().fit_transform(final_latents)
    # x_l = reduced[:, 0]
    # y_l = reduced[:, 1]
    # with open(f"./latent_{split}.pkl", "wb") as file:
    #     pickle.dump({'x': x_l, 'y': y_l, 'labels': labels}, file)
    
    # d = pd.read_pickle(f'./latent_{split}.pkl')

    # colors= ['green', 'red', 'yellow', 'grey', 'green', 'blue', 'black', 'purple']
    # for x, y, l in zip(d['x'], d['y'], d['labels']):
    #     plt.scatter(x, y, c=colors[l])
    # plt.savefig(f"./img_VAE_{split}.png")
    # plt.show()


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
            data[m] = data[m].permute(1, 0, 2)
            # print(f"Data after permutation: {data[m].size()}")
        for i_c in range(args.test.num_clips):
            for m in modalities:
                # extract the clip related to the modality
                clip = data[m][i_c].to(device)
                x_hat, _, mean, log_var = autoencoder(clip)
                mse_loss = reconstruction_loss(x_hat, clip)
                kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = mse_loss # + kld_loss
                total_loss += loss
            
    return total_loss/len(val_dataloader)

def train(autoencoder, train_dataloader, val_dataloader, device, model_args):
    logger.info(f"Start VAE training.")
    train_loss = []
    for m in modalities:
        autoencoder[m].load_on(device)
    opt = build_optimizer(autoencoder['RGB'], "sgd", model_args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=model_args.lr_steps, gamma=model_args.lr_gamma)
    reconstruction_loss = nn.MSELoss()
    autoencoder['RGB'].train(True)
    beta = frange_cycle_linear(0, 0.01, model_args.epochs, n_cycle=2)
    #beta = 200*[1]
    step_value = 1
    for epoch in range(model_args.epochs):
        total_loss = 0
        for i, (data, labels) in enumerate(train_dataloader):
            opt.zero_grad()
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)
                # print(f"Data after permutation: {data[m].size()}")
            for i_c in range(args.test.num_clips):
                for m in modalities:
                    # extract the clip related to the modality
                    clip = data[m][i_c].to(device)
                    x_hat, _, mean, log_var = autoencoder[m](clip)
                    # print(f"[DEBUG]: x_hat: {x_hat.type}, {x_hat.shape}  mean {mean.shape}, log_var {log_var.shape}")
                    mse_loss = reconstruction_loss(x_hat, clip)
                    # print(f"[DEBUG]: mse_loss {type(mse_loss)} - {mse_loss.shape} -{mse_loss}")
                    kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    # print(f"[DEBUG]: kld {type(kld_loss)} - {kld_loss.shape} - {kld_loss}")
                    loss = mse_loss + beta[epoch]*kld_loss
                    if loss.isnan():
                        logger.info(f"Loss exploding...")
                    # print(f"loss: {loss.shape} - {loss}")
                    total_loss += loss
                    wandb.log({"Beta": beta[epoch], "MSE LOSS": mse_loss, "KLD Loss": kld_loss, 'loss': loss, 'lr': scheduler.get_last_lr()[0]})
        total_loss.backward()
        opt.step()

        if epoch % 10 == 0:
            wandb.log({"validation_loss": validate(autoencoder['RGB'], val_dataloader, device, reconstruction_loss)})
        print(f"[{epoch+1}/{model_args.epochs}] - {total_loss/len(train_dataloader)}")
        scheduler.step()
    return autoencoder

def save_model(model, filename):
        try:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('./saved_models/VAE_RGB', filename))
        except Exception as e:
            logger.info("An error occurred while saving the checkpoint:")
            logger.info(e)

def plot_latent(autoencoder, dataloader, device, split = 'train'):
    """
    encodes rgb features, saves them in a latent_split.pkl file and plots them ina img_VAE_split.png file
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
