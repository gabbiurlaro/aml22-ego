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
        wandb.login(key='c87fa53083814af2a9d0ed46e5a562b9a5f8b3ec') # salvatore
        #wandb.login(key='ec198a4a4d14b77926dc5316ae6f02def3f71b17') # gabbo
        wandb.init(project="test-project", entity="egovision-aml22")
        #wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]
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
        models[m] = getattr(model_list, args.models[m].model)(1024, 512, 1024)

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    # action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
    #                                             args.total_batch, args.models_dir, num_classes,
    #   
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
                                                                     'val', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        #ae = train(models, train_loader, val_loader, device)
        #save_model(ae['RGB'],f"{args.name}.pth")
        #plot_latent(ae, train_loader, device, split='D1_train')
        
        load_model(models['RGB'], './saved_models/VAE_RGB/VAE_FT_D_16f.pth')
        reconstruct(models['RGB'], train_loader, device)
        # plot_latent(ae, train_loader, device)
        # reconstruct(ae, train_loader, device)

def reconstruct(autoencoder, dataloader, device):
    output = []
    labels = []
    final_latents = []
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            output = []
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)
                print(len(data[m]))
                for i_c in range(args.test.num_clips):
                    clip = data[m][i_c].to(device)
                    z = autoencoder[m](clip)
                    z = z.to(device).detach()
                    output.append(z)
                output = torch.stack(output)
                output = output.permute(1, 0, 2)
                #print(f'[DEBUG], Batch finito, output: {output.size()}')
                for j in range(len(output)):
                    final_latents.append(output[j])
                    labels.append(label[j].item())
    final_latents = torch.stack(final_latents).reshape(-1,1024)
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
                loss = mse_loss + kld_loss
                total_loss += loss
    return total_loss/len(val_dataloader)


def train(autoencoder, train_dataloader, val_dataloader, device, epochs=75):
    logger.info(f"Start VAE training.")
    train_loss = []
    for m in modalities:
        autoencoder[m].load_on(device)
    opt = torch.optim.SGD(autoencoder['RGB'].parameters(), lr=0.0001, weight_decay=10e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=10e-5)
    reconstruction_loss = nn.MSELoss()
    autoencoder['RGB'].train(True)
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_dataloader):
            opt.zero_grad()
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)
                # print(f"Data after permutation: {data[m].size()}")
            for i_c in range(args.test.num_clips):
                for m in modalities:
                    opt.zero_grad()
                    # extract the clip related to the modality
                    clip = data[m][i_c].to(device)
                    x_hat, _, mean, log_var = autoencoder[m](clip)
                    # print(f"From autoencoder: {x_hat.size()}")
                    mse_loss = reconstruction_loss(x_hat, clip)
                    kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    loss = mse_loss + kld_loss
                    loss.backward()
                    opt.step()
                    wandb.log({"MSE LOSS": mse_loss, "KLD Loss": kld_loss, 'loss': loss})
        if epoch % 20 == 0:
            wandb.log({"Valdation loss": validate(autoencoder['RGB'], val_dataloader, device, reconstruction_loss)})
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
        print(len(dataloader))
        for i, (data, label) in enumerate(dataloader):
            output = []
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)
                print(len(data[m]))
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
    ae.load_state_dict(torch.load(path)["model_state_dict"], strict=False)




if __name__ == '__main__':
    main()
