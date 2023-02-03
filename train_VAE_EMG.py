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
        # wandb.login(key='ec198a4a4d14b77926dc5316ae6f02def3f71b17') # gabbo
        wandb.init(project="test-project", entity="egovision-aml22")
        #wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]
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
        models[m] = getattr(model_list, args.models[m].model)()

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
        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, {'EMG': 32}, 5, {'EMG': True},
                                                                       None, load_feat=False),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'test', args.dataset,  {'EMG': 32}, 5, {'EMG': True},
                                                                     None, load_feat=False),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        ae = train(models, train_loader, val_loader, device, args.models.EMG)
        logger.info(f"TRAINING VAE FINISHED, SAVING THE MODELS...")
        save_model(ae['EMG'], f"{args.name}_lr{args.models.EMG.lr}.pth")
        logger.info(f"DONE in {args.name}_lr{args.models.EMG.lr}.pth")

        #plot_latent(ae, train_loader, device, split='D1_train')
        
        # load_model(models['RGB'], './saved_models/VAE_RGB/VAE_FT_D_16f.pth')
        # reconstruct(models['RGB'], train_loader, device, split="D1_train")
        # plot_latent(ae, train_loader, device)
        # reconstruct(ae, train_loader, device)
    elif args.action == "save":
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       args.split , args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=1, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        last_model = args.resume_from
        logger.info(f"Loading last model from {last_model}")
        load_model(models['EMG'], last_model)
        logger.info(f"Reconstructing features...")
        filename = f"./saved_features/reconstructed/{args.name}_{args.models.RGB.lr}.pkl"
        reconstructed_features = reconstruct(models, loader, device, args.split, save = True, filename=filename)
        
        # some statitics:
        logger.info(f"Reconstructed feature of {len(reconstructed_features['features_RGB'])} video")
        # print(f"Un sample è di questo tipo: {reconstructed_features['features_RGB'][0]['features'].shape}")
        logger.info(f"Filename is: ./saved_features/reconstructed/{args.name}_{args.models.RGB.lr}.pkl")

def reconstruct(autoencoder, dataloader, device, split=None, save = False, filename = None):
    result = {'features_RGB': []}

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            for m in modalities:
                autoencoder[m].train(False)
                data[m] = data[m].reshape(-1,16,5,32,32)
                data[m] = data[m].permute(2, 3, 1, 0,4 )    #  clip level
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
                result['features_RGB'].append({'features': clips.numpy(), 'label': label.item()})
                
    if save:
        with open(filename, "wb") as file:
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


def validate(autoencoder, val_dataloader, device, reconstruction_loss):
    total_loss = 0
    autoencoder.train(False)
    for i, (data, labels) in enumerate(val_dataloader):
        for m in modalities:
            print(m, data[m].shape)
            data[m] = data[m].reshape(-1,16,5,32,32)
            data[m] = data[m].permute(2, 3, 1, 0,4 )
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
    opt = torch.optim.Adam(autoencoder['EMG'].parameters(), model_args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=model_args.lr_steps, gamma=10e-2)
    reconstruction_loss = nn.MSELoss()
    autoencoder['EMG'].train(True)
    beta = 0.00001
    step_value = 1
    for epoch in range(model_args.epochs):
        total_loss = 0
        for i, (data, labels) in enumerate(train_dataloader):
            opt.zero_grad()
            for m in modalities:
                print(data[m].shape) # torch.Size([32, 16, 160, 32])
                data[m] = data[m].reshape(-1,16,5,32,32)
                data[m] = data[m].permute(2, 3, 1, 0,4 )
                print(f"Data after permutation: {data[m].size()}")
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
                    loss = mse_loss + beta*kld_loss
                    if loss.isnan():
                        logger.info(f"Loss exploding...")
                        exit(-1)
                    # print(f"loss: {loss.shape} - {loss}")
                    total_loss += loss
                    wandb.log({"MSE LOSS": mse_loss, "KLD Loss": kld_loss, 'loss': loss, 'lr': scheduler.get_last_lr()[0]})
                    loss.backward()
                    opt.step()
        if epoch % 10 == 0:
            step_value = 0.8*step_value
        if epoch % 20 == 0:
            wandb.log({"Validation loss": validate(autoencoder['EMG'], val_dataloader, device, reconstruction_loss)})
        print(f"[{epoch+1}/{model_args.epochs}] - {total_loss/len(train_dataloader)}")
        wandb.log({'train_loss': train_loss})
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
                data[m] = data[m].reshape(-1,16,5,32,32)
                data[m] = data[m].permute(2, 3, 1, 0,4 )
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




if __name__ == '__main__':
    main()
