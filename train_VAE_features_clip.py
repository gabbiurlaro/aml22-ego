from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
import matplotlib.pyplot as plt
from  sklearn.manifold import TSNE
#from models.VAE import Encoder, Decoder, VAE

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
        wandb.login(key='c87fa53083814af2a9d0ed46e5a562b9a5f8b3ec')
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
       # train(action_classifier, train_loader, val_loader, device, num_classes)
        ae = train(models['RGB'], train_loader, device)
        plot_latent(ae, train_loader, device)
    
def train(autoencoder, data, device, epochs=20):
    autoencoder.load_on(device)
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for m in modalities:
            for x, y in data:
                x[m] = x[m].reshape((160,1024)).to(device) # GPU
                #print(x[m].size())
                opt.zero_grad()
                x_hat = autoencoder(x[m])
                loss = ((x[m] - x_hat)**2).sum() + autoencoder.encoder.kl
                loss.backward()
                opt.step()
                wandb.log({'log_loss': loss})
    return autoencoder

def plot_latent(autoencoder, data, device, num_batches=100):
    plt.figure()
    latent = np.zeros((160,len(data), 2))
    Y = np.zeros((160,len(data),2))
    ue = {}
    for i, (x, y) in enumerate(data):
        for m in modalities:
            y = [[el]*5 for el in y]
            y = [el for sub in y for el in sub]
            z = autoencoder.encoder(x[m].reshape((160,1024)).to(device))
            z = z.to('cpu').detach().numpy()
            reduced = TSNE().fit_transform(z)
            latent[i] = reduced
            Y[i] = y
            # if i > num_batches:
            #     plt.colorbar()
            #     break
    #plt.show()
    # filtered = {}
    # ue['x'] = reduced[:, 0]
    # ue['y'] = reduced[:, 1]
    # for i in range(8): # ek has 8 classes
    #     filtered['x'] = [ue['x'][j]  for j, out in enumerate(Y) if out==i ]
    #     filtered['y'] = [ue['y'][j]  for j, out in enumerate(Y) if out==i ]
    #     plt.scatter(filtered['x'], filtered['y'], c=Y[i], label=Y[i])
    
    latent =  np.array(latent).reshape(len(data),2)
    print(f'latent: {latent.shape}, Y : {Y.shape}')
    plt.scatter(latent[:,0], latent[:,1], c=Y, label=Y)
    plt.legend()
    #plt.title(title)
    plt.savefig("./img_VAE.png")

if __name__ == '__main__':
    main()
