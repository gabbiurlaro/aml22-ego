from datetime import datetime
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionNetDataset, Basic_Transform
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
import pickle

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
        run = wandb.init(project="EMG-fe", entity="egovision-aml22", name=F"{args.models.EMG.model}_{args.models.EMG.lr}_fe")


def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    # device where everything is run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))

        # notice that here, the first parameter passed is the input dimension
        # In our case it represents the feature dimensionality which is equivalent to 1024 for I3D
        models[m] = getattr(model_list, args.models[m].model)(input_size = (16, args.train.num_frames_per_clip.EMG, args.train.num_frames_per_clip.EMG), 
                                                                output_size = (args.embeddings_size, 1, 1), num_classes= num_classes, num_clips=args.train.num_clips, use_batch_norm=True)

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                args.train.num_clips, args.models, args, device=device)

    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here
        train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, {'EMG': args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                       None, load_feat=False, kwargs={}),
                                                   batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'z', args.dataset,  {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                     None, load_feat=False),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        train(action_classifier, train_loader, val_loader, device, num_classes)
        save_model(models['EMG'], f"{args.name}_lr{args.models.EMG.lr}.pth")

    elif args.action == "validate":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       args.split , args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes)

    elif args.action == "save":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        
        loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                 args.split, args.dataset,
                                                                 args.save.num_frames_per_clip,
                                                                 1, args.save.dense_sampling,additional_info=True,
                                                                 **{"save": args.split}),
                                             batch_size=1, shuffle=False,
                                             num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes)
    
    elif args.action == "job_feature_extraction_aug":
        if args.augmentation:
            T_train_loaders = {}
            T_val_loaders = {}
            train_loaders = {}
            val_loaders = {}
            _features= {
                        'WD-MW': '../drive/MyDrive/actionnet_aug/Augmented_dataset_clip_WD-MW', 
                        'MW': '../drive/MyDrive/actionnet_aug/Augmented_dataset_clip_MW',
                        'WD': '../drive/MyDrive/actionnet_aug/Augmented_dataset_clip_WD', 
                        'MW-WD': '../drive/MyDrive/actionnet_aug/Augmented_dataset_clip_MW-WD',
                        }
            
            for a in _features.keys():
                args.dataset.EMG.features_name = _features[a]
                T_train_loaders[a] = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'train', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                            None, load_feat=True, additional_info=False, kwargs={'aug': True}),
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                T_val_loaders[a] = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'test', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=True, additional_info=False, kwargs={'aug': True}),
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                train_loaders[a] = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'train', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=True, additional_info=True, kwargs={'aug': True}),
                                                        batch_size=1, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                val_loaders[a] = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'test', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=True, additional_info=True, kwargs={'aug': True}),
                                                        batch_size=1, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
            if args.resume_from is not None:
                logger.info(f"Loading model from {args.resume_from}")
                action_classifier.load_last_model(args.resume_from)
                logger.info(f'modalities: {modalities}')
                logger.info(f' aug: {args.augmentation}')
                timestamp = datetime.now()
                logger.info('here')
                for a in train_loaders.keys():
                    save_feat(action_classifier, train_loaders[a], device, action_classifier.current_iter, num_classes, train=True, aug=_features[a])
                    save_feat(action_classifier, val_loaders[a], device, action_classifier.current_iter, num_classes, train=False,  aug=_features[a])
                    logger.info(f'Finished extracting train features, now exiting...')
            else:
                training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
                # all dataloaders are generated here
                T_train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'train', args.dataset, {'EMG': args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False, additional_info=False),
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

                T_val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                            'test', args.dataset,  {'EMG': args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False, additional_info=False),
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'train', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False, additional_info=True),
                                                        batch_size=1, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

                val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                            'test', args.dataset,  {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False,additional_info=True ),
                                                        batch_size=1, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                
                logger.info(f'Starting training...')
                
                train(action_classifier, T_train_loader, T_val_loader, device, num_classes)
                logger.info(f'Finished training, now validating...')
                validate(action_classifier, T_val_loader, device, action_classifier.current_iter, num_classes)
                logger.info(f'Finished validating, now saving model...')
                for a in train_loaders.keys():
                    train(action_classifier, T_train_loaders[a], T_val_loaders[a], device, num_classes)
                    logger.info(f'Finished training, now validating...')
                    validate(action_classifier, T_val_loaders[a], device, action_classifier.current_iter, num_classes)
                    logger.info(f'Finished validating, now saving model...')
                timestamp = datetime.now()
                save_model(models['EMG'], f"{args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
                logger.info(f"Model saved in {args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
                logger.info(f'Finished saving model, now extracting features...')
                save_feat(action_classifier, train_loader, device, action_classifier.current_iter, num_classes, train=True)
                save_feat(action_classifier, val_loader, device, action_classifier.current_iter, num_classes, train=False)

                for a in train_loaders.keys():
                    save_feat(action_classifier, train_loaders[a], device, action_classifier.current_iter, num_classes, train=True, aug=_features[a])
                    save_feat(action_classifier, val_loaders[a], device, action_classifier.current_iter, num_classes, train=False,  aug=_features[a])
                    logger.info(f'Finished extracting train features, now exiting...')
        else:
            if args.resume_from is not None:
                #ae = train(models, train_loader, val_loader, device, args.models.EMG)
                loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                        'train',args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False, additional_info=True),
                                                    batch_size=1, shuffle=False,
                                                    num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=True)
                logger.info(f'Finished extracting train features, now exiting...')

                loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                        'test', args.dataset, {'EMG': args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False, additional_info=True),
                                                    batch_size=1, shuffle=False,
                                                    num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=False)
                logger.info(f'Finished extracting test features, now exiting...')
            else:
                training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
                # all dataloaders are generated here
                train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                            'train', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False),
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

                val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                            'test', args.dataset,  {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips,{'EMG': False},
                                                                            None, load_feat=False),
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                

                loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                    args.split, args.dataset,
                                                                    args.save.num_frames_per_clip,
                                                                    1, args.save.dense_sampling,additional_info=True,
                                                                    **{"save": args.split}),
                                                batch_size=1, shuffle=False,
                                                num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
                                                
                logger.info(f'Starting training...')
                train(action_classifier, train_loader, val_loader, device, num_classes)
                logger.info(f'Finished training, now validating...')
                validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes)
                logger.info(f'Finished validating, now saving model...')
                timestamp = datetime.now()
                save_model(models['EMG'], f"{args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
                logger.info(f"Model saved in {args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
                logger.info(f'Finished saving model, now extracting features...')
                save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=False)
                logger.info(f'Finished extracting {args.split} features, now exiting...')
    
    elif args.action == "job_feature_extraction":
        transform = Basic_Transform()

        if args.resume_from is not None:
            #ae = train(models, train_loader, val_loader, device, args.models.EMG)
            loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                    'train',args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                        transform=transform, load_feat=False, additional_info=True, kwargs={}),
                                                batch_size=1, shuffle=False,
                                                num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
            save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=True)
            logger.info(f'Finished extracting train features, now exiting...')
            loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                    'test', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                        transform=transform, load_feat=False, additional_info=True, kwargs={}),
                                                batch_size=1, shuffle=False,
                                                num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
            save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=False)
            logger.info(f'Finished extracting test features, now exiting...')
        else:
            training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
            # all dataloaders are generated here
            train_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[0], modalities,
                                                                        'train', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                       transform=transform, load_feat=False, kwargs={}),
                                                    batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                        'test', args.dataset,  {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                       transform=transform, load_feat=False, kwargs={}),
                                                    batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
            
            loader = torch.utils.data.DataLoader(ActionNetDataset(args.dataset.shift.split("-")[1], modalities,
                                                                 'train', args.dataset, {'EMG':args.train.num_frames_per_clip.EMG}, args.train.num_clips, {'EMG': False},
                                                                       transform=transform, load_feat=False, additional_info=True,
                                                                kwargs={"save": args.split}),
                                            batch_size=1, shuffle=False,
                                            num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
                                            
            logger.info(f'Starting training...')
            train(action_classifier, train_loader, val_loader, device, num_classes , num_clips=args.train.num_clips)
            logger.info(f'Finished training, now validating...')
            validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes,  num_clips=args.test.num_clips)
            logger.info(f'Finished validating, now saving model...')
            timestamp = datetime.now()
            save_model(models['EMG'], f"{args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
            logger.info(f"Model saved in {args.name}_lr{args.models.EMG.lr}_{timestamp}.pth")
            logger.info(f'Finished saving model, now extracting features...')
            save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=False, num_clips=args.save.num_clips)
            save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=False, num_clips=args.save.num_clips)
            logger.info(f'Finished extracting {args.split} features, now exiting...')
            split = 'train'
            save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=True)
            logger.info(f'Finished extracting {split} features')
            logger.info(f'Now extracting features...')
            save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, train=False)
            logger.info(f'Finished extracting {split} features, now exiting...')

    else:
        raise NotImplementedError
    



def save_feat(model, loader, device, it, num_classes, train=False, num_clips = 5, aug=None):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities
    batch = 1
    model.reset_acc()
    model.train(False)
    results_dict = {"features": []}
    num_samples = 0
    logits = {}
    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label, video_name, uid) in enumerate(loader):
            
            label = label.to(device)
            #logger.info(f'video_name: {video_name},  data: {data["EMG"].shape} {data["EMG"][0].shape}')
            for m in modalities:
                data[m] = data[m].reshape(-1, 16, num_clips, args.train.num_frames_per_clip.EMG, args.train.num_frames_per_clip.EMG)
                data[m] = data[m].permute(2, 0, 1, 3, 4)
                data[m] = data[m].to(device)
                logits[m] = torch.zeros((args.save.num_clips, batch, num_classes)).to(device)
            
                output, feat = model(data)
                logits[m] = output[m]
                swap = [feat[i][m] for i in range(args.save.num_clips)]

                final_features = torch.stack(swap)

                logits[m] = torch.mean(logits[m], dim=0) # average over clips to predict the label
           
                sample = {}
                
                sample['label'] = label.item()
                sample['uid'] = uid.item()
                sample['untrimmed_video_name'] = video_name
                sample[f'features_{m}'] = final_features.cpu().numpy()    

                results_dict['features'].append(sample)

                #logger.info(f'main : feat: len_keys: {len(feat.keys())}, keys: {feat.keys()}, \n feat_:{feat}')
            num_samples += batch

            #model.compute_accuracy(logits, label)

            #if (i_val + 1) % (len(loader) // 5) == 0:
            #    logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(loader),
            #                                                              model.accuracy.avg[1], model.accuracy.avg[5]))
        os.makedirs("saved_features", exist_ok=True)
        if aug:
            filename = str('../drive/MyDrive/EXTRACTED_FEATURES_AUG_1/' + 'Augmented_features_' + aug.split("/")[-1].split('_')[3]  + "_" + ('train' if train else 'test') + ".pkl")
            pickle.dump(results_dict, open(filename, 'wb'))
        else:
            pickle.dump(results_dict, open(os.path.join("saved_features/", args.name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" +
                                                        ('train' if train else 'test') + "_" +
                                                        args.split + ".pkl"), 'wb'))
    #logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
    #            .format(np.array(class_accuracies.values()).mean()))
    #test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
    #                'class_accuracies': np.array(class_accuracies.values())}

    # with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
    #                                      f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
    #     f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter #test_results['top1']))

    return 0

def train(action_classifier, train_loader, val_loader, device, num_classes, num_clips):
    """
    function to train the model on the test set
    action_classifier: Task containing the model to be trained
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    num_classes: int, number of classes in the classification problem
    """
    global training_iterations, modalities

    data_loader_source = iter(train_loader)
    action_classifier.train(True)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)
    wandb.watch(action_classifier.task_models['EMG'])

    # the batch size should be total_batch but batch accumulation is done with batch size = batch_size.
    # real_iter is the number of iterations if the batch size was really total_batch
    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter % args.models['EMG'].lr_steps == 0:
            # learning rate decay at iteration = lr_steps
            action_classifier.reduce_learning_rate()
        # gradient_accumulation_step is a bool used to understand if we accumulated at least total_batch
        # samples' gradient
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        start_t = datetime.now()
        # the following code is necessary as we do not reason in epochs so as soon as the dataloader is finished we need
        # to redefine the iterator
        try:
            source_data, source_label = next(data_loader_source)
        except StopIteration:
            data_loader_source = iter(train_loader)
            source_data, source_label = next(data_loader_source)
        end_t = datetime.now()

        logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
                    f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

        ''' Action recognition'''
        source_label = source_label.to(device)
# properly reshaping the input data
       # for m in modalities:
            # put the data in the proper format for the model processing
       #     batch, _, height, width = source_data[m].shape
       #     source_data[m] = source_data[m].reshape(batch, args.train.num_clips, args.train.num_frames_per_clip[m],
        #                                            -1, height, width)
        #    source_data[m] = source_data[m].permute(1, 0, 3, 2, 4, 5)
        data = source_data
        logits = []
        
        
        for m in modalities:
            #print(f'yoyo1: {data[m].size()}, {data[m].shape}')
            data[m] = data[m].reshape(-1,16, num_clips, args.train.num_frames_per_clip.EMG, args.train.num_frames_per_clip.EMG)
            data[m] = data[m].permute(2, 0, 1, 3,4 )
            #print(f'yoyo2: {data[m].size()}, {data[m].shape}')
            data[m] = data[m].to(device)
        
        logits, _  = action_classifier.forward(data)

        action_classifier.compute_loss(logits, source_label, loss_weight=1)
        action_classifier.backward(retain_graph=False)
        action_classifier.compute_accuracy(logits, source_label)

        action_classifier.wandb_log()
    
        # update weights and zero gradients if total_batch samples are passed
        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args.train.num_iter, action_classifier.loss.val, action_classifier.loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done, notice we validate and
        # save the last 9 models
        if gradient_accumulation_step and real_iter % 10 == 0:
            val_metrics = validate(action_classifier, val_loader, device, int(real_iter), num_classes,  num_clips=num_clips)
            wandb.log({'accuracy on val': val_metrics['top1']})
           
            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']

            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)

def validate(model, val_loader, device, it, num_classes, num_clips):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}
    #print(f'val: {val_loader.dataset.__len__()}')
    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)
            #print(f'data: {data.size()}, {data.shape }, label: {label.size()}, {label.shape}')
            for m in modalities:
                #print(f'yoyo1: {data[m].size()}, {data[m].shape}')
                data[m] = data[m].reshape(-1,16, args.train.num_clips,args.train.num_frames_per_clip.EMG,args.train.num_frames_per_clip.EMG)
                data[m] = data[m].permute(2, 0, 1, 3,4 )
                #print(f'yoyo2: {data[m].size()}, {data[m].shape}')
                data[m] = data[m].to(device)
                batch = data[m].shape[0]
                #print('num_classes: ', num_classes)
                logits[m] = torch.zeros((batch, num_classes)).to(device)


            output, _ = model(data)
            #print(f'output: {output.size()}, {output.shape}')
            for m in modalities:
                logits[m] = output[m]
            
            #print(f"label: {label.size()}, {label.shape}")
            # for m in modalities:
            #     logits[m] = torch.mean(logits[m], dim=0)
            #print(f"output1: {output}, {output['EMG']} {output['EMG'].shape}")
            model.compute_accuracy(logits, label)

            # if (i_val + 1) % (len(val_loader) // 5) == 0:
            #     logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
            #                                                               model.accuracy.avg[1], model.accuracy.avg[5]))

        # class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        # for i_class, class_acc in enumerate(class_accuracies):
        #     logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
        #                                                  int(model.accuracy.correct[i_class]),
        #                                                  int(model.accuracy.total[i_class]),
        #                                                  class_acc))

    # logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
    #             .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5]}

    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
                                         f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results

def save_model(model, filename):
        try:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('./saved_models/VAE_RGB', filename))
        except Exception as e:
            logger.info("An error occurred while saving the checkpoint:")
            logger.info(e)




if __name__ == '__main__':
    main()
