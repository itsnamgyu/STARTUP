import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import torch.utils.data

from tqdm import tqdm
import argparse

import os
import numpy as np

import utils
import data

import time
import models


import wandb
import warnings

import random 
from collections import OrderedDict

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, cars_few_shot, cub_few_shot, \
    places_few_shot, plantae_few_shot
from datasets import miniImageNet_few_shot, tiered_ImageNet_few_shot, ImageNet_few_shot

import copy
import math

import warnings

from nx_ent import NTXentLoss


class projector_SIMCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''
    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class apply_twice:
    '''
        A wrapper for torchvision transform. The transform is applied twice for 
        SimCLR training
    '''
    def __init__(self, transform, transform2=None):
        self.transform = transform

        if transform2 is not None:
            self.transform2 = transform2
        else:
            self.transform2 = transform

    def __call__(self, img):
        return self.transform(img), self.transform2(img)

def pseudolabel_dataset(embedding, clf, dataset, transform, transform_test, params):
    '''
        pseudolabel the dataset with the teacher model (embedding, clf)
    '''

    # Change the transform of the target dataset to the deterministic transformation
    dataset.d.transform = transform_test
    dataset.d.target_transform = (lambda x: x)

    embedding.eval()
    clf.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=params.bsize,
                        shuffle=False, drop_last=False, num_workers=params.num_workers)

    # do an inference on the full target dataset
    probs_all = []
    for X, _ in loader:
        X = X.cuda()

        with torch.no_grad():
            feature = embedding(X)
            logits = clf(feature)
            probs = F.softmax(logits, dim=1)
            probs += 1e-6

        probs_all.append(probs)
    
    probs_all = torch.cat(probs_all, dim=0).cpu()

    # Update the target dataset with the pseudolabel
    if hasattr(dataset.d, 'targets'):
        dataset.d.targets = probs_all
        samples = [(i[0], probs_all[ind_i])for ind_i, i in enumerate(dataset.d.samples)]
        dataset.d.samples = samples
        dataset.d.imgs = samples
    elif hasattr(dataset.d, "labels"):
        dataset.d.labels = probs_all
    else:
        raise ValueError("No Targets variable found!")
    
    # Switch the dataset's augmentation back to the stochastic augmentation 
    dataset.d.transform = transform
    return dataset

def main(args):
    # Set the scenes
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, 'checkpoint.log'), __name__)
    trainlog = utils.savelog(args.dir, 'train')
    vallog = utils.savelog(args.dir, 'val')

    wandb.init(project='STARTUP',
               group=__file__,
               name=f'{__file__}_{args.dir}')

    wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # seed the random number generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create Models
    ###########################
    if args.model == 'resnet10':
        backbone = models.ResNet10()
        feature_dim = backbone.final_feat_dim
    elif args.model == 'resnet12':
        backbone = models.Resnet12(width=1, dropout=0.1)
        feature_dim = backbone.output_size
    elif args.model == 'resnet18':
        backbone = models.resnet18(remove_last_relu=False,
                                                    input_high_res=True)
        feature_dim = 512
    else:
        raise ValueError('Invalid backbone model')

    backbone_sd_init = copy.deepcopy(backbone.state_dict())

    # load the teacher
    # specified at args.teacher_path
    if args.teacher_path is not None:
        if args.teacher_path_version == 0:
            state = torch.load(args.teacher_path)['state']
            clf_state = OrderedDict()
            state_keys = list(state.keys())
            for _, key in enumerate(state_keys):
                if "feature." in key:
                    # an architecture model has attribute 'feature', load architecture 
                    # feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)
                elif "classifier." in key:
                    newkey = key.replace("classifier.", "")
                    clf_state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            sd = state
            clf_sd = clf_state
        elif args.teacher_path_version == 1:
            temp = torch.load(args.teacher_path)
            sd = temp['model']
            clf_sd = temp['clf']
        else:
            raise ValueError("Invalid load path version!")

        backbone.load_state_dict(sd)

    backbone = nn.DataParallel(backbone).cuda()

    num_classes = clf_sd['weight'].shape[0]
    clf_teacher = nn.Linear(feature_dim, num_classes).cuda()
    clf_teacher.load_state_dict(clf_sd)

    # the student classifier head
    clf = nn.Linear(feature_dim, num_classes).cuda()

    # initialize the student classifier head with the teacher classifier head
    if args.use_pretrained_clf:
        print("Loading Pretrained Classifier")
        clf.load_state_dict(clf_sd)

    # projection head for SimCLR
    clf_SIMCLR = projector_SIMCLR(feature_dim, args.projection_dim).cuda()
    ############################

    ###########################
    # Create DataLoader
    ###########################

    # create the base dataset
    if args.base_dataset == 'miniImageNet':
        base_transform = miniImageNet_few_shot.TransformLoader(args.image_size).get_composed_transform(aug=True)
        base_transform_test = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(root=args.base_path, transform=base_transform)
        if args.base_split is not None:
            base_dataset = miniImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
    elif args.base_dataset == 'tiered_ImageNet':
        if args.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        base_transform = tiered_ImageNet_few_shot.TransformLoader(args.image_size).get_composed_transform(aug=False)
        base_transform_test = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)
        if args.base_split is not None:
            base_dataset = tiered_ImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
    elif args.base_dataset == 'ImageNet':
        if args.base_no_color_jitter:
            base_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:  
            warnings.warn("Using ImageNet with Color Jitter")  
            base_transform = ImageNet_few_shot.TransformLoader(
                args.image_size).get_composed_transform(aug=True)
        base_transform_test = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)

        if args.base_split is not None:
            base_dataset = ImageNet_few_shot.construct_subset(base_dataset, args.base_split)
        print("Size of Base dataset:", len(base_dataset))
    else:
        raise ValueError("Invalid base dataset!")

    # create the target dataset
    if args.target_dataset == 'ISIC':
        transform = ISIC_few_shot.TransformLoader(args.image_size).get_composed_transform(aug=True)
        transform_test = ISIC_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = ISIC_few_shot.SimpleDataset(transform, split=args.target_subset_split)
    elif args.target_dataset == 'EuroSAT':
        transform = EuroSAT_few_shot.TransformLoader(args.image_size).get_composed_transform(aug=True)
        transform_test = EuroSAT_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = EuroSAT_few_shot.SimpleDataset(transform, split=args.target_subset_split)
    elif args.target_dataset == 'CropDisease':
        transform = CropDisease_few_shot.TransformLoader(args.image_size).get_composed_transform(aug=True)
        transform_test = CropDisease_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = CropDisease_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'ChestX':
        transform = Chest_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = Chest_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = Chest_few_shot.SimpleDataset(transform, split=args.target_subset_split)
    elif args.target_dataset == 'cars':
        transform = cars_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = cars_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = cars_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'cub':
        transform = cub_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = cub_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = cub_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'places':
        transform = places_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = places_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = places_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'plantae':
        transform = plantae_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = plantae_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = plantae_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'miniImageNet_test':
        transform = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = miniImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'tiered_ImageNet_test':
        if args.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        transform = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = tiered_ImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    else:
        raise ValueError('Invalid dataset!')

    # pseudolabel target dataset
    dataset = pseudolabel_dataset(backbone, clf_teacher, dataset,
                       transform, transform_test, args)

    print("Size of target dataset", len(dataset))
    dataset_test = copy.deepcopy(dataset)

    transform_twice = apply_twice(transform)
    transform_test_twice = apply_twice(transform_test, transform)

    dataset.d.transform = transform_twice
    dataset_test.d.transform = transform_test_twice

    ind = torch.randperm(len(dataset))

    # initialize the student's backbone with random weights
    if args.backbone_random_init:
        backbone.module.load_state_dict(backbone_sd_init)

    # split the target dataset into train and val
    # 10% of the unlabeled data is used for validation
    train_ind = ind[:int(0.9*len(ind))]
    val_ind = ind[int(0.9*len(ind)):]    

    trainset = torch.utils.data.Subset(dataset, train_ind)
    valset = torch.utils.data.Subset(dataset_test, val_ind)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize,
                                              num_workers=args.num_workers,
                                              shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.bsize,
                                              num_workers=args.num_workers,
                                              shuffle=False, drop_last=False)

    # Generate trainset and valset for base dataset
    base_ind = torch.randperm(len(base_dataset))

    base_train_ind = base_ind[:int((1 - args.base_val_ratio)*len(base_ind))]
    base_val_ind = base_ind[int((1 - args.base_val_ratio)*len(base_ind)):]

    
    base_dataset_val = copy.deepcopy(base_dataset)
    base_dataset_val.transform = base_transform_test

    base_trainset = torch.utils.data.Subset(base_dataset, base_train_ind)
    base_valset = torch.utils.data.Subset(base_dataset_val, base_val_ind)

    print("Size of base validation set", len(base_valset))

    base_trainloader = torch.utils.data.DataLoader(base_trainset, batch_size=args.bsize,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, drop_last=True)
    base_valloader = torch.utils.data.DataLoader(base_valset, batch_size=args.bsize * 2,
                                            num_workers=args.num_workers,
                                            shuffle=False, drop_last=False)
    ############################

    ############################
    # Specify Loss Function
    ############################
    criterion = NTXentLoss('cuda', args.bsize, args.temp, True)
    if args.batch_validate:
        criterion_val = NTXentLoss('cuda', args.bsize, args.temp, True)
    else:
        criterion_val = NTXentLoss('cuda', len(valset), args.temp, True)
    ############################

    ###########################
    # Create Optimizer
    ###########################

    optimizer = torch.optim.SGD([
            {'params': backbone.parameters()},
            {'params': clf.parameters()},
            {'params': clf_SIMCLR.parameters()}
        ],
            lr=0.1, momentum=0.9,
            weight_decay=args.wd,
            nesterov=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min', factor=0.5,
                                                           patience=10, verbose=False, 
                                                           cooldown=10,
                                                           threshold_mode='rel', 
                                                           threshold=1e-4, min_lr=1e-5)

    #######################################
    starting_epoch = 0

    # whether to resume from the latest checkpoint
    if args.resume_latest:
        import re
        pattern = "checkpoint_(\d+).pkl"
        candidate = []
        for i in os.listdir(args.dir):
            match = re.search(pattern, i)
            if match:
                candidate.append(int(match.group(1)))

        # if nothing found, then start from scratch
        if len(candidate) == 0:
            print('No latest candidate found to resume!')
            logger.info('No latest candidate found to resume!')
        else:
            latest = np.amax(candidate)
            load_path = os.path.join(args.dir, f'checkpoint_{latest}.pkl')
            if latest >= args.epochs:
                print('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                logger.info('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                import sys
                sys.exit(0)
            else:
                best_model_path = os.path.join(args.dir, 'checkpoint_best.pkl')

                # first load the previous best model
                best_epoch = load_checkpoint(backbone, clf, clf_SIMCLR, 
                            optimizer, scheduler, best_model_path)

                logger.info('Latest model epoch: {}'.format(latest))

                logger.info(
                    'Validate the best model checkpointed at epoch: {}'.format(best_epoch))

                # Validate to set the right loss
                performance_val = validate(backbone, clf,  clf_SIMCLR, 
                                            valloader, base_valloader, criterion_val, 
                                           best_epoch, args.epochs, logger, vallog, args, postfix='Validation')

                loss_val = performance_val['Loss_test/avg']
                error_val = 100 - performance_val['top1_test_per_class/avg']

                best_error = error_val
                best_loss = loss_val

                sd_best = torch.load(os.path.join(
                    args.dir, 'checkpoint_best.pkl'))

                if latest > best_epoch:

                    starting_epoch = load_checkpoint(
                        backbone, clf, clf_SIMCLR, optimizer, scheduler, load_path)
                else:
                    starting_epoch = best_epoch

                logger.info(
                    'Continue Training at epoch: {}'.format(starting_epoch))

    
    
    
    
    ###########################################
    ####### Learning rate test ################
    ###########################################
    if starting_epoch == 0:
        ### Start by doing a learning rate test 
        lr_candidates = [1e-1, 5e-2, 3e-2, 1e-2, 5e-3, 3e-3, 1e-3]

        step = 50

        # number of training epochs to get at least 50 updates
        warm_up_epoch = math.ceil(step / len(trainloader))

        # keep track of the student model initialization
        # Need to keep reloading when testing different learning rates
        sd_current = copy.deepcopy(backbone.state_dict())
        sd_head = copy.deepcopy(clf.state_dict())
        sd_head_SIMCLR = copy.deepcopy(clf_SIMCLR.state_dict())


        vals = []

        # Test the learning rate by training for one epoch
        for current_lr in lr_candidates:
            lr_log = utils.savelog(args.dir, f'lr_{current_lr}')

            # reload the student model
            backbone.load_state_dict(sd_current)
            clf.load_state_dict(sd_head)
            clf_SIMCLR.load_state_dict(sd_head_SIMCLR)

            # create the optimizer
            optimizer = torch.optim.SGD([
                {'params': backbone.parameters()},
                {'params': clf.parameters()},
                {'params': clf_SIMCLR.parameters()}
            ],
                lr=current_lr, momentum=0.9,
                weight_decay=args.wd,
                nesterov=False)

            logger.info(f'*** Testing Learning Rate: {current_lr}')

            # training for a bit
            for i in range(warm_up_epoch):
                perf = train(backbone, clf, clf_SIMCLR, optimizer, 
                        trainloader, base_trainloader, criterion,
                        i, warm_up_epoch, logger, lr_log, args, turn_off_sync=True)

            # compute the validation loss for picking learning rates
            perf_val = validate(backbone, clf, clf_SIMCLR, valloader, 
                            base_valloader, criterion_val, 
                            1, 1, logger, vallog, args, postfix='Validation', 
                        turn_off_sync=True)
            vals.append(perf_val['Loss_test/avg'])

        # pick the best learning rates
        current_lr = lr_candidates[int(np.argmin(vals))]

        # reload the models 
        backbone.load_state_dict(sd_current)
        clf.load_state_dict(sd_head)
        clf_SIMCLR.load_state_dict(sd_head_SIMCLR)
            
        logger.info(f"** Learning with lr: {current_lr}")
        optimizer = torch.optim.SGD([
            {'params': backbone.parameters()},
            {'params': clf.parameters()},
            {'params': clf_SIMCLR.parameters()}
        ],
            lr=current_lr, momentum=0.9,
            weight_decay=args.wd,
            nesterov=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min', factor=0.5,
                                                            patience=10, verbose=False, 
                                                            cooldown=10,
                                                            threshold_mode='rel', 
                                                            threshold=1e-4, min_lr=1e-5)

        scheduler.step(math.inf)

        best_loss = math.inf
        best_epoch = 0
        checkpoint(backbone, clf, clf_SIMCLR, 
             optimizer, scheduler, os.path.join(
            args.dir, f'checkpoint_best.pkl'), 0)
    
    ############################
    # save the initialization
    checkpoint(backbone, clf, clf_SIMCLR, 
                optimizer, scheduler, 
               os.path.join(
                   args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch)

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):
            perf = train(backbone, clf, clf_SIMCLR, optimizer, trainloader, 
                        base_trainloader, criterion,
                         epoch, args.epochs, logger, trainlog, args)

            scheduler.step(perf['Loss/avg'])


            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(backbone, clf, clf_SIMCLR, 
                        optimizer, scheduler, 
                           os.path.join(
                               args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)

            if (epoch == starting_epoch) or ((epoch + 1) % args.eval_freq == 0):
                performance_val = validate(backbone, clf, clf_SIMCLR, valloader,
                            base_valloader, criterion_val,  
                            epoch+1, args.epochs, logger, vallog, args, postfix='Validation')
                
                loss_val = performance_val['Loss_test/avg']

                if best_loss > loss_val:
                    best_epoch = epoch + 1
                    checkpoint(backbone, clf, clf_SIMCLR, 
                        optimizer, scheduler, os.path.join(
                        args.dir, f'checkpoint_best.pkl'), best_epoch)
                    logger.info(
                        f"*** Best model checkpointed at Epoch {best_epoch}")
                    best_loss = loss_val

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(backbone, clf, clf_SIMCLR, 
                    optimizer, scheduler, os.path.join(
                args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)
    finally:
        trainlog.save()
        vallog.save()
    return


def checkpoint(model, clf, clf_SIMCLR, optimizer, scheduler, save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    sd = {
        'model': copy.deepcopy(model.module.state_dict()),
        'clf': copy.deepcopy(clf.state_dict()),
        'clf_SIMCLR': copy.deepcopy(clf_SIMCLR.state_dict()),
        'opt': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'epoch': epoch
    }

    torch.save(sd, save_path)
    return sd


def load_checkpoint(model, clf, clf_SIMCLR, optimizer, scheduler, load_path):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path)
    model.module.load_state_dict(sd['model'])
    clf.load_state_dict(sd['clf'])
    clf_SIMCLR.load_state_dict(sd['clf_SIMCLR'])
    optimizer.load_state_dict(sd['opt'])
    scheduler.load_state_dict(sd['scheduler'])
    
    return sd['epoch']




def train(model, clf, clf_SIMCLR, 
    optimizer, trainloader, base_trainloader, criterion_SIMCLR, epoch, 
    num_epochs, logger, trainlog, args, turn_off_sync=False):

    meters = utils.AverageMeterSet()
    model.train()
    clf.train()
    clf_SIMCLR.train()

    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    nll_criterion = nn.NLLLoss(reduction='mean')

    base_loader_iter = iter(base_trainloader)

    end = time.time()
    for i, ((X1, X2), y) in enumerate(trainloader):
        meters.update('Data_time', time.time() - end)

        current_lr = optimizer.param_groups[0]['lr']
        meters.update('lr', current_lr, 1)

        X1 = X1.cuda()
        X2 = X2.cuda()
        y = y.cuda()

        # Get the data from the base dataset
        try:
            X_base, y_base = base_loader_iter.next()
        except StopIteration:
            base_loader_iter = iter(base_trainloader)
            X_base, y_base = base_loader_iter.next()
        
        X_base = X_base.cuda()
        y_base = y_base.cuda()

        optimizer.zero_grad()

        # cross entropy loss on the base dataset
        features_base = model(X_base)
        logits_base = clf(features_base)
        log_probability_base = F.log_softmax(logits_base, dim=1)
        loss_base = nll_criterion(log_probability_base, y_base)

        f1 = model(X1)
        f2 = model(X2)

        # SIMCLR Loss on the target dataset
        z1 = clf_SIMCLR(f1)
        z2 = clf_SIMCLR(f2)

        loss_SIMCLR = criterion_SIMCLR(z1, z2)

        # Pseudolabel loss on the target dataset
        logits_xtask_1 = clf(f1)
        logits_xtask_2 = clf(f2)
        log_probability_1 = F.log_softmax(logits_xtask_1, dim=1)
        log_probability_2 = F.log_softmax(logits_xtask_2, dim=1)

        loss_xtask = (kl_criterion(log_probability_1, y) +
                      kl_criterion(log_probability_2, y)) / 2
        

        loss = loss_base + loss_SIMCLR + loss_xtask

        loss.backward()
        optimizer.step()

        meters.update('Loss', loss.item(), 1)
        meters.update('KL_Loss_target', loss_xtask.item(), 1)
        meters.update('CE_Loss_source', loss_base.item(), 1)
        meters.update('SIMCLR_Loss_target', loss_SIMCLR.item(), 1)

        perf = utils.accuracy(logits_xtask_1.data,
                              y.argmax(dim=1).data, topk=(1, ))

        meters.update('top1', perf['average'][0].item(), len(X1))
        meters.update('top1_per_class', perf['per_class_average'][0].item(), 1)
        
        perf_base = utils.accuracy(logits_base.data,
                              y_base.data, topk=(1, ))

        meters.update('top1_base', perf_base['average'][0].item(), len(X_base))

        meters.update('top1_base_per_class', perf_base['per_class_average'][0].item(), 1)
        

        meters.update('Batch_time', time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            values = meters.values()
            averages = meters.averages()
            sums = meters.sums()

            logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}] '
                             'Batch Time: {meters[Batch_time]:.4f} '
                             'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                             'Average KL Loss (Target): {meters[KL_Loss_target]:.4f} '
                             'Average SimCLR Loss (Target): {meters[SIMCLR_Loss_target]:.4f} '
                             'Average CE Loss (Source): {meters[CE_Loss_source]: .4f} '
                             'Learning Rate: {meters[lr]:.4f} '
                             'Top1: {meters[top1]:.4f} '
                             'Top1_per_class: {meters[top1_per_class]:.4f} '
                             'Top1_base: {meters[top1_base]:.4f} '
                             'Top1_base_per_class: {meters[top1_base_per_class]:.4f} '
                             ).format(
                epoch=epoch, epochs=num_epochs, step=i+1, steps=len(trainloader), meters=meters)

            logger.info(logger_string)

        if (args.iteration_bp is not None) and (i+1) == args.iteration_bp:
            break

    logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step}] Batch Time: {meters[Batch_time]:.4f} '
                     'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                     'Average KL Loss (Target): {meters[KL_Loss_target]:.4f} '
                     'Average SimCLR Loss (Target): {meters[SIMCLR_Loss_target]:.4f} '
                     'Average CE Loss (Source): {meters[CE_Loss_source]: .4f} '
                     'Learning Rate: {meters[lr]:.4f} '
                     'Top1: {meters[top1]:.4f} '
                     'Top1_per_class: {meters[top1_per_class]:.4f} '
                     'Top1_base: {meters[top1_base]:.4f} '
                     'Top1_base_per_class: {meters[top1_base_per_class]:.4f} '
                     ).format(
        epoch=epoch+1, epochs=num_epochs, step=0, meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    trainlog.record(epoch+1, {
        **values,
        **averages,
        **sums
    })

    if not turn_off_sync:
        wandb.log({'loss': averages['Loss/avg']}, step=epoch+1)
        wandb.log(
            {'ce_loss_source': averages['CE_Loss_source/avg']}, step=epoch+1)
        wandb.log(
            {'kl_loss_target': averages['KL_Loss_target/avg']}, step=epoch+1)
        wandb.log(
            {'SimCLR_loss_target': averages['SIMCLR_Loss_target/avg']}, step=epoch+1)
        wandb.log({'top1': averages['top1/avg'],
                'top1_per_class': averages['top1_per_class/avg'],
                }, step=epoch+1)
        wandb.log({'top1_base': averages['top1_base/avg'],
                   'top1_base_per_class': averages['top1_base_per_class/avg'],
                   }, step=epoch+1)

    return averages


def validate(model, clf, clf_simclr, 
             testloader, base_loader, criterion_SIMCLR, epoch, num_epochs, logger,
            testlog, args, postfix='Validation', turn_off_sync=False):
    meters = utils.AverageMeterSet()
    model.eval()
    clf.eval()
    clf_simclr.eval()

    criterion_xtask = nn.KLDivLoss(reduction='batchmean')
    nll_criterion = nn.NLLLoss(reduction='mean')

    logits_xtask_test_all = []

    if args.batch_validate:
        losses_SIMCLR = []
    else:
        z1s = []
        z2s = []

    ys_all = []

    end = time.time()
    # Compute the loss for the target dataset
    with torch.no_grad():
        for _, ((Xtest, Xrand), y) in enumerate(testloader):
            Xtest = Xtest.cuda()
            Xrand = Xrand.cuda()
            y = y.cuda()

            ftest = model(Xtest)
            frand = model(Xrand)

            ztest = clf_simclr(ftest)
            zrand = clf_simclr(frand)

            # get the logits for xtask
            logits_xtask_test = clf(ftest)
            logits_xtask_test_all.append(logits_xtask_test)
            ys_all.append(y)

            if args.batch_validate:
                if len(Xtest) != args.bsize:
                    criterion_small_set = NTXentLoss(
                        'cuda', len(Xtest), args.temp, True)
                    losses_SIMCLR.append(criterion_small_set(ztest, zrand))
                else:
                    losses_SIMCLR.append(criterion_SIMCLR(ztest, zrand))
            else:
                z1s.append(ztest)
                z2s.append(zrand)
            

    ys_all = torch.cat(ys_all, dim=0)
    logits_xtask_test_all = torch.cat(logits_xtask_test_all, dim=0)

    log_probability = F.log_softmax(logits_xtask_test_all, dim=1)

    loss_xtask = criterion_xtask(log_probability, ys_all)

    if args.batch_validate:
        loss_SIMCLR = torch.stack(losses_SIMCLR).mean()
    else:

        z1s = torch.cat(z1s, dim=0)
        z2s = torch.cat(z2s, dim=0)
        loss_SIMCLR = criterion_SIMCLR(z1s, z2s)

    logits_base_all = []
    ys_base_all = []
    with torch.no_grad():
        # Compute the loss on the source base dataset
        for X_base, y_base in base_loader:
            X_base = X_base.cuda()
            y_base = y_base.cuda()

            features = model(X_base)
            logits_base = clf(features)

            logits_base_all.append(logits_base)
            ys_base_all.append(y_base)
    
    ys_base_all = torch.cat(ys_base_all, dim=0)
    logits_base_all = torch.cat(logits_base_all, dim=0)

    log_probability_base = F.log_softmax(logits_base_all, dim=1)

    loss_base = nll_criterion(log_probability_base, ys_base_all)

    loss = loss_xtask + loss_SIMCLR + loss_base

    meters.update('CE_Loss_source_test', loss_base.item(), 1)
    meters.update('KL_Loss_target_test', loss_xtask.item(), 1)
    meters.update('SIMCLR_Loss_target_test', loss_SIMCLR.item(), 1)
    meters.update('Loss_test', loss.item(), 1)

    perf = utils.accuracy(logits_xtask_test_all.data,
                          ys_all.argmax(dim=1).data, topk=(1, ))

    meters.update('top1_test', perf['average'][0].item(), 1)
    meters.update('top1_test_per_class',
                  perf['per_class_average'][0].item(), 1)

    perf_base = utils.accuracy(logits_base_all.data,
                          ys_base_all.data, topk=(1, ))

    meters.update('top1_base_test', perf_base['average'][0].item(), 1)
    meters.update('top1_base_test_per_class',
                  perf_base['per_class_average'][0].item(), 1)

    meters.update('Batch_time', time.time() - end)

    logger_string = ('{postfix} Epoch: [{epoch}/{epochs}]  Batch Time: {meters[Batch_time]:.4f} '
                    'Average Test Loss: {meters[Loss_test]:.4f} '
                     'Average Test KL Loss (Target): {meters[KL_Loss_target_test]: .4f} '
                     'Average Test SimCLR Loss (Target): {meters[SIMCLR_Loss_target_test]: .4f} '
                     'Average CE Loss (Source): {meters[CE_Loss_source_test]: .4f} '
                     'Top1_test: {meters[top1_test]:.4f} '
                     'Top1_test_per_class: {meters[top1_test_per_class]:.4f} '
                     'Top1_base_test: {meters[top1_base_test]:.4f} '
                     'Top1_base_test_per_class: {meters[top1_base_test_per_class]:.4f} ').format(
        postfix=postfix, epoch=epoch, epochs=num_epochs, meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    testlog.record(epoch, {
        **values,
        **averages,
        **sums
    })

    if postfix != '':
        postfix = '_' + postfix

    if not turn_off_sync:
        wandb.log({'loss' + postfix: averages['Loss_test/avg']}, step=epoch)
        wandb.log(
            {'kl_loss_target' + postfix: averages['KL_Loss_target_test/avg']}, step=epoch)
        wandb.log(
            {'ce_loss_source' + postfix: averages['CE_Loss_source_test/avg']}, step=epoch)
        wandb.log(
            {'SimCLR_loss_target': averages['SIMCLR_Loss_target_test/avg']}, step=epoch)
        wandb.log({'top1' + postfix: averages['top1_test/avg'],
               'top1_per_class' + postfix: averages['top1_test_per_class/avg'],
              }, step=epoch)
        wandb.log({'top1_base' + postfix: averages['top1_base_test/avg'],
                   'top1_base_per_class' + postfix: averages['top1_base_test_per_class/avg'],
                   }, step=epoch)

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STARTUP')
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to save the checkpoints')
    
    parser.add_argument('--bsize', type=int, default=32,
                        help='batch_size for STARTUP')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Frequency (in epoch) to evaluate on the val set')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay for the model')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    
    parser.add_argument('--iteration_bp', type=int, help='which step to break in the training loop')
    parser.add_argument('--model', type=str, default='resnet10',
                        help='Backbone model')


    parser.add_argument('--teacher_path', type=str, required=True, 
                        help='path to the teacher model')
    parser.add_argument('--teacher_path_version', type=int, default=1,
                        help='how to load the teacher')

    parser.add_argument('--use_pretrained_clf', action='store_true',
                        help="whether to initialize student's classifier (this is the teacher classifier)")
    parser.add_argument('--backbone_random_init', action='store_true',
                        help="Use random initialized backbone ")
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection Dimension for SimCLR')
    parser.add_argument('--temp', type=float, default=1,
                        help='Temperature of SIMCLR')


    parser.add_argument('--base_dataset', type=str, required=True, help='base_dataset to use')
    parser.add_argument('--base_path', type=str, required=True, help='path to base dataset')
    parser.add_argument('--base_split', type=str, help='split for the base dataset')
    parser.add_argument('--base_no_color_jitter', action='store_true', help='remove color jitter for ImageNet')
    parser.add_argument('--base_val_ratio', type=float, default=0.05, help='amount of base dataset set aside for validation')
    
    parser.add_argument('--batch_validate', action='store_true',
                        help='to do batch validate rather than validate on the full dataset (Ideally, for SimCLR,'+
                        ' the validation should be on the full dataset but might not be feasible due to hardware constraints')

    parser.add_argument('--target_dataset', type=str, required=True,
                        help='the target domain dataset')
    parser.add_argument('--target_subset_split', type=str,
                        help='path to the csv files that specifies the unlabeled split for the target dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')
    
    args = parser.parse_args()
    main(args)
