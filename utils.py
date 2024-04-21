import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

import sklearn.metrics


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)

    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    classes = train_set.classes

    return train_loader, test_loader, classes


def prepare_flowers102_dataloader(num_workers=8,train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3),
        torchvision.transforms.RandomAffine(degrees=30, shear=20),
        transforms.ToTensor(),
        torchvision.transforms.Resize((256, 256))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Resize((256, 256))
    ])

    train_set = torchvision.datasets.Flowers102(
        root="data",
        split="train",
        transform=train_transform,
        download=True)
    
    test_set = torchvision.datasets.Flowers102(
        root="data",
        split="test",
        transform=test_transform,
        download=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers)

    classes = 102

    return train_loader, test_loader, classes
    




def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


def create_classification_report(model, device, test_loader):

    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            y_true += data[1].numpy().tolist()
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().numpy().tolist()

    classification_report = sklearn.metrics.classification_report(
        y_true=y_true, y_pred=y_pred)

    return classification_report


def train_model(model,
                train_loader,
                test_loader,
                device,
                l2_regularization_strength=1e-4,
                learning_rate=1e-1,
                num_epochs=200,
                use_amp=False):
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=l2_regularization_strength)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        0, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):

        # Record the start time of the training epoch.
        # start_time = time.time()

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        accumulated_time = 0

        torch.cuda.synchronize()
        start_time = time.time()

        for inputs, labels in train_loader:

            # torch.cuda.synchronize()
            # start_time = time.time()

            # The data transfer takes ~100 ms on my machine.
            inputs = inputs.to(device)
            labels = labels.to(device)

            # torch.cuda.synchronize()
            # end_time = time.time()
            # accumulated_time += end_time - start_time

            # torch.cuda.synchronize()
            # start_time = time.time()
            # end_time = time.time()
            # accumulated_time += end_time - start_time

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)


            # outputs = model(inputs)
            # loss = criterion(outputs, labels)

            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # torch.cuda.synchronize()
            # end_time = time.time()
            # accumulated_time += end_time - start_time

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # Record the end time of the training epoch.
        torch.cuda.synchronize()
        end_time = time.time()
        accumulated_time += end_time - start_time

        # Compute the elapsed training time.
        elapsed_time = accumulated_time

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        eval_loss = 0
        eval_accuracy = 0
        # model.eval()
        # eval_loss, eval_accuracy = evaluate_model(model=model,
        #                                           test_loader=test_loader,
        #                                           device=device,
        #                                           criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        print(
            "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f} Elapsed Time: {:.0f}ms"
            .format(epoch + 1, train_loss, train_accuracy, eval_loss,
                    eval_accuracy, elapsed_time * 1000))

    return model


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def create_model(num_classes=10, model_func=torchvision.models.resnet18):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = model_func(num_classes=num_classes, pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model