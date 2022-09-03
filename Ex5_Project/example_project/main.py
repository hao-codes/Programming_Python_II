# -*- coding: utf-8 -*-
"""example_project/main.py

Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 15.04.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Main file of example project.
"""

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import SimpleCNN
from datasets import CIFAR10, RotatedImages
from utils import plot


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, file_names = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get outputs of the specified model
            outputs = model(inputs)
            
            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
            
            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train()
    return loss


def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 50_000, device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Prepare a path to plot to
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)
    
    # Load or download CIFAR10 dataset
    cifar10_dataset = CIFAR10(data_folder="cifar10")
    
    # Split dataset into training, validation and test set (CIFAR10 dataset
    # is already randomized, so we do not necessarily have to shuffle again)
    trainingset = torch.utils.data.Subset(
        cifar10_dataset,
        indices=np.arange(int(len(cifar10_dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(
        cifar10_dataset,
        indices=np.arange(int(len(cifar10_dataset) * (3 / 5)), int(len(cifar10_dataset) * (4 / 5))))
    testset = torch.utils.data.Subset(
        cifar10_dataset,
        indices=np.arange(int(len(cifar10_dataset) * (4 / 5)), len(cifar10_dataset)))

    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    trainingset_eval = RotatedImages(dataset=trainingset, rotation_angle=45)
    validationset = RotatedImages(dataset=validationset, rotation_angle=45)
    testset = RotatedImages(dataset=testset, rotation_angle=45)
    trainloader = torch.utils.data.DataLoader(trainingset_eval, batch_size=1, shuffle=False, num_workers=0)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create datasets and dataloaders with rotated targets with augmentation (for training)
    trainingset_augmented = RotatedImages(dataset=trainingset, rotation_angle=45,
                                          transform_chain=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                              transforms.RandomVerticalFlip()]))
    trainloader_augmented = torch.utils.data.DataLoader(trainingset_augmented, batch_size=16, shuffle=True,
                                                        num_workers=0)
    
    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 10_000  # plot every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)
    
    # Train until n_updates updates have been reached
    while update < n_updates:
        for data in trainloader_augmented:
            # Get next samples
            inputs, targets, ids = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Get outputs of our network
            outputs = net(inputs)
            
            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
            
            # Plot output
            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plotpath, update)
            
            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(), global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)
            
            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()
            
            # Increment update counter, exit if maximum number of updates is reached
            # Here, we could apply some early stopping heuristic and also exit if its
            # stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")
    
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, dataloader=testloader, loss_fn=mse, device=device)
    
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")
    
    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
