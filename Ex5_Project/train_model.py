"""
Author: Hao Zheng
Matr.Nr.: K01608113
Exercise 5
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Imports from other python files
from datasets import new_ImageDataset, new_GridImageSet
from model_architecture import myCNN
from utils import plot, evaluate_model

input_path = "training"

dataset_folder = "training"
# resize images to test image shape (100, 100)
im_shape = 100
resize_transforms = transforms.Compose([
    transforms.Resize(size=im_shape),
    transforms.CenterCrop(size=(im_shape, im_shape)),
])

# network = {}


# learningrate = 1e-3
# weight_decay = 1e-5
# results_path = "results"
# n_updates = 50000
# n_updates = 2000

'''
File for training the model, most of code is based on the example project and was adjusted our dataset and my model.
For training the model, certain training parameters, folderpaths can be adjusted in the working_config.json.

The training images must be in a folder "training" in the same directory as train_model.py(same goes for working_config.json).

To start the training, following command should be run in the terminal:
python3 train_model.py working_config.json

The training loss can be monitored with tensorboard.


'''


def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 50_000, device: torch.device = torch.device("cuda:0")):
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    # Make path for plots
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)
    # Load paths to training images
    image_dataset = new_ImageDataset("training")

    # Assign 1/5th of our samples to a test set, 1/5th to a validation set, and
    # the remaining 3/5th to a training set using  random splits.
    n_samples = len(image_dataset)
    shuffled_indices = np.random.permutation(n_samples)
    testset_inds = shuffled_indices[:int(n_samples / 5)]
    validationset_inds = shuffled_indices[int(n_samples / 5):int(n_samples / 5) * 2]
    trainingset_inds = shuffled_indices[int(n_samples / 5) * 2:]

    # Create PyTorch subsets from our subset indices
    testset = Subset(image_dataset, indices=testset_inds)
    validationset = Subset(image_dataset, indices=validationset_inds)
    trainingset = Subset(image_dataset, indices=trainingset_inds)
    # Create datasets and dataloaders with inputs & targets created by ex4
    trainingset_eval = new_GridImageSet(dataset=trainingset)
    validationset = new_GridImageSet(dataset=validationset)
    testset = new_GridImageSet(dataset=testset)

    trainloader = DataLoader(trainingset_eval, batch_size=6, shuffle=False, num_workers=0)
    valloader = DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Load model from model_architecture_py
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = myCNN(**network_config)
    model.to(device)

    # Define loss function: mse ( metric on server is RMSE = mse^(1/2)
    mse = torch.nn.MSELoss()

    # Use adam optimizer as optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 100  # plot every x updates
    validate_at = 200  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    # Save model timestamp to keep track of different models
    timestemp = str(datetime.now())
    timestemp = timestemp.split(".")[0].replace(" ", "_")
    saved_model_file = os.path.join(results_path, timestemp + "_model.pt")
    torch.save(model, saved_model_file)

    # Train until n_updates updates have been reached
    while update < n_updates:
        for i, (data) in enumerate(trainloader):
            # Get next samples
            inputs, targets, ids = data

            inputs = inputs.to(device)
            targets = targets.to(device)
            # target_array = target_array.to(device)
            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = model(inputs)

            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            # Plot output
            # polot(input, target, prediction)
            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plotpath, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(model, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(model, saved_model_file)

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
    print(f"Computing scores for model" + timestemp)
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, dataloader=testloader, loss_fn=mse, device=device)

    # Print out loss for training/validation/test
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    # Write results to log file for comparison of different models
    result_text = f"\n#############Scores Model " + timestemp + "\n" + f"Training Loss : {train_loss} \nValidation Loss: {val_loss} \nTest Loss: {test_loss}"
    extra_info = "\n" + str(
        network_config) + "\n" + f"Number of Images used: {len(image_dataset)}, validate_at = {validate_at}, n updates: {n_updates}"
    # Write result to file

    file_object = open('model_logs', 'a')
    file_object.write(result_text + extra_info)
    file_object.close()


#############################
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
