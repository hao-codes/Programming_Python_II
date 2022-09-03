"""
Author: Hao Zheng
Matr.Nr.: K01608113
Exercise 5
"""
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os

from matplotlib import pyplot as plt


input_path = "training"
im_shape = 100
resize_transforms = transforms.Compose([
    transforms.Resize(size=im_shape),
    transforms.CenterCrop(size=(im_shape, im_shape)),
])

'''evalute_model and the plot function are both taken from the example project '''


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

            inputs, targets_whole, file_names = data
            inputs = inputs.to(device)
            # targets = targets.to(device)
            targets_whole = targets_whole.to(device)
            # Get outputs of the specified model
            outputs = model(inputs)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets_whole).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train()
    return loss


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    '''
    Plotting function taken from the example project(grayscale images) and slightly adjusted for our RGB images:
    This function helps to compare our predictions with the original images to see how our model is doing during the training
    '''
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            temp = data[i, 0]  # * 255
            ax.imshow(np.transpose(data[i], (1, 2, 0))[:, :, 0:3], interpolation="none")
            # ax.imshow(temp, cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)

    plt.close(fig)


def convert_predictions(predictions: np.array, known_array: np.array):
    '''Function converts our predictions to the required format (1D np array with order: all R values, all G values, all B values of off-grid locations) '''

    # convert model_output back to range [0;255]
    arr = np.array(predictions * 255, dtype=int)
    # get indices of all values in the known array equal to zero
    unknown_indices = np.array(np.where(known_array[0] == 0))
    # create an array
    formatted_predictions = np.empty((predictions.shape[0], len(unknown_indices[0])), dtype=np.uint8)

    # go through all unknown indices
    for i, index_index in enumerate(range(unknown_indices.shape[1])):
        formatted_predictions[:, i] = arr[:, unknown_indices[0, index_index], unknown_indices[1, index_index]]
    # flatten the array like the target array in ex4 to 1D
    return formatted_predictions.flatten()

# print("utils")
