"""
Author: Hao Zheng
Matr.Nr.: K01608113
Exercise 5
"""
import random
import os
import numpy as np
import torch.utils.data
import dill as pkl
import datetime
import pandas as pd

from utils import convert_predictions


def main(modelname: str):
    '''
    File for applying the trained model on the unknown test data:
    Run python3 test_model.py Name_of_Model_file
    e.g. Name_of_Model_file = best_model.pt
    '''

    input_path = "results"
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    map_location = torch.device('cpu')
    # open test files from pkl file
    testfiles = pd.read_pickle(r'inputs.pkl')

    n_files = len(testfiles["sample_ids"])

    current_dir = os.path.abspath(os.getcwd())

    # Put model_name here
    # model_name = "2022-07-13_12_13_01_model.pt"
    model_path = os.path.join(current_dir, input_path + "/" + modelname)
    # Load model
    model = torch.load(model_path, map_location="cpu")

    # print(current_dir)

    print("### Start making predictions ###")

    ###### Apply Model on Test files and save predictions #####

    # List for saving model
    all_results = []

    for i in range(n_files):
        # convert input array and known array to torch.tensor
        input_array = testfiles["input_arrays"][i] / 255
        input_array = torch.from_numpy(input_array)
        known_array = testfiles["known_arrays"][i]
        known_tensor = torch.from_numpy(np.array([known_array[0]]))
        # concatenate input and known array to have correct dimensions for the model
        modelinput = torch.concat((input_array, known_tensor), dim=0).float()
        # apply model for predictions
        output = model(modelinput)
        # convert output back to np array
        result = output.cpu().detach().numpy()

        # convert result in 1D array - RRRGGGBBBB shape
        final_result = convert_predictions(result, known_array)
        all_results.append(final_result)

        print(f"{i + 1}/{n_files}")

    # Use timestamp for naming of the outputs, so they are not overwritten when different models are tested
    timestamp = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    file_name = timestamp + "_outputs.pkl"
    with open(file_name, 'wb') as file:
        pkl.dump(all_results, file=file)

    print("Making predictions with model " + modelname + " finished.")


if __name__ == "__main__":
    import sys

    modelname = sys.argv[1]

    print(f"Using " + modelname + " as model:")
    main(modelname)
