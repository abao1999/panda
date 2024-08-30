import os
import argparse

from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split, TestData
from chronos_dysts.utils import (
    load_and_split_dataset_from_arrow,
    get_dyst_filepaths,
    generate_sample_forecasts,
)

def simple_test_split():
    prediction_length = 4
    offset = -4
    num_rolls = 1
    distance = None # if None specified, distance is set to prediction_length

    # Create a sample dataset
    dataset = ListDataset(
        [
            {"start": "2020-01-01", "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            {"start": "2020-01-01", "target": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
            {"start": "2020-01-01", "target": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]},
        ],
        freq="h"
    )

    # Split the dataset with an offset of 5
    train_dataset, test_template = split(dataset, offset=offset)

    # Generate test instances with a prediction length of 2
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=distance)

    # Print the results
    print("Train Dataset:")
    for entry in train_dataset:
        print(entry)

    print("\nTest Data:")
    dim_idx = 0
    idx = 0
    for test_pair in test_data:
        if (idx) % num_rolls == 0:
            print("DIM: ", dim_idx)
            dim_idx += 1
        print("--- window: ", idx % num_rolls)
        # input is data seen by the model, label is the ground truth to predict
        input, label = test_pair
        input_data = input["target"]
        label_data = label["target"]
        print("input data: ", input_data)
        print("label data: ", label_data)
        print("input shape: ", input_data.shape)
        print("label shape: ", label_data.shape)
        idx += 1

def test_split_from_arrow(filepath):
    num_rolls = 1
    # test_data is of type TestData
    test_data = load_and_split_dataset_from_arrow(
        prediction_length=512,
        offset=-512,
        num_rolls=num_rolls,
        filepath=filepath,
        verbose=True
    )
    print("Test Data:")
    print(len(test_data))
    print("Input test data: ", test_data.input.test_data)
    print("Label test data: ", test_data.label.test_data)

    dim_idx = 0
    idx = 0
    for test_pair in test_data:
        if (idx) % num_rolls == 0:
            print("DIM: ", dim_idx)
            dim_idx += 1
        print("--- window: ", idx % num_rolls)
        # input is data seen by the model, label is the ground truth to predict
        input, label = test_pair
        input_data = input["target"]
        label_data = label["target"]
        print("input shape: ", input_data.shape)
        print("label shape: ", label_data.shape)
        idx += 1


if __name__ == "__main__":
    simple_test_split()

    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    parser.add_argument("sample_idx", help="sample index", type=int)
    args = parser.parse_args()

    filepaths = get_dyst_filepaths(args.dyst_name)
    sample_filepath = filepaths.pop(args.sample_idx)
    print("sample filepath: ", sample_filepath)
    test_split_from_arrow(sample_filepath)