import numpy as np


def zero_to(input_list,replace = 1):
    modified_list = []
    for x in input_list:
        if x == 0:
            modified_list.append(replace)
        else:
            modified_list.append(x)
    return modified_list

def calculate_mape(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Both lists must have the same length")

    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    percentage_errors = [(error / a) * 100 if a != 0 else 0 for a, error in zip(actual, absolute_errors)]

    mape = sum(percentage_errors) / len(actual)
    return mape

def remove_zeros(l) -> list:
    for i in range(len(l)):
        if l[i] == 0:
            l[i] = 1
    return l

def mape_(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - actual: List or array of actual values
    - predicted: List or array of predicted values

    Returns:
    - mape: Mean Absolute Percentage Error
    """
    actual, predicted = map(lambda x: [0.0001 if i == 0 else i for i in x], [actual, predicted])

    # Convert lists to NumPy arrays
    actual, predicted = np.array(actual), np.array(predicted)

    absolute_percentage_errors = np.abs((actual - predicted) / actual)
    mape = (100 / len(actual)) * np.sum(absolute_percentage_errors)

    return mape
