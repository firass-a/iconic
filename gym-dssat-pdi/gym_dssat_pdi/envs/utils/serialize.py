"""
From https://stackoverflow.com/questions/65354137/how-to-convert-nested-dictionary-with-numpy-array-to-json-and-back
"""
from numpy import array

# def convert(x):
#     if hasattr(x, "tolist"):  # numpy arrays have this
#         return x.tolist()
#     raise TypeError(x)

def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        x = x.tolist()
    return x

def deconvert(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return array(value)  # cast back to array
    return x