"""Shared validation of Sobol sequence blocks, including Saltelli base blocks."""

import warnings

import numpy as np


def validateSobolSetup(count, skipValue):
    if isinstance(skipValue, (bool, np.bool_)) or not isinstance(skipValue, (int, np.integer)):
        raise TypeError("skipValue must be an integer.")
    if skipValue < 0:
        raise ValueError("skipValue must be greater than or equal to 0.")
    if count & (count - 1):
        warnings.warn(
            f"Sobol base size {count} is not a power of 2; balance is not guaranteed. "
            f"Consider using {1 << count.bit_length()} points and skipValue=0.",
            UserWarning,
            stacklevel=3,
        )
    elif skipValue % count:
        warnings.warn(
            f"Sobol skipValue={skipValue} is not aligned to a block of {count} points; "
            "balance is not guaranteed. Use skipValue=0 or a multiple of the base size.",
            UserWarning,
            stacklevel=3,
        )
