from typing import Any, cast

import numpy as np
import numpy.typing as npt

robot_data_type = dict[str, Any]

data_buffer_type = list[list[robot_data_type]]
f64arr = npt.NDArray[np.float64]


def castf64(arr: Any) -> f64arr:
    """Cast an array to f64arr. This is only used for type hinting. No actual casting is done during runtime."""
    return cast(f64arr, arr)
