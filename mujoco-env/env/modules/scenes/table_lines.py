import numpy as np
import numpy.typing as npt

from env.modules.scenes.base_scene import BaseScene


class TableLines(BaseScene):
    """
    ---> y
    |
    v      -----------------------------------------------------
    x      |               |                                   |
           |               |                   -----           |
    robot  |               L         T         | B |           |
           |               |                   -----           |
           |               |                                   |
           -----------------------------------------------------
    B: box_center_xyz
    L: line_center_xyz
    T: table_center_xyz
    """

    def __init__(
        self,
        table_center_xyz: list[float],
        table_width: float,
        table_length: float,
        line_center_xyz: list[float],
        box_center_xyz: list[float],
        box_width: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.box_center_xyz: npt.NDArray[np.float64] = np.array(box_center_xyz)
        self.line_center_xyz: npt.NDArray[np.float64] = np.array(line_center_xyz)
        self.table_center_xyz: npt.NDArray[np.float64] = np.array(table_center_xyz)
        self.table_width: float = table_width
        self.table_length: float = table_length
        self.box_width: float = box_width
