import numpy as np
import numpy.typing as npt

from env.modules.scenes.base_scene import BaseScene


class TableOnly(BaseScene):
    def __init__(self, table_center_xyz: list[float], **kwargs):
        super().__init__(**kwargs)
        self.table_center_xyz: npt.NDArray[np.float64] = np.array(table_center_xyz)

    def get_table_center_xyz(self) -> npt.NDArray[np.float64]:
        return self.table_center_xyz
