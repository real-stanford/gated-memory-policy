from typing import TYPE_CHECKING


from env.modules.objects.base_rigid_object import BaseRigidObject

if TYPE_CHECKING:
    pass

from loguru import logger


class Button(BaseRigidObject):
    def __init__(self, press_height_threshold: float, **kwargs):
        logger.info(kwargs)
        super().__init__(**kwargs)
        self.press_height_threshold: float = press_height_threshold

    @property
    def is_pressed(self) -> bool:
        # TODO: Implement this
        return False
