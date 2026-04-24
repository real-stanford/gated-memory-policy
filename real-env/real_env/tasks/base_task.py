import copy
import enum
from abc import ABC, abstractmethod
import json
import os
import threading
from typing import Any, Optional


from real_env.agents.base_agent import BaseAgent
from real_env.agents.policy_agent import PolicyAgent
from real_env.agents.spacemouse_agent import SpacemouseAgent
from real_env.controllers.base_controller_client import BaseControllerClient
from real_env.peripherals.base_camera import BaseCameraClient
from robologger.loggers.main_logger import MainLogger

import sys
import select
import termios
import tty

from robologger.classes import Morphology


class TaskControlMode(enum.IntEnum):
    SPACEMOUSE = 0
    POLICY = 1
    POLICY_SINGLE_STEP = 2


class EpisodeStatus(enum.Enum):
    """Status of episode execution during policy control."""

    RUNNING = "running"
    COMPLETED = "completed"


terminal_key_name: str | None = None
continue_running: bool = True


def keyboard_thread():
    # Use stdin to only record terminal input
    global continue_running
    global terminal_key_name
    while continue_running:
        terminal_key_name = read_char_with_timeout()
        if terminal_key_name:
            if (
                terminal_key_name == "q"
                or ord(terminal_key_name) == 3
                or ord(terminal_key_name) == 28
            ):
                print("Quitting input thread...")
                continue_running = False


def read_char_with_timeout(timeout=100):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Put terminal into raw mode to read single character
        tty.setcbreak(fd)
        # Wait for input with timeout
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        else:
            return None  # Timeout, no input
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


global_key_name: str | None = None


def on_press(key: "keyboard.Key | keyboard.KeyCode | None"):
    global global_key_name
    if key is not None:
        if hasattr(key, "char"):
            global_key_name = key.char
        else:
            global_key_name = None
    else:
        global_key_name = None


class BaseTask(ABC):
    def __init__(
        self,
        root_dir: str,
        project_name: str,
        task_name: str,
        run_name: str,
        logger_endpoints: dict[str, str],
        spacemouse_agent: SpacemouseAgent,
        config_str: str = "",
    ):
        self.task_config: dict[str, Any] = json.loads(config_str)
        self.spacemouse_agent: SpacemouseAgent = spacemouse_agent
        os.makedirs(root_dir, exist_ok=True)

        t = threading.Thread(target=keyboard_thread, daemon=True)
        t.start()
        self.control_mode: TaskControlMode = TaskControlMode.SPACEMOUSE

        self.main_logger: MainLogger = MainLogger(
            name="main_logger",
            root_dir=root_dir,
            project_name=project_name,
            task_name=task_name,
            run_name=run_name,
            logger_endpoints=logger_endpoints,
            morphology=Morphology.SINGLE_ARM,
            success_config="input_false",
        )

        self.clients: list[
            BaseControllerClient | BaseCameraClient | PolicyAgent
        ] = []  # Should be registered in the subclass

    def run(self):
        global terminal_key_name
        global global_key_name
        from pynput import keyboard

        with keyboard.Listener(on_press=on_press) as listener:
            # All starting commands should be typed in the corresponding terminal, including:
            # - Start policy control: c
            # - Start policy control with single step: C
            # - Reset: r
            # - Reset policy agent: R
            # - Disconnect: q
            # - Delete last episode: x
            # - Export data: e

            # All stopping commands can be detected globally by the keyboard listener, including:
            # - Stop policy control: s
            # - Stop policy control successfully: y
            # - Stop policy control unsuccessfully: n

            while True:
                if terminal_key_name is not None:  # Timeout
                    last_terminal_key_name = terminal_key_name
                    print(f"terminal key pressed: {terminal_key_name}")
                    terminal_key_name = None
                    if ord(last_terminal_key_name) == 3:  # Ctrl+C
                        raise KeyboardInterrupt
                    elif ord(last_terminal_key_name) == 28:  # Ctrl+\
                        raise KeyboardInterrupt
                    elif last_terminal_key_name == "c":  # Continue
                        self.start_episode()
                        self.control_mode = TaskControlMode.POLICY
                    elif last_terminal_key_name == "C":
                        self.control_mode = TaskControlMode.POLICY_SINGLE_STEP
                    # elif last_terminal_key_name == ""
                    elif last_terminal_key_name == "r":
                        self.reset()
                    elif last_terminal_key_name == "R":
                        self.reset_policy_agent()
                    elif last_terminal_key_name == "q":
                        self.disconnect()
                        return
                    elif last_terminal_key_name == "d":
                        self.display_robot_state()
                    elif last_terminal_key_name == "x":
                        deleted_idx = self.delete_last_episode()
                        if deleted_idx is not None:
                            print(f"Deleted episode {deleted_idx}")
                        else:
                            print("Failed to delete last episode")
                    elif last_terminal_key_name == "e":
                        self.export_data()

                if global_key_name is not None:
                    last_global_key_name = global_key_name
                    global_key_name = None
                    if (
                        last_global_key_name == "s"
                        or last_global_key_name == "y"
                        or last_global_key_name == "n"
                    ):
                        if self.control_mode == TaskControlMode.POLICY:
                            print(f"global key pressed: {last_global_key_name}")
                            print("Stopping episode...")
                            is_successful = None
                            if last_global_key_name == "y":
                                is_successful = True
                            elif last_global_key_name == "n":
                                is_successful = False
                            self.run_spacemouse_control()
                            self.stop_episode(is_successful=is_successful)
                            self.control_mode = TaskControlMode.SPACEMOUSE

                if self.control_mode == TaskControlMode.POLICY:
                    status = self.run_policy_control()
                    if (
                        status == EpisodeStatus.COMPLETED
                    ):  # NOTE: iterative casting specific condition
                        print(
                            "Episode completed, stopping recording and returning to spacemouse control"
                        )
                        self.stop_episode()
                        self.control_mode = TaskControlMode.SPACEMOUSE
                elif self.control_mode == TaskControlMode.POLICY_SINGLE_STEP:
                    self.run_policy_control()
                    self.control_mode = TaskControlMode.SPACEMOUSE
                elif self.control_mode == TaskControlMode.SPACEMOUSE:
                    self.run_spacemouse_control()
                else:
                    raise ValueError(f"Invalid control mode: {self.control_mode}")

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def reset_policy_agent(self):
        ...

    @abstractmethod
    def run_policy_control(self) -> EpisodeStatus | None:
        """
        Execute one step of policy control.

        Returns:
            EpisodeStatus.COMPLETED if episode finished, EpisodeStatus.RUNNING if ongoing,
            or None for tasks that don't use status returns (legacy behavior).
        """
        ...

    @abstractmethod
    def run_spacemouse_control(self):
        ...

    def gather_configs(self):
        configs: dict[str, Any] = {}
        configs["task"] = copy.deepcopy(self.task_config)
        for client in self.clients:
            assert (
                client.name not in configs
            ), f"Client name {client.name} already in configs. Please ensure client names are unique."
            configs[client.name] = copy.deepcopy(client.get_config())

        return configs

    def start_episode(self):
        return self.main_logger.start_recording(episode_config=self.gather_configs())

    def stop_episode(self, is_successful: bool | None = None):
        return self.main_logger.stop_recording(is_successful=is_successful)

    def delete_last_episode(self):
        return self.main_logger.delete_last_episode()

    @abstractmethod
    def display_robot_state(self):
        ...

    @abstractmethod
    def disconnect(self):
        ...

    def batch_run(self, *args, **kwargs):
        """
        Optional method for running multiple episodes sequentially.
        Subclasses should override this with their own specific signature.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("batch_run not implemented for this task")

    def export_data(self):
        print("Please override export_data in the subclass")
