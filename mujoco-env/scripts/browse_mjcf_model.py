### This script is borrowed from robosuite
### https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/browse_mjcf_model.py
### copyright of robosuite is under MIT License

"""Visualize MJCF models.

Loads MJCF XML models from file and renders it on screen.

Example:
    $ python browse_mjcf_model.py --filepath ../models/assets/arenas/table_arena.xml
"""

import argparse

import mujoco
from mujoco import viewer
from pynput import keyboard

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 0.7],
    "distance": 1.5,
    "azimuth": 180,
    "elevation": -20,
}

continue_running = True


def on_press(key):
    global continue_running
    try:
        if key == keyboard.Key.esc:
            continue_running = False
    except AttributeError:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    args = parser.parse_args()

    print("Loading model from: {}".format(args.filepath))
    model = mujoco.MjModel.from_xml_path(args.filepath)
    data = mujoco.MjData(model)

    vwr = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    # vwr = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    vwr.cam.lookat = DEFAULT_FREE_CAM["lookat"]
    vwr.cam.distance = DEFAULT_FREE_CAM["distance"]
    vwr.cam.azimuth = DEFAULT_FREE_CAM["azimuth"]
    vwr.cam.elevation = DEFAULT_FREE_CAM["elevation"]

    vwr.sync()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'ESC' to exit.")

    while continue_running:
        pass

    vwr.close()
    listener.stop()
    listener.join()
