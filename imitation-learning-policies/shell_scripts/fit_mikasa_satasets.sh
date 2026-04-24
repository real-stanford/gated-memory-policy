#!/bin/bash

mikasa_tasks=(
    # InterceptGrabMedium-v0
    InterceptMedium-v0
    RememberColor3-v0
    RememberColor5-v0
    RememberColor9-v0
    ShellGameTouch-v0

    # RememberShape3-v0
    # RememberShape5-v0
    # RememberShapeAndColor3x3-v0
    # RotateLenientPos-v0
    # RotateStrictPos-v0
    # TakeItBack-v0
)


for task in ${mikasa_tasks[@]}; do
    python scripts/fit_dataset_normalizer.py /storage/hdd1/jeff/mikasa-zarr/$task mikasa_remember_color_3 --quantile 1.0
done
