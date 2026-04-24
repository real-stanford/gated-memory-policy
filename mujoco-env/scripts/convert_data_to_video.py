import os
import sys

import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.utils.visualize_utils import (
    convert_data_to_video,
)


@click.command()
@click.argument("data_dir", type=str)
@click.option(
    "--split_videos",
    is_flag=True,
    help="Split videos for each camera",
    default=True,
)
@click.option(
    "--freq_hz",
    type=int,
    help="Frequency of the video",
    default=10,
)
def main(data_dir: str, split_videos: bool, freq_hz: int):

    convert_data_to_video(data_dir, split_videos, "all", freq_hz, add_black_screen_in_the_end=False)

    # try:
    #     convert_data_to_video_parallel(
    #         data_dir, split_videos, "failure", freq_hz
    #     )
    #     convert_data_to_video_parallel(
    #         data_dir, split_videos, "success", freq_hz
    #     )
    # except KeyError:
    #     convert_data_to_video_parallel(
    #         data_dir, split_videos, "all", freq_hz
    #     )


if __name__ == "__main__":
    main()
