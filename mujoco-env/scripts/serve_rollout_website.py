import time

import click

from env.utils.visualize_utils import RolloutWebsiteServer


@click.command()
@click.argument("data_dir", type=str)
@click.option("--exclude-dirs", type=str, default="", help="Comma-separated list of directories to exclude from the website")
@click.option("--port", type=int, default=5000, help="Port to serve the website on")
def main(data_dir: str, exclude_dirs: str, port: int):
    RolloutWebsiteServer(data_dir, exclude_dirs.split(",")).start(port=port)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
