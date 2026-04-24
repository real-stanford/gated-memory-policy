import multiprocessing as mp
import os
import subprocess

import click


def extract_data(dataset_name: str, data_dir: str, output_dir: str, post_script: str):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(f"{output_dir}/{dataset_name}{post_script}"):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    print(
        f"Decompressing {data_dir}/{dataset_name}{post_script}.tar.lz4 to {output_dir}/{dataset_name}{post_script}"
    )
    os.makedirs(f"{output_dir}/{dataset_name}{post_script}", exist_ok=True)
    subprocess.run(
        [
            f"lz4 -d -c {data_dir}/{dataset_name}{post_script}.tar.lz4 | tar xf - -C {output_dir}"
        ],
        cwd=output_dir,
        shell=True,
        check=True,
    )


def compress_data(dataset_name: str, data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(f"{output_dir}/{dataset_name}.zarr.tar.lz4"):
        print(f"Skipping {dataset_name} because it already exists in {output_dir}")
        return
    print(
        f"Compressing {data_dir}/{dataset_name}.zarr to {output_dir}/{dataset_name}.zarr.tar.lz4"
    )
    subprocess.run(
        [
            f"tar cf - {dataset_name}.zarr | lz4 -c > {output_dir}/{dataset_name}.zarr.tar.lz4"
        ],
        cwd=data_dir,
        shell=True,
        check=True,
    )


def clean_all_data(output_dir: str):
    subprocess.run(
        [f"rm -rf {output_dir}/*.zarr"], cwd=output_dir, shell=True, check=True
    )


@click.command()
@click.argument("datasets", type=str, required=True)
@click.option("--data_dir", type=str, required=True)
@click.option("--output_dir", type=str, required=True)
@click.option("--post_script", type=str, default=".zarr")
def main(data_dir: str, output_dir: str, datasets: str, post_script: str):
    datasets_list = datasets.split(",")
    num_processes = mp.cpu_count()
    with mp.Pool(num_processes) as pool:
        pool.starmap(
            extract_data,
            [
                (dataset_name, data_dir, output_dir, post_script)
                for dataset_name in datasets_list
            ],
        )


if __name__ == "__main__":
    main()
