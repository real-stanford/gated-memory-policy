from imitation_learning.datasets.normalizer import get_multi_task_normalizer

def fit_multi_task_normalizer(normalizer_json_paths: list[str], output_normalizer_json_path: str):
    get_multi_task_normalizer(normalizer_json_paths, output_normalizer_json_path)

if __name__ == "__main__":
    normalizer_json_paths = [
        "data/iphumi_data/pick-and-place-back-correction-all_normalizer.json",
        "data/iphumi_data/pick-and-place-back-all_normalizer.json",
        "data/iphumi_data/pick-and-place-back-and-correction_normalizer.json",
    ]
    output_normalizer_json_path = 'data/iphumi_data/place-back-mixed-correction-all_normalizer.json'
    fit_multi_task_normalizer(normalizer_json_paths, output_normalizer_json_path)