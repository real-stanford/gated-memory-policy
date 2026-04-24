from imitation_learning.datasets.aggregated_dataset import AggregatedDataset
from imitation_learning.datasets.iphumi_dataset import _process_source_data

class UMISingleTrajDataset(AggregatedDataset):
    pass

UMISingleTrajDataset._process_source_data = _process_source_data