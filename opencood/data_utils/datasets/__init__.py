from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.kitti_intermediate_fusion_dataset import \
    KittiIntermediateFusionDataset

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'KittiIntermediateFusionDataset': KittiIntermediateFusionDataset
}

# the final range for evaluation
GT_RANGE = [-100, -40, -5, 100, 40, 3]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True, isSim=False):
    fusion_cfg = dataset_cfg['fusion']
    if isSim and 'source_core_method' in fusion_cfg:
        dataset_name = fusion_cfg['source_core_method']
    elif (not isSim) and 'target_core_method' in fusion_cfg:
        dataset_name = fusion_cfg['target_core_method']
    else:
        dataset_name = fusion_cfg['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in __all__, error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        isSim=isSim
    )

    return dataset
