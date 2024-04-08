from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import single_model_update,single_gpu_cotta,Efficient_adaptation,DPT,single_gpu_ours,single_gpu_AuxAdapt,single_gpu_RDumb
from .test_backup import multi_gpu_test
from .train import get_root_logger, set_random_seed, train_segmentor,init_random_seed

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor',  'single_model_update','single_gpu_cotta',
    'show_result_pyplot','init_random_seed','single_gpu_ours','single_gpu_AuxAdapt','multi_gpu_test','single_gpu_RDumb'
]
