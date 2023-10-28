from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test,single_gpu_cotta,Efficient_adaptation
from .train import get_root_logger, set_random_seed, train_segmentor,init_random_seed

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test','single_gpu_cotta',
    'show_result_pyplot','init_random_seed','single_gpu_language_cotta','single_gpu_language_cotta_xiao'
]
