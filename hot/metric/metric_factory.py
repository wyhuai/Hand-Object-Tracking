from .grasp_metric import GraspMetric
from .place_metric import PlaceMetric
from .move_metric import MoveMetric
from .catch_metric import CatchMetric
from .rotate_metric import RotateMetric
from .regrasp_metric import RegraspMetric
from .throw_metric import ThrowMetric
from .long_metric import LongMetric
from .grab_metric import GrabMetric
from .tracking_metric import TrackingMetric

# hot/metrics/metric_factory.py
def create_metric(skill_name, num_envs, device, max_episode_length, **kwargs):
    h_offset = 0 if 'higher' not in skill_name else 0.1
    skill_name = skill_name.split('_')[0]
    metric_classes = {
        'grasp': GraspMetric,
        'place': PlaceMetric,
        'move': TrackingMetric,
        'catch': CatchMetric,
        'throw': ThrowMetric,
        'rotate': RotateMetric,
        'regrasp': RegraspMetric,
        'bottle': LongMetric,
        '001': GrabMetric,
        '000': GrabMetric,
    }
    MetricClass = metric_classes.get(skill_name)
    if MetricClass:
        return MetricClass(num_envs, device, h_offset, max_episode_length)
    else:
        return None  # 或者返回一个默认的 Metric
