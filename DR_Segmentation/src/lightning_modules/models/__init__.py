from .quality_model import QualityModel
from .grading_model import GradingModel
from .segment_model import SegmentModel


def get_drac_model(task, backbone, num_classes=3, dropout=0.1, stochastic_depth=0.2):
    if task == 'quality':
        return QualityModel(backbone, num_classes, dropout, stochastic_depth)
    
    elif task == 'grading':
        return GradingModel(backbone, num_classes)
    
    elif task == 'segment':
        return SegmentModel(backbone, num_classes)
    
    else:
        raise ValueError