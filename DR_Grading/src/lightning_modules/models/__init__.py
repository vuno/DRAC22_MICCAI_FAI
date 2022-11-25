from .quality_model import QualityModel
from .grading_model import GradingModel

def get_drac_model(task, backbone, num_classes=3):
    if task == 'quality':
        return QualityModel(backbone, num_classes)
    
    elif task == 'grading':
        return GradingModel(backbone, num_classes)
    
    else:
        raise ValueError