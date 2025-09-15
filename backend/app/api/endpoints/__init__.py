# Import all endpoint modules to make them available
from . import train, predict, annotate, auto_annotation, projects, models

__all__ = ['train', 'predict', 'annotate', 'auto_annotation', 'projects', 'models']