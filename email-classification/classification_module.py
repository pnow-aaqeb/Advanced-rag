from nest.core import PyNestFactory, Module
        
from .classification_controller import ClassificationController
from .classification_service import ManualEmailClassifier


@Module(imports=[], controllers=[ClassificationController], providers=[ManualEmailClassifier])
class ClassificationModule:
    pass

