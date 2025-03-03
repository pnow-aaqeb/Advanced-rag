from nest.core import Module
from .email_classification_controller import EmailClassificationController
from .email_classification_service import EmailClassificationService
from .bge_singleton_service import BGESingleton
from .pinecone_singleton_service import PineconeSingleton
from prisma import Prisma
from src.database.mongodb import MongoDB


@Module(
    controllers=[EmailClassificationController],
    providers=[EmailClassificationService,BGESingleton,PineconeSingleton,MongoDB],
    imports=[],
    exports=[MongoDB]
)   
class EmailClassificationModule:
    def __init__(self, mongodb: MongoDB):
        self.mongodb = mongodb