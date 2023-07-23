from nersemble.model_manager.base import NeRSembleBaseModelManager, NeRSembleBaseModelFolder


class NeRSembleModelManager(NeRSembleBaseModelManager):

    def __init__(self, run_name: str):
        super().__init__(f"nersemble", run_name)


class NeRSembleModelFolder(NeRSembleBaseModelFolder[NeRSembleModelManager]):

    def __init__(self):
        super().__init__("nersemble", 'NERS')
