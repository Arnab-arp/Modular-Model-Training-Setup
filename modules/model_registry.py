import os
from importlib import import_module


class ModelRegistry:
    """
    This is a Class Method that registers models in the model_directory
    via a class decorator. So that in future, whatever models we make in the 'models' directory
    before instantiating the 
    `class Model(nn.Module):`
    we add the @ @Registry.register_model

    Example:
    ```
    from torch import nn
    from modules.model_registry import Registry

    @Registry.register_model
    class Model(nn.Module):
        ...
    ```
    """
    def __init__(self):
        self._registry = {}

    def register_model(self, cls):
        self._registry[cls.__name__] = cls
        return self._registry

    def registered_models(self):
        if not self._registry:
            print("No Models Registered")
            return []
        return self._registry

    def discover_models(self, models_dir="modules.models"):
        sub_path = models_dir.replace('.', '\\')
        abs_path = os.path.join(os.getcwd(), sub_path)

        if not os.path.exists(abs_path):
            print(f'No Directory Found on Path : {abs_path}')
            return

        skipped_files = ('__pycache__', '__init__.py')
        for filename in os.listdir(abs_path):
            if filename.endswith(".py") and (filename not in skipped_files):
                module_name = filename[:-3]
                import_module(f"{models_dir}.{module_name}")
                

Registry = ModelRegistry()