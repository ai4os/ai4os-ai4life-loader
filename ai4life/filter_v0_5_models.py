import os
from . import utils, config
utils.filter_and_load_models(os.path.join(config.MODELS_PATH, 'collection.json'))