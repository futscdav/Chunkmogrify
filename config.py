#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import yaml

class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def has_set(self, attr):
        return attr in self and self[attr] is not None

def _into_deep_dotdict(regular_dict):
    new_dict = dotdict(regular_dict)
    for k, v in regular_dict.items():
        if type(v) == dict:
            new_dict[k] = _into_deep_dotdict(v)
    return new_dict

def _load_config(path):
    with open(path) as fs:
        loaded = yaml.safe_load(fs)
    return _into_deep_dotdict(loaded)

config = None
def global_config():
    global config
    if config is None:
        config = _load_config("_config.yaml")
    return config