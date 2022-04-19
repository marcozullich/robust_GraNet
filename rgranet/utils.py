import os
import yaml
from typing import Collection
from functools import singledispatch
from types import SimpleNamespace
from collections.abc import Mapping


def coalesce(element, value_if_none):
    return value_if_none if element is None else element


def make_subdirectory(path):
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def get_file_content(file_path, error_message=None):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        if error_message is not None:
            raise FileNotFoundError(error_message)
        raise FileNotFoundError(f"Could not find file {file_path}")

def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

# class NameSpace(SimpleNamespace, Mapping):
#     def __init__(self, **kwargs):
#         super(SimpleNamespace, self).__init__(**kwargs)
    
#     def __repr__(self):
#         super(SimpleNamespace, self).__repr__()
    
#     def __eq__(self, other):
#         super(SimpleNamespace, self).__eq__(other)
    
#     def __getitem__(self, key):
#         return self.__dict__[key]
    
#     def __iter__(self):
#         return self.__dict__
    
#     def __len__(self):
#         return len(self.__dict__)

@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]

class DictOfLists(dict):
    def __init__(self, keys:Collection=None):
        if keys is None:
            keys = []
        d = {c:[] for c in keys}
        super().__init__(d)

    def get_item(self, key, index=-1):
        val = self.get(key)
        if val is not None:
            if len(val) > 0:
                return val[index]
            return None
        return None
    
    def len_key(self, key):
        return len(self.get(key, []))
    
    def add_keys(self, *keys):
        for k in keys:
            if self.get(k) is None:
                self[k] = []

