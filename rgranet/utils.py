
from typing import Collection


def coalesce(element, value_if_none):
    return value_if_none if element is None else element

def get_file_content(file_path, error_message=None):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        if error_message is not None:
            raise FileNotFoundError(error_message)
        raise FileNotFoundError(f"Could not find file {file_path}")

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