import json
import jsonpickle


def save_json(obj, fname='test.json'):

    obj = jsonpickle.encode(obj)
    
    with open(fname,'w') as f:
        json.dump(obj, f)

def load_json(fname):

    with open(fname, 'r') as f:
        obj = json.load(f)
        obj = jsonpickle.decode(obj)

        return obj        