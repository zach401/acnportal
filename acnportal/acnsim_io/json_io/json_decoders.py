from functools import wraps
from pydoc import locate
import warnings

def json_reader(from_json):
    @wraps(from_json)
    def edit_loaded(inclass, in_json, context_dict={}, loaded_dict={}, cls_kwargs={}):
        obj_id, context_dict = in_json['id'], in_json['context_dict']
        try:
            obj_dict = context_dict[obj_id]
        except KeyError:
            raise KeyError(f"Object with ID {obj_id} not found in context_dict.")

        if obj_id is not None and obj_id in loaded_dict:
            return loaded_dict[obj_id]

        try:
            assert obj_dict['class'] == \
                f'{inclass.__module__}.{inclass.__name__}'
        except AssertionError:
            # TODO: Better message here.
            warnings.warn("Deserializing subtype.")

        in_dict = obj_dict['args']

        out_obj = from_json(inclass, in_dict, context_dict, loaded_dict, cls_kwargs)

        if obj_id is not None:
            loaded_dict[obj_id] = out_obj
        return out_obj
    return edit_loaded

def read_from_id(obj_id, context_dict, loaded_dict={}):
    if obj_id in loaded_dict:
        return loaded_dict[obj_id]
    if obj_id not in context_dict:
        raise KeyError(f"Object with ID {obj_id} not found in context_dict.")
    obj_type = context_dict[obj_id]['class']
    obj_class = locate(obj_type)
    obj = obj_class.from_json({'id': obj_id, 'context_dict': context_dict}, loaded_dict=loaded_dict)
    loaded_dict[obj_id] = obj
    return obj
