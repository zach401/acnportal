from functools import wraps
from pydoc import locate

def json_reader(from_json):
    # TODO: Class kwargs for all from_jsons
    @wraps(from_json)
    def edit_loaded(inclass, in_json, loaded_dict={}, cls_kwargs={}):
        obj_id, context_dict = in_json
        try:
            obj_dict = context_dict[obj_id]
        except KeyError:
            # TODO: Raise error
            assert 1 == 0
        # TODO: Args different from nominal after
        # decorator, which is confusing.
        if obj_id is not None and obj_id in loaded_dict:
            return loaded_dict[obj_id]

        # TODO: this might be overl restrictive
        # assert obj_dict['class'] == \
        #     f'{inclass.__module__}.{inclass.__name__}'

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
        # TODO: Throw a missing object error.
        raise KeyError("missing object")
    obj_type = context_dict[obj_id]['class']
    # pydoc.locate safely imports a function given its . delimited
    # location.
    obj_class = locate(obj_type)
    obj = obj_class.from_json((obj_id, context_dict), loaded_dict)
    loaded_dict[obj_id] = obj
    return obj
