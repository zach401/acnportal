from functools import wraps

def json_writer(to_json):
    @wraps(to_json)
    def edit_context(obj, context_dict={}):
        obj_id = f'{id(obj)}'
        if obj_id in context_dict:
            return obj_id, context_dict

        obj_type = f'{obj.__module__}.{obj.__class__.__name__}'

        args_dict = to_json(obj, context_dict)

        obj_dict = {'class' : obj_type, 'args' : args_dict}
        context_dict[obj_id] = obj_dict
        # TODO: Nominal return type different from return type
        # after decorator, which is confusing.
        # TODO: Maybe should be a dict return type? Need to change
        # unpacking then.
        return obj_id, context_dict
    return edit_context