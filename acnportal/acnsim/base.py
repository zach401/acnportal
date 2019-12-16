import json
import sys
from functools import wraps
from pydoc import locate

def read_from_id(obj_id, context_dict, loaded_dict={}):
    if obj_id in loaded_dict:
        return loaded_dict[obj_id]
    if obj_id not in context_dict:
        raise KeyError(f"Object with ID {obj_id} "
                        "not found in context_dict.")
    obj_type = context_dict[obj_id]['class']
    obj_class = locate(obj_type)

    obj = obj_class.from_registry(
        {'id': obj_id, 'context_dict': context_dict}, 
        loaded_dict=loaded_dict
    )

    loaded_dict[obj_id] = obj
    return obj

"""
Base class for all ACN-Sim objects. Includes functions for
representation and serialization of ACN-Sim objects.
"""
class BaseSimObj:
    def __repr__(self):
        """ General string representation of an ACN-Sim object. """
        attr_repr_lst = []
        for key, value in self.__dict__.items():
            attr_repr_lst.append(f'{key}={value}')
        attr_repr = ', '.join(attr_repr_lst)
        return (
            f'{self.__module__}.{self.__class__.__name__}'
            f'({attr_repr})'
        )

    @classmethod
    def from_json(cls, in_json):
        return cls.from_registry(json.loads(in_json))

    @classmethod
    def from_registry(cls, in_json, loaded_dict={}, cls_kwargs={}):
        obj_id, context_dict = in_json['id'], in_json['context_dict']
        try:
            obj_dict = context_dict[obj_id]
        except KeyError:
            raise KeyError(f"Object with ID {obj_id} "
                            "not found in context_dict.")

        if obj_id is not None and obj_id in loaded_dict:
            return loaded_dict[obj_id]

        try:
            assert obj_dict['class'] == \
                f'{cls.__module__}.{cls.__name__}'
        except AssertionError:
            # TODO: Better message here.
            warnings.warn("Deserializing subtype.")

        in_dict = obj_dict['args']

        out_obj = cls.from_dict(in_dict, context_dict, 
                                loaded_dict, cls_kwargs)

        if obj_id is not None:
            loaded_dict[obj_id] = out_obj
        return out_obj

    @classmethod
    def from_dict(cls, in_dict, context_dict, 
                  loaded_dict, cls_kwargs={}):
        raise NotImplementedError

    def to_json(self):
        return json.dumps(self.to_registry())

    def to_registry(self, context_dict={}):
        """ Returns a JSON serializable representation of self. """
        obj_id = f'{id(self)}'
        if obj_id in context_dict:
            return {'id': obj_id, 'context_dict': context_dict}

        obj_type = f'{self.__module__}.{self.__class__.__name__}'

        args_dict = self.to_dict(context_dict)

        obj_dict = {'class' : obj_type, 'args' : args_dict}
        context_dict[obj_id] = obj_dict
        
        return {'id': obj_id, 'context_dict': context_dict}

    def to_dict(self, context_dict={}):
        raise NotImplementedError