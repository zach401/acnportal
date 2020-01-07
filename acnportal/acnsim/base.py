import json
import sys
from functools import wraps
from pydoc import locate
import warnings


def read_from_id(obj_id, context_dict, loaded_dict=None):
    if loaded_dict is None:
        loaded_dict = {}
    if obj_id in loaded_dict:
        return loaded_dict[obj_id]
    if obj_id not in context_dict:
        raise KeyError(f"Object with ID {obj_id} not found in context_dict.")
    obj_type = context_dict[obj_id]['class']
    obj_class = locate(obj_type)

    obj = obj_class.from_registry(
        {'id': obj_id, 'context_dict': context_dict}, loaded_dict=loaded_dict)

    loaded_dict[obj_id] = obj
    return obj

def none_to_empty_dict(*args):
    """ Returns a new args list that replaces each None arg with {}. """
    out_arg_lst = []
    for arg in args:
        if arg is None:
            out_arg_lst.append({})
        else:
            out_arg_lst.append(arg)
    return out_arg_lst

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
        return f'{self.__module__}.{self.__class__.__name__}({attr_repr})'

    def to_json(self):
        return json.dumps(self.to_registry())

    def to_registry(self, context_dict=None):
        """ Returns a JSON serializable representation of self. """
        context_dict, = none_to_empty_dict(context_dict)
        obj_id = f'{id(self)}'
        if obj_id in context_dict:
            return {'id': obj_id, 'context_dict': context_dict}

        obj_type = f'{self.__module__}.{self.__class__.__name__}'

        args_dict = self.to_dict(context_dict)

        # Check that all attributes have been serialized
        # Warn if some attributes weren't serialized
        if args_dict.keys() != self.__dict__.keys():
            unserialized_keys = \
                set(self.__dict__.keys()) - set(args_dict.keys())
            warnings.warn(
                f"Attributes {unserialized_keys} present in object of type "
                f"{obj_type} but not handled by object's to_dict method. "
                 "Serialized object may not load correctly. Write a to_dict "
                 "method and re-dump, or write an appropriate from_dict "
                 "method to accurately load.",
                UserWarning
            )
            for key in unserialized_keys:
                unserialized_attr = self.__dict__[key]
                # Try calling the attr's to_registry method if it has one.
                try:
                    args_dict[key] = unserialized_attr.to_registry(
                        context_dict=context_dict)['id']
                    continue
                except AttributeError:
                    pass
                # Try dumping the object using the native JSON serializer.
                try:
                    json.dumps(unserialized_attr)
                except TypeError:
                    warnings.warn(
                        f"Attribute {key} could not be serialized. Dumping "
                         "the attribute's repr() representation. This "
                         "attribute will not be fully loaded.",
                        UserWarning
                    )
                    args_dict[key] = repr(unserialized_attr)
                else:
                    args_dict[key] = unserialized_attr

        obj_dict = {'class': obj_type, 'args': args_dict}
        context_dict[obj_id] = obj_dict

        return {'id': obj_id, 'context_dict': context_dict}

    def to_dict(self, context_dict=None):
        raise NotImplementedError

    @classmethod
    def from_json(cls, in_json):
        return cls.from_registry(json.loads(in_json))

    @classmethod
    def from_registry(cls, in_json, loaded_dict=None, cls_kwargs=None):
        loaded_dict, cls_kwargs = none_to_empty_dict(loaded_dict, cls_kwargs)
        obj_id, context_dict = in_json['id'], in_json['context_dict']
        try:
            obj_dict = context_dict[obj_id]
        except KeyError:
            raise KeyError(
                f"Object with ID {obj_id} not found in context_dict.")

        if obj_id is not None and obj_id in loaded_dict:
            return loaded_dict[obj_id]

        try:
            assert obj_dict['class'] == \
                f'{cls.__module__}.{cls.__name__}'
        except AssertionError:
            # TODO: Better message here.
            warnings.warn(
                f"Deserializing as type {cls.__module__}.{cls.__name__}. "
                f"Object was serialized as type {obj_dict['class']}.",
                UserWarning
            )

        in_dict = obj_dict['args']

        out_obj = cls.from_dict(in_dict, context_dict, loaded_dict, cls_kwargs)
        if out_obj.__dict__.keys() != in_dict.keys():
            unloaded_attrs = set(in_dict.keys()) - set(out_obj.__dict__.keys())
            warnings.warn(
                f"Attributes {unloaded_attrs} present in object of type "
                f"{obj_dict['class']} but not handled by object's from_dict "
                 "method. Loaded object may have inaccurate attributes.",
                UserWarning
            )
            for attr in unloaded_attrs:
                try:
                    setattr(
                        out_obj, attr,
                        read_from_id(in_dict[attr], context_dict, loaded_dict)
                    )
                except (KeyError, TypeError):
                    warnings.warn(
                        f"Loader for attribute {attr} not found. Setting "
                        f"attribute {attr} directly.",
                        UserWarning
                    )
                    setattr(out_obj, attr, in_dict[attr])

        if obj_id is not None:
            loaded_dict[obj_id] = out_obj
        return out_obj

    @classmethod
    def from_dict(cls, in_dict, context_dict, loaded_dict, cls_kwargs=None):
        raise NotImplementedError
