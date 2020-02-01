"""
This module contains definitions shared by all ACN-Sim objects.
"""
import json
from pydoc import locate
import warnings
import pkg_resources
import pandas
import numpy


def none_to_empty_dict(*args):
    """ Returns a new args list that replaces each None arg with {}. """
    out_arg_lst = []
    for arg in args:
        if arg is None:
            out_arg_lst.append({})
        else:
            out_arg_lst.append(arg)
    return out_arg_lst


def read_from_id(obj_id, context_dict, loaded_dict=None):
    """
    Given an object ID and a dictionary mapping object ID's to JSON
    serializable representations of ACN-Sim objects, returns the ACN-Sim
    object represented by the given object_id.

    Optionally, a loaded_dict that contains already-loaded objects
    may be provided to avoid duplicated work.

    Args:
        obj_id (str): Object ID of ACN-Sim object to be loaded.
        context_dict (Dict[str, JSON Serializable]): Dict mapping object
            ID's to object JSON serializable representations.
        loaded_dict (Dict[str, BaseSimObj-like]): Dict mapping object
            ID's to ACN-Sim objects.

    Returns:
        BaseSimObj-like: The loaded ACN-Sim object.

    Raises:
        KeyError: Raised if `obj_id` is not found in `context_dict`.
    """
    loaded_dict, = none_to_empty_dict(loaded_dict)

    # Check if this object has already been loaded; return the loaded
    # object if this is the case.
    if obj_id in loaded_dict:
        return loaded_dict[obj_id]

    if obj_id not in context_dict:
        raise KeyError(
            f"Object with ID {obj_id} not found in context_dict."
        )

    # Get the class of this object from the context_dict.
    obj_type = context_dict[obj_id]['class']
    obj_class = locate(obj_type)

    # 'version' is None since we've already checked the version of the
    # parent object.
    obj = obj_class.from_registry(
        {'id': obj_id,
         'context_dict': context_dict,
         'version': None,
         'dependency_versions': None},
        loaded_dict=loaded_dict
    )

    loaded_dict[obj_id] = obj
    return obj


class BaseSimObj:
    """
    Base class for all ACN-Sim objects. Includes functions for
    representation and serialization of ACN-Sim objects.
    """
    def __repr__(self):
        """
        General string representation of an ACN-Sim object.

        Returns:
            str: A representation of the object in the following form:
                [module name].[class name](attr1=[value1],
                                           attr2=[value2],
                                           ...)
        """
        attr_repr_lst = []
        for key, value in self.__dict__.items():
            attr_repr_lst.append(f'{key}={value}')
        attr_repr = ', '.join(attr_repr_lst)
        return f'{self.__module__}.{self.__class__.__name__}({attr_repr})'

    def to_json(self):
        """ Returns a JSON string representing self. """
        return json.dumps(self.to_registry())

    def to_registry(self, context_dict=None):
        """
        Returns a JSON serializable representation of self.

        The serializer is invoked using `obj.to_json()`, but the
        conversion into a serializable representation occurs in this
        method.

        Serializer behaviors:

        - Any native ACN-Sim object (that is, any unaltered object
        provided in `acnsim`) dumps without error and with
        accurate preservation of attributes.

        - An extension of an ACN-Sim object that appropriately defines a
        `to_dict` method will dump without error and with
        accurate attribute preservation. This also applies to ACN-Sim
        objects that have as attributes extensions of ACN-Sim objects
        with defined `to_dict` methods.

        - An extension that doesn't define `to_dict` will dump without
        error, but with partial preservation of attributes. Objects that
        are not JSON serializable and which do not have a `to_registry`
        method will not be completely serialized unless said objects are
        extensions of built-in ACN-Sim objects with no extra attributes.

        - A warning is thrown for each attribute not explicitly handled
        in the serialization.

        - A warning is thrown for each attribute that is neither naively
        serializable nor handled by any object's `to_registry` method.

        The serializer returns a dict with two keys. An `id`, and a
        `context_dict` that maps ID's to JSON serializable
        representations of  objects. This ensures that objects appearing
        multiple times in another object's attributes are not re-loaded
        as different objects. For example, a Simulator object may have
        an EV appearing in both the `ev_history`, and the
        `event_history`, as part of a `PluginEvent`. The `id`-based
        serialization ensures that on loading, the same `EV` object
        occupies both the `ev_history` and the `event_history`.

        As serialization of nested objects occurs recursively, this
        method also optionally accepts a `context_dict` that contains
        objects (via the ID's) that have already been converted into a
        serializable form, to which the serializable form of this object
        is added.

        Args:
            context_dict (Dict[str, JSON Serializable]): Dict mapping
                object ID's to object JSON serializable representations.

        Returns:
            Dict[str, JSON Serializable]: A JSON serializable
                representation of this object. Takes the form:
                    ```
                    {'id': [This object's id],
                     'context_dict': [The dict mapping all object id's
                                      to their JSON serializable
                                      representations],
                     'version': [commit hash of acnportal at creation
                                 of this object]}
                    ```

        Warns:
            UserWarning: If any attributes are present in the object
                not handled by the object's `to_dict` method.
            UserWarning: If any of the unhandled attributes are not
                JSON serializable.

        """
        context_dict, = none_to_empty_dict(context_dict)
        obj_id = f'{id(self)}'

        # Check if this object has already been converted, and return
        # the appropriate dict if this is the case.
        if obj_id in context_dict:
            return {'id': obj_id, 'context_dict': context_dict}

        obj_type = f'{self.__module__}.{self.__class__.__name__}'

        # Get a dictionary of the attributes of this object using the
        # object's to_dict method.
        args_dict = self.to_dict(context_dict)

        # Check that all attributes have been serialized.
        # Warn if some attributes weren't serialized.
        if args_dict.keys() != self.__dict__.keys():
            unserialized_keys = \
                set(self.__dict__.keys()) - set(args_dict.keys())
            warnings.warn(
                f"Attributes {unserialized_keys} present in object of type "
                f"{obj_type} but not handled by object's to_dict method. "
                f"Serialized object may not load correctly. Write a to_dict "
                f"method and re-dump, or write an appropriate from_dict "
                f"method to accurately load.",
                UserWarning
            )
            for key in unserialized_keys:
                unserialized_attr = self.__dict__[key]
                # Try calling the attr's to_registry method.
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
                        f"the attribute's repr() representation. This "
                        f"attribute will not be fully loaded.",
                        UserWarning
                    )
                    args_dict[key] = repr(unserialized_attr)
                else:
                    args_dict[key] = unserialized_attr

        obj_dict = {'class': obj_type, 'args': args_dict}
        context_dict[obj_id] = obj_dict

        # Check versions of acnportal and certain dependencies.
        acnportal_version = pkg_resources.require('acnportal')[0].version
        dependency_versions = {
            'numpy': numpy.__version__,
            'pandas': pandas.__version__
        }

        return {
            'id': obj_id,
            'context_dict': context_dict,
            'version': acnportal_version,
            'dependency_versions': dependency_versions
        }

    def to_dict(self, context_dict=None):
        """ Converts the object's attributes into a JSON serializable
        dict. Each ACN-Sim object defines this method differently.

        Args:
            context_dict (Dict[str, JSON Serializable]): Dict mapping
                object ID's to object JSON serializable representations.

        Returns:
            Dict[str, JSON Serializable]: Dict of the object's
                attributes, converted to JSON serializable forms.
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, in_json):
        """ Returns an ACN-Sim object loaded from in_json. """
        return cls.from_registry(json.loads(in_json))

    @classmethod
    def from_registry(cls, in_json, loaded_dict=None, cls_kwargs=None):
        """
        Returns an object of type `cls` from a JSON serializable
        representation of the object.

        The deserializer is invoked using `cls.from_json()`, but the
        conversion from a serializable representation to an ACN-Sim
        object occurs in this method.

        Deserializer behaviors:

        - Any native ACN-Sim object (that is, any unaltered object
        provided in `acnsim`) that was dumped (serialized) with the
        object's `to_registry` method loads without error and with
        accurate values for each attribute, assuming the original object
        was of type `cls`.

        - An extension of an ACN-Sim object that appropriately defines a
        `from_dict` method will load without error and with
        accurate attribute values. This also applies to ACN-Sim
        objects that have as attributes extensions of ACN-Sim objects
        with defined `from_dict` methods.

        - An extension that doesn't define `from_dict` may not load
        correctly. The extension will load correctly only if the
        constructor of the object whose `from_dict` method is called
        takes the same arguments as this object, and any extra
        attributes are JSON-serializable or readable from an ID.

        - A warning is thrown for each attribute not explicitly loaded
        in the deserialization.

        - A warning is thrown for each attribute that is neither naively
        serializable nor readable from an ID or with a `from_registry`
        method.

        - A warning is thrown if any of acnportal, numpy, or pandas have
        different versions in the current program from the versions
        present when the object was serialized.

        The deserializer returns an object of type `cls`.

        As deserialization of nested objects occurs recursively, this
        method also optionally accepts a `loaded_dict` that contains
        objects (via the ID's) that have already been loaded, to which
        the deserialized object is added.

        Args:
            in_json (Dict[str, JSON Serializable]): A JSON Serializable
                representation of this object. Takes the form:
                    ```
                    {'id': [This object's id],
                     'context_dict': [The dict mapping all object id's
                                      to their JSON serializable
                                      representations]}
                    ```
            loaded_dict (Dict[str, BaseSimObj-like]): Dict mapping
                object ID's to loaded ACN-Sim objects.
            cls_kwargs (Dict[str, object]): Optional extra arguments
                to be passed to object's constructor.

        Returns:
            BaseSimObj-like: Loaded object.

        Raises:
            KeyError: Raised if object represented by `in_json` is not
                found in `context_dict`.

        Warns:
            UserWarning: If the acnportal version of the loaded object
                is different from that of the acnportal doing
                the loading, or if numpy or pandas is a different
                version. If no version is provided ('version' maps
                to None), no warning is raised.
            UserWarning: If any attributes are present in the object
                not handled by the object's `from_dict` method.
            UserWarning: If any of the unhandled attributes are not
                readable from an ID. This is fine unless this attribute
                is not meant to be JSON serializable, in which case
                the loaded attribute will be incorrect.

        """
        loaded_dict, cls_kwargs = none_to_empty_dict(loaded_dict, cls_kwargs)
        obj_id, context_dict, acnportal_version, dependency_versions = (
            in_json['id'],
            in_json['context_dict'],
            in_json['version'],
            in_json['dependency_versions']
        )

        # Check current versions of acnportal and certain dependencies
        # against serialized versions.
        current_version = pkg_resources.require('acnportal')[0].version
        if (acnportal_version is not None
                and current_version != acnportal_version):
            warnings.warn(
                f"Version {acnportal_version} of input acnportal object does "
                f"not match current version {current_version}."
            )

        current_dependency_versions = {
            'numpy': numpy.__version__,
            'pandas': pandas.__version__
        }
        if dependency_versions is not None:
            for pkg in dependency_versions.keys():
                if (current_dependency_versions[pkg]
                        != dependency_versions[pkg]):
                    warnings.warn(
                        f"Current version of dependency {pkg} does not match "
                        f"serialized version. "
                        f"Current: {current_dependency_versions[pkg]}, "
                        f"Serialized: {dependency_versions[pkg]}.",
                    )

        try:
            obj_dict = context_dict[obj_id]
        except KeyError:
            raise KeyError(
                f"Object with ID {obj_id} not found in context_dict.")

        # Check if this object has already been converted, and return
        # the appropriate dict if this is the case.
        if obj_id is not None and obj_id in loaded_dict:
            return loaded_dict[obj_id]

        if obj_dict['class'] != f'{cls.__module__}.{cls.__name__}':
            warnings.warn(
                f"Deserializing as type {cls.__module__}.{cls.__name__}. "
                f"Object was serialized as type {obj_dict['class']}.",
                UserWarning
            )

        # in_dict is a dict mapping attribute names to JSON serializable
        # values.
        in_dict = obj_dict['args']

        # Call this class' from_dict method to convert the JSON
        # representation of this object's attributes into the actual
        # object.
        out_obj = cls.from_dict(in_dict, context_dict, loaded_dict, cls_kwargs)

        # Check that all attributes have been loaded.
        # Warn if some attributes weren't loaded.
        if out_obj.__dict__.keys() != in_dict.keys():
            unloaded_attrs = set(in_dict.keys()) - set(out_obj.__dict__.keys())
            warnings.warn(
                f"Attributes {unloaded_attrs} present in object of type "
                f"{obj_dict['class']} but not handled by object's from_dict "
                f"method. Loaded object may have inaccurate attributes.",
                UserWarning
            )
            for attr in unloaded_attrs:
                # Try reading this attribute from an ID in in_dict.
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
                    # If the attribute was originally JSON serializable,
                    # this is correct loading.
                    setattr(out_obj, attr, in_dict[attr])

        # Add this object to the dictionary of loaded objects.
        if obj_id is not None:
            loaded_dict[obj_id] = out_obj
        return out_obj

    @classmethod
    def from_dict(cls, in_dict, context_dict, loaded_dict, cls_kwargs=None):
        """ Converts a JSON serializable representation of an ACN-Sim
        object into an actual ACN-Sim object.

        Args:
            in_dict (Dict[str, JSON Serializable]): A JSON Serializable
                representation of this object's attributes.
            context_dict (Dict[str, JSON Serializable]): Dict mapping
                object ID's to object JSON serializable representations.
            loaded_dict (Dict[str, BaseSimObj-like]): Dict mapping
                object ID's to ACN-Sim objects.
            cls_kwargs (Dict[str, object]): Optional extra arguments
                to be passed to object's constructor.

        Returns:
            BaseSimObj-like: Loaded object
        """
        raise NotImplementedError
