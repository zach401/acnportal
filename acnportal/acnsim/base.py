# coding=utf-8
"""
This module contains a base class shared by all ACN-Sim objects.
"""
import json
import operator
import os
from typing import Optional, Dict, Any, Tuple

import numpy as np

# noinspection PyProtectedMember
from pydoc import locate
import warnings
import pkg_resources
import pandas
from pandas.io.common import stringify_path, get_handle, get_filepath_or_buffer
import numpy

__NOT_SERIALIZED_FLAG__ = "__NOT_SERIALIZED__"


# Uses methodology discussed in https://stackoverflow.com/a/16372436
# by Martijn Pieters.


def _operator_error_hooks(cls):
    operator_hooks = [
        name for name in dir(operator) if name.startswith("__") and name.endswith("__")
    ]

    def add_hook(name):
        # noinspection PyUnusedLocal
        def op_hook(*args, **kwargs):
            raise TypeError(
                f"This object is a stub object whose methods are not "
                f"callable. Call {name} after correctly instantiating"
                f" this object."
            )

        try:
            setattr(cls, name, op_hook)
        except (AttributeError, TypeError):
            pass

    for hook_name in operator_hooks:
        add_hook(hook_name)
    return cls


@_operator_error_hooks
class ErrorAllWrapper:
    """
    This wrapper class wraps a string representing an object that acnsim
     does not know how to
    serialize. If any operator or method is called on the wrapped
    string, an error is raised. The only attribute accessible is the
    data attribute, which returns the wrapped string.
    """

    def __init__(self, data):
        self._data = data

    def __getattr__(self, item):
        if item == "data":
            return
        raise TypeError(
            f"This object is a stub object whose methods are not "
            f"callable. Call {item} after correctly instantiating this "
            f"object."
        )

    @property
    def data(self):
        return self._data


class NpEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            json.JSONEncoder.default(self, o)


class BaseSimObj:
    """
    Base class for all ACN-Sim objects. Includes functions for
    representation and serialization of ACN-Sim objects.
    """

    def __repr__(self):
        """
        General string representation of an ACN-Sim object. Unless they
        are non-iterable builtins, attributes' default (object) repr
        functions are used.

        Returns:
            str: A representation of the object in the following form:
                [module name].[class name](attr1=[value1],
                                           attr2=[value2],
                                           ...)
        """
        attr_repr_lst = []
        for key, value in self.__dict__.items():
            if value.__class__.__module__ == "builtins":
                try:
                    _ = iter(value)
                except TypeError:
                    attr_repr_lst.append(f"{key}={value}")
                else:
                    attr_repr_lst.append(f"{key}={object.__repr__(value)}")
            else:
                attr_repr_lst.append(f"{key}={object.__repr__(value)}")
        attr_repr = ", ".join(attr_repr_lst)
        return f"{self.__module__}.{self.__class__.__name__}({attr_repr})"

    @staticmethod
    def _none_to_empty_dict(*args):
        """
        Returns a new args list that replaces each None arg with {}.
        """
        out_arg_lst = []
        for arg in args:
            if arg is None:
                out_arg_lst.append({})
            else:
                out_arg_lst.append(arg)
        return out_arg_lst

    def to_json(self, path_or_buf=None):
        """ Returns a JSON string representing self.
        Currently, only non-compressed file types are supported as the
        output file type.

        Args:
            path_or_buf (FilePathOrBuffer): File path or object. If not
            specified, the result is returned as a string.
        """
        # The code here is from pandas 1.0.1, io.json.to_json(), with
        # modifications.
        json_serializable_data = self._to_registry()[0]
        if json_serializable_data["version"] is None:
            warnings.warn(
                f"Missing a recorded version of acnportal in the dict "
                f"representation. Loading will not run an acnportal "
                f"version check.",
                UserWarning,
            )
        if json_serializable_data["dependency_versions"] is None:
            warnings.warn(
                f"Missing recorded versions of dependencies in the "
                f"dict representation. Loading will not run a "
                f"dependency version check.",
                UserWarning,
            )
        path_or_buf = stringify_path(path_or_buf)
        if isinstance(path_or_buf, str):
            fh, _ = get_handle(path_or_buf, "w")
            try:
                json.dump(json_serializable_data, fh, cls=NpEncoder)
                # Add a newline to the EOF.
                fh.write("\n")
            finally:
                fh.close()
        elif path_or_buf is None:
            return json.dumps(json_serializable_data, cls=NpEncoder)
        else:
            json.dump(json_serializable_data, path_or_buf, cls=NpEncoder)
            # Add a newline to the EOF.
            path_or_buf.write("\n")

    def _to_registry(self, context_dict=None):
        """
        Returns a JSON serializable representation of self.

        The serializer is invoked using `obj.to_json()`, but the
        conversion into a serializable representation occurs in this
        method.

        This method is protected (i.e. underscored) so users do not
        directly use this method. PyNoInspection comments have been
        added where this function is used within the package.

        Serializer behaviors:

        - Any native ACN-Sim object (that is, any unaltered object
        provided in `acnsim`) dumps without error and with
        accurate preservation of attributes.

        - An extension of an ACN-Sim object that appropriately defines a
        `_to_dict` method will dump without error and with
        accurate attribute preservation. This also applies to ACN-Sim
        objects that have as attributes extensions of ACN-Sim objects
        with defined `_to_dict` methods.

        - An extension that doesn't define `_to_dict` will dump without
        error, but with partial preservation of attributes. Objects that
        do not inherit from BaseSimObj and either are not themselves
        JSON serializable or contain attributes that are not JSON
        serializable will not be serialized.

        - A warning is thrown for each attribute not explicitly handled
        in the serialization.

        - A warning is thrown for each attribute that is neither naively
        serializable nor handled by any object's `_to_registry` method.

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
            Dict[str, JSON Serializable]: For clarity, the context_dict
                is returned since it is modified by this function.

        Warns:
            UserWarning: If any attributes are present in the object
                not handled by the object's `_to_dict` method.
            UserWarning: If any of the unhandled attributes are not
                JSON serializable.

        """
        first_call = context_dict is None
        (context_dict,) = self._none_to_empty_dict(context_dict)
        obj_id = f"{id(self)}"

        # Check if this object has already been converted, and return
        # the appropriate dict if this is the case.
        if obj_id in context_dict:
            return {"id": obj_id, "context_dict": context_dict}, context_dict

        obj_type = f"{self.__module__}.{self.__class__.__name__}"

        # Get a dictionary of the attributes of this object using the
        # object's _to_dict method.
        attribute_dict, context_dict = self._to_dict(context_dict)

        # Check that all attributes have been serialized.
        # Warn if some attributes weren't serialized.
        if attribute_dict.keys() != self.__dict__.keys():
            unserialized_keys = set(self.__dict__.keys()) - set(attribute_dict.keys())
            warnings.warn(
                f"Attributes {unserialized_keys} present in object of "
                f"type {obj_type} but not handled by object's _to_dict "
                f"method. Serialized object may not load correctly. "
                f"Write a _to_dict method and re-dump, or write an "
                f"appropriate _from_dict method to accurately load.",
                UserWarning,
            )
            for key in unserialized_keys:
                unserialized_attr = self.__dict__[key]
                if isinstance(unserialized_attr, BaseSimObj):
                    # Try calling the attr's _to_registry method.
                    # noinspection PyProtectedMember
                    registry, context_dict = unserialized_attr._to_registry(
                        context_dict=context_dict
                    )
                    attribute_dict[key] = registry["id"]
                    continue
                # Try dumping the object using the native JSON
                # serializer.
                try:
                    json.dumps(unserialized_attr)
                except TypeError:
                    warnings.warn(
                        f"Attribute {key} could not be serialized. "
                        f"Dumping the attribute's repr() "
                        f"representation. This attribute will not be "
                        f"fully loaded.",
                        UserWarning,
                    )
                    attribute_dict[key] = [
                        __NOT_SERIALIZED_FLAG__,
                        repr(unserialized_attr),
                    ]
                else:
                    attribute_dict[key] = unserialized_attr

        obj_dict = {"class": obj_type, "attributes": attribute_dict}
        context_dict[obj_id] = obj_dict

        # Check versions of acnportal and certain dependencies.
        # We only need to check the versions if the context dict is
        # empty, indicating the first level of the recursive call.
        acnportal_version, dependency_versions = None, None
        if first_call:
            acnportal_version = pkg_resources.require("acnportal")[0].version
            dependency_versions = {
                "numpy": numpy.__version__,
                "pandas": pandas.__version__,
            }

        return (
            {
                "id": obj_id,
                "context_dict": context_dict,
                "version": acnportal_version,
                "dependency_versions": dependency_versions,
            },
            context_dict,
        )

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Converts the object's attributes into a JSON serializable
        dict. Each ACN-Sim object defines this method differently.

        Args:
            context_dict (Dict[str, JSON Serializable]): Dict mapping
                object ID's to object JSON serializable representations.

        Returns:
            Dict[str, JSON Serializable]: Dict of the object's
                attributes, converted to JSON serializable forms.
            context_dict (Dict[str, JSON Serializable]): Dict mapping
                object ID's to object JSON serializable representations.
                This is returned as context_dict may have been modified
                by this method.
        """
        raise NotImplementedError

    @staticmethod
    def _build_from_id(obj_id, context_dict, loaded_dict=None):
        """
        Given an object ID and a dictionary mapping object ID's to JSON
        serializable representations of ACN-Sim objects, returns the ACN-Sim
        object represented by the given object_id.

        This method is protected (i.e. underscored) so users do not
        directly use this method. PyNoInspection comments have been
        added where this function is used within the package.

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
            Dict[str, BaseSimObj-like]: Dict mapping object ID's to ACN-Sim
                objects. This is the loaded_dict passed to and modified by
                this function.

        Raises:
            KeyError: Raised if `obj_id` is not found in `context_dict`.
        """
        # Check if this object has already been loaded; return the loaded
        # object if this is the case.
        if obj_id in loaded_dict:
            return loaded_dict[obj_id], loaded_dict

        if obj_id not in context_dict:
            raise KeyError(f"Object with ID {obj_id} not found in context_dict.")

        # Get the class of this object from the context_dict.
        obj_type = context_dict[obj_id]["class"]
        obj_class = locate(obj_type)

        # 'version' is None since we've already checked the version of the
        # parent object.
        # noinspection PyProtectedMember
        obj, loaded_dict = obj_class._from_registry(
            {
                "id": obj_id,
                "context_dict": context_dict,
                "version": None,
                "dependency_versions": None,
            },
            loaded_dict=loaded_dict,
        )

        loaded_dict[obj_id] = obj
        return obj, loaded_dict

    @classmethod
    def from_json(cls, path_or_buf=None):
        """ Returns an ACN-Sim object loaded from in_registry.
        Note URLs have not been tested as path_or_buf input.

        Args:
            path_or_buf (Union[str, FilePathOrBuffer]): a valid JSON
                str, path object or file-like object. Any valid string
                path is acceptable.
        """
        # The code here is from pandas 1.0.1, io.json.from_json(), with
        # modifications.
        filepath_or_buffer, _, _, should_close = get_filepath_or_buffer(path_or_buf)

        exists = False
        if isinstance(filepath_or_buffer, str):
            try:
                exists = os.path.exists(filepath_or_buffer)
            except (TypeError, ValueError):
                pass

        if exists:
            filepath_or_buffer, _ = get_handle(filepath_or_buffer, "r")
            should_close = True

        if isinstance(filepath_or_buffer, str):
            should_close = False
            out_registry = json.loads(filepath_or_buffer)
        else:
            out_registry = json.load(filepath_or_buffer)
        if should_close:
            filepath_or_buffer.close()

        if out_registry["version"] is None:
            warnings.warn(
                f"Missing a recorded version of acnportal in the "
                f"loaded registry. Object may have been dumped with a "
                f"different version of acnportal.",
                UserWarning,
            )
        if out_registry["dependency_versions"] is None:
            warnings.warn(
                f"Missing recorded dependency versions of acnportal in "
                f"the loaded registry. Object may have been dumped "
                f"with different dependency versions of acnportal.",
                UserWarning,
            )

        out_obj = cls._from_registry(out_registry)[0]
        return out_obj

    @classmethod
    def _from_registry(cls, in_registry, loaded_dict=None):
        """
        Returns an object of type `cls` from a JSON serializable
        representation of the object.

        The deserializer is invoked using `cls.from_json()`, but the
        conversion from a serializable representation to an ACN-Sim
        object occurs in this method.

        This method is protected (i.e. underscored) so users do not
        directly use this method. PyNoInspection comments have been
        added where this function is used within the package.

        Deserializer behaviors:

        - Any native ACN-Sim object (that is, any unaltered object
        provided in `acnsim`) that was dumped (serialized) with the
        object's `_to_registry` method loads without error and with
        accurate values for each attribute, assuming the original object
        was of type `cls`.

        - An extension of an ACN-Sim object that appropriately defines a
        `_from_dict` method will load without error and with
        accurate attribute values. This also applies to ACN-Sim
        objects that have as attributes extensions of ACN-Sim objects
        with defined `_from_dict` methods.

        - An extension that doesn't define `_from_dict` may not load
        correctly. The extension will load correctly only if the
        constructor of the object whose `_from_dict` method is called
        takes the same arguments as this object, and any extra
        attributes are JSON-serializable or readable from an ID.

        - A warning is thrown for each attribute not explicitly loaded
        in the deserialization.

        - A warning is thrown for each attribute that is neither naively
        serializable nor readable from an ID or with a `_from_registry`
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
            in_registry (Dict[str, JSON Serializable]): A JSON
                Serializable representation of this object. Takes the
                form:
                    ```
                    {'id': [This object's id],
                     'context_dict': [The dict mapping all object id's
                                      to their JSON serializable
                                      representations]}
                    ```
            loaded_dict (Dict[str, BaseSimObj-like]): Dict mapping
                object ID's to loaded ACN-Sim objects.

        Returns:
            BaseSimObj-like: Loaded object.
            loaded_dict (Dict[str, BaseSimObj-like]): Dict mapping
                object ID's to loaded ACN-Sim objects.

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
                not handled by the object's `_from_dict` method.
            UserWarning: If any of the unhandled attributes are not
                readable from an ID. This is fine unless this attribute
                is not meant to be JSON serializable, in which case
                the loaded attribute will be incorrect.

        """
        (loaded_dict,) = cls._none_to_empty_dict(loaded_dict)
        obj_id, context_dict, acnportal_version, dependency_versions = (
            in_registry["id"],
            in_registry["context_dict"],
            in_registry["version"],
            in_registry["dependency_versions"],
        )

        # Check current versions of acnportal and certain dependencies
        # against serialized versions.
        if acnportal_version is not None:
            current_version = pkg_resources.require("acnportal")[0].version
            if current_version != acnportal_version:
                warnings.warn(
                    f"Version {acnportal_version} of input acnportal "
                    f"object does not match current version "
                    f"{current_version}."
                )

        if dependency_versions is not None:
            current_dependency_versions = {
                "numpy": numpy.__version__,
                "pandas": pandas.__version__,
            }
            for pkg in dependency_versions.keys():
                if current_dependency_versions[pkg] != dependency_versions[pkg]:
                    warnings.warn(
                        f"Current version of dependency {pkg} does not "
                        f"match serialized version. "
                        f"Current: {current_dependency_versions[pkg]}, "
                        f"Serialized: {dependency_versions[pkg]}.",
                    )

        try:
            obj_dict = context_dict[obj_id]
        except KeyError:
            raise KeyError(f"Object with ID {obj_id} not found in context_dict.")

        # Check if this object has already been converted, and return
        # the appropriate dict if this is the case.
        if obj_id in loaded_dict:
            return loaded_dict[obj_id], loaded_dict

        if obj_dict["class"] != f"{cls.__module__}.{cls.__name__}":
            warnings.warn(
                f"Deserializing as type "
                f"{cls.__module__}.{cls.__name__}. Object was "
                f"serialized as type {obj_dict['class']}.",
                UserWarning,
            )

        # attribute_dict is a dict mapping attribute names to JSON
        # serializable values.
        attribute_dict = obj_dict["attributes"]

        # Call this class' _from_dict method to convert the JSON
        # representation of this object's attributes into the actual
        # object.
        out_obj, loaded_dict = cls._from_dict(attribute_dict, context_dict, loaded_dict)

        # Check that all attributes have been loaded.
        # Warn if some attributes weren't loaded.
        if out_obj.__dict__.keys() != attribute_dict.keys():
            unloaded_attrs = set(attribute_dict.keys()) - set(out_obj.__dict__.keys())
            unrecorded_attrs = set(out_obj.__dict__.keys()) - set(attribute_dict.keys())
            if len(unloaded_attrs) > 0:
                warnings.warn(
                    f"Attributes {unloaded_attrs} present in object of "
                    f"type {obj_dict['class']} but not handled by object's "
                    f"_from_dict method. Loaded object may have inaccurate "
                    f"attributes.",
                    UserWarning,
                )
            if len(unrecorded_attrs) > 0:
                warnings.warn(
                    f"Attributes {unrecorded_attrs} present in object of "
                    f"type {obj_dict['class']} but not recorded in "
                    f"serialization. Loaded object may need to have "
                    f"attributes set. ",
                    UserWarning,
                )
            for attr in unloaded_attrs:
                # Try reading this attribute from an ID in
                # attribute_dict.
                try:
                    out_attr, loaded_dict = cls._build_from_id(
                        attribute_dict[attr], context_dict, loaded_dict=loaded_dict
                    )
                    setattr(out_obj, attr, out_attr)
                except (KeyError, TypeError):
                    if (
                        isinstance(attribute_dict[attr], list)
                        and attribute_dict[attr][0] == __NOT_SERIALIZED_FLAG__
                    ):
                        warnings.warn(
                            f"Loader for attribute {attr} not found. "
                            f"Setting attribute {attr} directly.",
                            UserWarning,
                        )
                        loaded_attr_value = ErrorAllWrapper(attribute_dict[attr][1])
                    else:
                        loaded_attr_value = attribute_dict[attr]
                    # If the attribute was originally JSON serializable,
                    # this is correct loading.
                    try:
                        setattr(out_obj, attr, loaded_attr_value)
                    except AttributeError:
                        # attr could be protected for out_obj. Warn if it is.
                        warnings.warn(
                            f"Attribute {attr} is protected for object of class "
                            f"{out_obj.__class__}. Not setting {attr} to "
                            f"{loaded_attr_value}. Please see {out_obj.__class__} "
                            f"implementation for more info."
                        )

        # Add this object to the dictionary of loaded objects.
        loaded_dict[obj_id] = out_obj
        return out_obj, loaded_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, "BaseSimObj"]] = None,
    ) -> Tuple["BaseSimObj", Dict[str, "BaseSimObj"]]:
        """ Converts a JSON serializable representation of an ACN-Sim
        object into an actual ACN-Sim object.

        Args:
            attribute_dict (Dict[str, JSON Serializable]): A JSON
                Serializable representation of this object's attributes.
            context_dict (Dict[str, JSON Serializable]): Dict mapping
                object ID's to object JSON serializable representations.
            loaded_dict (Dict[str, BaseSimObj-like]): Dict mapping
                object ID's to ACN-Sim objects.

        Returns:
            BaseSimObj-like: Loaded object
            (Dict[str, BaseSimObj-like]): Dict mapping object ID's to
                ACN-Sim objects. This is returned as loaded_dict may
                have been modified by this method.
        """
        raise NotImplementedError
