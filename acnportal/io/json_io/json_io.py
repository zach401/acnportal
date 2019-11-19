# import json
# from .json_registry import *
# from acnportal import acnsim
# from datetime import datetime, date
# from dateutil import parser
# import copy
# import warnings
# import numpy as np

# # JSON Serialization module. All classes in acnsim are registered with
# # encoder and decoder functions that detail how to write and read an
# # acnsim object (e.g. Simulator, Network, EV, etc.) to and from a JSON
# # file. If the user defines a new class, the user must register the
# # new class for the new class to be JSON serializable.

# def to_json(obj):
#     """
#     Returns a JSON representation of the object obj if the object is
#     either natively JSON serializable or the object's class is
#     registered. Raises an InvalidJSONError if the class is not
#     registered and is not natively JSON serializable. Precedence is
#     given to the registered class encoder over any native
#     serialization method if such a conflict occurs.

#     Args:
#         obj (Object): Object to be converted into JSON-serializable
#             form.

#     Returns:
#         JSON str

#     Raises:
#         InvalidJSONError: Exception raised when object whose class
#             has not yet been registered is attempted to be
#             serialized.
#     """
#     class_name = obj.__class__.__name__

#     # If the class is not registered, try directly encoding to JSON.
#     if class_name not in ENCODER_REGISTRY:
#         try:
#             return json.dumps(obj)
#         except TypeError:
#             raise InvalidJSONError(f"Attempting to encode object "
#                 f"{obj} without registered encoder for class "
#                 f"{class_name}")

#     # Use registered encoder function to encode to JSON.
#     encoder = ENCODER_REGISTRY[class_name]
#     # Include class in final json string.
#     return encoder(copy.deepcopy(obj))
#     # return json.dumps({
#     #     'class': class_name,
#     #     'args': encoder(copy.deepcopy(obj))
#     # })

# def from_json(in_json):
#     """
#     Converts a JSON representation of an object into an instance of
#     the class. If the object class is registered with a decoder, that
#     decoder is used to decode the object. If not, an InvalidJSONError
#     is raised.

#     Args:
#         obj_json (JSON str): JSON string from which to construct the
#             object.

#     Returns:
#         Object

#     Raises:
#         InvalidJSONError: Exception raised when object whose class
#             has not yet been registered is attempted to be
#             read from JSON.
#     """
#     json_dict = json.loads(in_json)

#     # If the object does not have a class attribute in the json_dict,
#     # acnsim's decoding will not work with it. An InvalidJSONError is
#     # raised.
#     if 'class' not in json_dict:
#         raise InvalidJSONError(f"Decoded JSON has no 'class' "
#             f"attribute for decoder selection. Use JSON loading "
#             f"directly or check JSON source.")

#     # If the object has a class attribute but the class is not
#     # registered with a decoder, an InvalidJSONError is raised. The
#     # user may need to register the class to decode.
#     if json_dict['class'] not in DECODER_REGISTRY:
#         raise InvalidJSONError(f"Class {json_dict['class']} to "
#             f"decode is not registered with a decoder.")

#     decoder = DECODER_REGISTRY[json_dict['class']]
#     return decoder(json_dict)
