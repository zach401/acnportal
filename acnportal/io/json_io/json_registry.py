from .json_encoders import *
from .json_decoders import *

class RegistrationError(Exception):
    """
    Exception which is raised when trying to register a class with
    invalid parameters.
    """
    pass

def register_json_class(reg_cls, encoder=None, decoder=None):
    """
    Registers a class for JSON serialization, using user-provided
    encoder and decoder functions to handle reading/writing.

    If no encoder function is provided, the superclass is checked for
    membership in the registry; if the superclass is a member, the
    superclass encoder function is used. Otherwise, an error is
    thrown.

    If no decoder function is provided, the superclass is checked for
    membership in the registry; if the superclass is a member, the
    superclass decoder function is used. Otherwise, an error is
    thrown.

    Args:
        reg_cls (Class): Class to register with encoder/decoder
            registry.
        encoder (obj -> JSON str) : Takes as input an instance
            of reg_cls and returns a JSON string representing the
            instance.
        decoder (JSON str -> obj) : Takes as input a JSON string
            representing an instance of reg_cls and returns an
            instance of reg_cls.

    Returns:
        None

    Raises:
        RegistrationError: Exception raised when class registration
            fails.
    """
    assert ENCODER_REGISTRY.keys() == DECODER_REGISTRY.keys()
    class_name = reg_cls.__name__

    # Check if class is already registered
    if class_name in ENCODER_REGISTRY:
        warnings.warn(
            f"Class {class_name} has already been registered. "
            "Attempting to override previous encoder/decoder "
            "functions.")

    # If one of encoder/decoder is omitted, try using the superclass
    # encoder/decoder. If the superclass isn't registered, throw an
    # error.
    if encoder is None or decoder is None:
        assert hasattr(reg_cls, super)
        if reg_cls.super().__name__ in ENCODER_REGISTRY:
            ENCODER_REGISTRY[class_name] = \
                ENCODER_REGISTRY[reg_cls.super().__name__]
            DECODER_REGISTRY[class_name] = \
                DECODER_REGISTRY[reg_cls.super().__name__]
        else:
            raise RegistrationError(f"Class {class_name} has no "
                "registered superclass. Please provide encoder and "
                "decoder functions.")
    else:
        ENCODER_REGISTRY[class_name] = encoder
        DECODER_REGISTRY[class_name] = decoder