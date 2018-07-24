API Reference
=============

Simulator Interface
-------------------

The Interface is used by the user who is writing the scheduling algorithm to extract relevant data from the simulator.

When extending the ``BaseAlgorithm`` class the Interface instance has already been created and is accessed by:

.. code-block:: python

    self.interface

.. autoclass:: sim.acnlib.Interface.Interface
    :members:

The simulator ``ACNsim``
------------------------

.. autoclass:: sim.acnlib.ACNsim.ACNsim
    :members:
