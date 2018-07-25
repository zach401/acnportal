API Reference
=============

``Interface``
-------------

The Interface is used by the user who is writing the scheduling algorithm to extract relevant data from the simulator.

When extending the ``BaseAlgorithm`` class the Interface instance has already been created and is accessed by:

.. code-block:: python

    self.interface

.. autoclass:: sim.acnlib.Interface.Interface
    :members:

``ACNsim``
----------

.. autoclass:: sim.acnlib.ACNsim.ACNsim
    :members:

``TestCase``
------------

.. autofunction:: sim.acnlib.TestCase.generate_test_case_local

.. autoclass:: sim.acnlib.TestCase.TestCase

``OutputAnalyzer``
------------------

.. autoclass:: sim.acnlib.OutputAnalyzer.OutputAnalyzer
    :members:

``SimulationOutput``
--------------------

.. autoclass:: sim.acnlib.SimulationOutput.SimulationOutput
    :members:

``Event``
+++++++++

.. autoclass:: sim.acnlib.SimulationOutput.Event
    :members:

``EV``
------

.. autoclass:: sim.acnlib.EV.EV

``EVSE``
--------

.. autoclass:: sim.acnlib.EVSE.EVSE

