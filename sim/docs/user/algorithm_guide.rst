.. _writing-a-scheduling-algorithm:

Writing a scheduling algorithm
==============================

Last updated: 07/23/2018, Daniel Johansson

This section will describe how to write a scheduling algorithm for Electrical Vehicles charging
at a charging network.

When writing the scheduling algorithm nothing more than the API to the simulation will be know to the
user. The API will provide resources for the user which will be used when implementing the scheduling algorithm.

The ``BaseAlgorithm`` class
---------------------------

All scheduling algorithms for the ACN network must extend the class ``BaseAlgorithm`` to be able to run. The ``BaseAlgorithm``
class is known by the simulator, so for the user to be able to integrate code with the simulator he or she has to extend
the ``schedule`` function of the base class.

Along with the ``schedule`` function there exists some internal functions used by the simulator and also som utlity functions
to make it easier to create the scheduling algorithm.

The utility functions that can be used in the class extending the ``BaseAlgorithm`` class to define the scheduler are:

``self.get_increased_charging_rate(current_rate, allowable_rates)``
    As many EVSEs has limited sets of pilot signal this function returns the next increased pilot signal
    depending of the current pilot signal applied at the charging station.

``self.get_decreased_charging_rate(current_rate, allowable_rates)``
    As many EVSEs has limited sets of pilot signal this function returns the next decreased pilot signal
    depending of the current pilot signal applied at the charging station.

``self.get_decreased_charging_rate_nz(current_rate, allowable_rates)``
    As many EVSEs has limited sets of pilot signal this function returns the next decreased pilot signal
    depending of the current pilot signal applied at the charging station. This function also prevents
    that the calculated pilot signal will be 0 as the pilot signal should never be set to 0 before the
    EV has finished charging.

The ``schedule`` function
-------------------------

``schedule(active_EVs)``
    The schedule function provides the current ``acitve_EVs`` of the simulation which are available to apply a
    pilot signal to.

    This function must return a dictionary with the ``session_id`` as keys and the calculated pilot signals as values

Simulation interface (API)
--------------------------

The ``BaseAlgorithm`` class provides the API to the simulator which provides information from the simulation that can
be usefull when writing a scheduling algorithm. The interface object can be accessed by writing ``self.interface``
from the new scheduler class.

For example, to access the maximum charging rate that is allowed in the simulation it is possible to write:

.. code-block:: python

    max_rate = self.interface.get_max_charging_rate()

Examples
--------