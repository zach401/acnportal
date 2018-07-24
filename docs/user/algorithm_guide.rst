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

.. autoclass:: sim.BaseAlgorithm.BaseAlgorithm
    :members: get_decreased_charging_rate, get_decreased_charging_rate_nz, get_increased_charging_rate

The ``BaseAlgorithm.schedule`` function
+++++++++++++++++++++++++++++++++++++++

To write a custom scheduling algorithm the user must implement the ``schedule`` function in the class inheriting the
``BaseAlgorithm``.

.. automethod:: sim.BaseAlgorithm.BaseAlgorithm.schedule

Simulation interface (API)
--------------------------

The ``BaseAlgorithm`` class provides the API to the simulator which provides information from the simulation that can
be useful when writing a scheduling algorithm. The interface object can be accessed by writing ``self.interface``
from the new scheduler class.

For example, to access the maximum charging rate that is allowed in the simulation it is possible to write:

.. code-block:: python

    max_rate = self.interface.get_max_charging_rate()

More information about the API and what resources are available are described here.

Examples
--------

Here follows some examples of already implemented algorithms located in the ``BaseAlgorithm`` module.

To use them in the simulation script, write the following code (the simulation code should be located in ``sim/``):

.. code-block:: python

    from BaseAlgorithm import EarliestDeadlineFirstAlgorithm, LeastSlackFirst, MLLF
    scheduler1 = EarliestDeadlineFirstAlgorithm()
    scheduler2 = LeastSlackFirst()
    scheduler3 = MLLF(preemtion=True, queue_length=2)

Earliest Deadline First
+++++++++++++++++++++++

.. code-block:: python

    class EarliestDeadlineFirstAlgorithm(BaseAlgorithm):

        def __init__(self):
            pass

        def schedule(self, active_EVs):
            schedule = {}
            earliest_EV = self.get_earliest_EV(active_EVs)
            last_applied_pilot_signals = self.interface.get_last_applied_pilot_signals()
            for ev in active_EVs:
                charge_rates = []
                last_pilot_signal = 0
                allowable_pilot_signals = self.interface.get_allowable_pilot_signals(ev.station_id)
                if ev.session_id in last_applied_pilot_signals:
                    last_pilot_signal = last_applied_pilot_signals[ev.session_id]
                if ev.session_id == earliest_EV.session_id:
                    new_rate = self.get_increased_charging_rate(last_pilot_signal, allowable_pilot_signals)
                    charge_rates.append(new_rate)
                else:
                    new_rate = self.get_decreased_charging_rate_nz(last_pilot_signal, allowable_pilot_signals)
                    charge_rates.append(new_rate)
                schedule[ev.session_id] = charge_rates
            return schedule

        def get_earliest_EV(self, EVs):
            earliest_EV = None
            for ev in EVs:
                if earliest_EV == None or earliest_EV.departure > ev.departure:
                    earliest_EV = ev
            return earliest_EV

Least Laxity First
++++++++++++++++++

.. code-block:: python

    class LeastLaxityFirstAlgorithm(BaseAlgorithm):

        def __init__(self):
            self.max_charging_rate = self.max_charging_rate = self.interface.get_max_charging_rate()

        def schedule(self, active_EVs):
            schedule = {}
            current_time = self.interface.get_current_time()
            least_slack_EV = self.get_least_laxity_EV(active_EVs, current_time)
            last_applied_pilot_signals = self.interface.get_last_applied_pilot_signals()
            for ev in active_EVs:
                charge_rates = []
                last_pilot_signal = 0
                allowable_pilot_signals = self.interface.get_allowable_pilot_signals(ev.station_id)
                if ev.session_id in last_applied_pilot_signals:
                    last_pilot_signal = last_applied_pilot_signals[ev.session_id]
                if ev.session_id == least_slack_EV.session_id:
                    new_rate = self.get_increased_charging_rate(last_pilot_signal, allowable_pilot_signals)
                    charge_rates.append(new_rate)
                else:
                    new_rate = self.get_decreased_charging_rate_nz(last_pilot_signal, allowable_pilot_signals)
                    charge_rates.append(new_rate)
                schedule[ev.session_id] = charge_rates
            return schedule

            return schedule

        def get_least_laxity_EV(self, EVs, current_time):
            least_slack_EV = None
            least_slack = 0
            for ev in EVs:
                current_slack = self.get_slack_time(ev, current_time)
                if least_slack_EV == None or least_slack > current_slack:
                    least_slack = current_slack
                    least_slack_EV = ev
            return least_slack_EV

        def get_laxity(self, EV, current_time):
            laxity = (EV.departure - current_time) - (EV.requested_energy - EV.energy_delivered) / self.max_charging_rate
            return laxity