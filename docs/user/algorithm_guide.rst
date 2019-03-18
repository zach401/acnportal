.. _writing-a-scheduling-algorithm:

Writing a scheduling algorithm
==============================

Last updated: 03/12/2019

This section will describe how to write a scheduling algorithm for Electrical Vehicles charging
at a charging network.

When writing the scheduling algorithm nothing more than the API to the simulation will be know to the
user. The API will provide resources for the user which will be used when implementing the scheduling algorithm.

The ``BaseAlgorithm`` class
---------------------------

All scheduling algorithms for the ACN network must extend the class ``BaseAlgorithm`` to be able to run. The ``BaseAlgorithm``
class is known by the simulator, so for the user to be able to integrate code with the simulator he or she has to extend
the ``schedule`` function of the base class.

Along with the ``schedule`` function there exists some internal functions used by the simulator and also some utlity functions
to make it easier to create the scheduling algorithm.

The utility functions that can be used in the class extending the ``BaseAlgorithm`` class to define the scheduler are:

.. autoclass:: sim.BaseAlgorithm.BaseAlgorithm
    :members: get_decreased_charging_rate, get_decreased_charging_rate_nz, get_increased_charging_rate

The ``BaseAlgorithm.schedule`` function
+++++++++++++++++++++++++++++++++++++++

To write a custom scheduling algorithm the user must implement the ``schedule`` function in the class inheriting the
``BaseAlgorithm``.

.. automethod:: sim.BaseAlgorithm.BaseAlgorithm.schedule

The list that is passed to the ``schedule`` function holds the information about the current status of the active EVs.

The API reference has more information about the :class:`EV<sim.acnlib.EV.EV>` object.

Simulation interface (API)
--------------------------

The ``BaseAlgorithm`` class provides the API to the simulator which provides information from the simulation that can
be useful when writing a scheduling algorithm. The interface object can be accessed by writing ``self.interface``
in a function of the new scheduler class.

For example, to access the maximum charging rate that is allowed in the simulation it is possible to write:

.. code-block:: python

    max_rate = self.interface.get_max_charging_rate()

More information about the API and what resources are available are described :class:`here<sim.acnlib.Interface.Interface>`.

Examples
--------

Here follows some examples of already implemented algorithms located in the ``BaseAlgorithm`` module.

To use them in the simulation script, write the following code (the simulation script should be located in ``sim/``):

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

        def schedule(self, active_evs):
            schedule = {}
            earliest_EV = self.get_earliest_EV(active_evs)
            last_applied_pilot_signals = self.interface.get_last_applied_pilot_signals()
            for ev in active_evs:
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
            pass

        def schedule(self, active_evs):
            self.max_charging_rate = self.interface.get_max_charging_rate()
            schedule = {}
            current_time = self.interface.get_current_time()
            least_slack_EV = self.get_least_laxity_EV(active_evs, current_time)
            last_applied_pilot_signals = self.interface.get_last_applied_pilot_signals()
            for ev in active_evs:
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

Multi Least Laxity First
++++++++++++++++++++++++

.. code-block:: python

    class MLLF(BaseAlgorithm):
        '''
        Multi Least Laxity First

        This algorithm builds upon the Least Laxity scheduling algorithm but also includes
        the possiblity to have many processes (sessions) running at the same time.
        The algorithm calculates a laxity value of every session and then rank them according
        to this value. The sessions with least laxity get moved to a queue which will get prioritized in
        the charging schedule. A session in this queue will receive the maximum pilot signal. The other
        sessions will get the minumum pilot signal

        There is an option to select if the charging schedule will allow preemtion of the active queue.
        If there is no preemtion allowed a session once entered the active queue will not leave until it
        has finished charging. If preemtion is allowed the ready queue will get recalculated every iteration
        to allow new sessions with smaller laxity to interrupt the current sessions in the queue.
        '''

        def __init__(self, preemption=False, queue_length=4):
            self.queue = []
            self.preemption = preemption
            self.queue_length = queue_length

        def schedule(self, active_evs):
            self.max_charging_rate = self.interface.get_max_charging_rate()
            schedule = {}
            current_time = self.interface.get_current_time()
            last_applied_pilot_signals = self.interface.get_last_applied_pilot_signals()

            # No preemtion: check queue and remove non-active EVs
            # Preemtion: Always empty the queue
            for session_id in self.queue:
                if self.preemption:
                    self.queue = []
                else:
                    found = False
                    for ev in active_evs:
                        if ev.session_id == session_id:
                            found = True
                            break
                    if found == False:
                        self.queue.remove(session_id)

            # choose the EVs that should be evaluated for laxity and then sort them
            # If no preemtion the EVs already in the queue should be omitted.
            ev_laxity = []
            for ev in active_evs:
                if self.preemption or (not self.preemption and ev.session_id not in self.queue):
                    ev_info = {'session_id': ev.session_id, 'laxity': self.get_laxity(ev, current_time)}
                    ev_laxity.append(ev_info)

            sorted_ev_laxity = self.sort_by_laxity(ev_laxity)

            # add the EVs to the queue
            ql = len(self.queue)
            for i in range(min(self.queue_length - ql, len(sorted_ev_laxity))):
                ev = sorted_ev_laxity[i]
                self.queue.append(ev['session_id'])

            # calculate the new pilot signals
            for ev in active_evs:
                charge_rates = []
                last_pilot_signal = 0
                allowable_pilot_signals = self.interface.get_allowable_pilot_signals(ev.station_id)
                if ev.session_id in last_applied_pilot_signals:
                    last_pilot_signal = last_applied_pilot_signals[ev.session_id]
                # determine pilot signal
                if ev.session_id in self.queue:
                    new_rate = self.get_increased_charging_rate(last_pilot_signal, allowable_pilot_signals)
                    charge_rates.append(new_rate)
                else:
                    new_rate = self.get_decreased_charging_rate_nz(last_pilot_signal, allowable_pilot_signals)
                    charge_rates.append(new_rate)
                schedule[ev.session_id] = charge_rates
            return schedule


        def get_laxity(self, EV, current_time):
            laxity = (EV.departure - current_time) - (EV.requested_energy - EV.energy_delivered) / self.max_charging_rate
            return laxity

        def sort_by_laxity(self, list):
            return sorted(list, key=lambda ev: ev['laxity'])