'''
Script for generating a test case scenario.
This scenario is stored in a pickled file that will be read when
the simulator starts.
'''

import pickle
from datetime import datetime, timedelta
import random

def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

file_name = 'test_session.p'
sessions = []

start_time = datetime.now()
end_time = start_time + timedelta(days=1)
dt = end_time - start_time
delta_days = dt.days

for i in range(1,20):
    arrival_time = random_date(start_time, end_time)
    departure_time = random_date(arrival_time, arrival_time + timedelta(days=1))
    requested_energy = random.uniform(1,8)
    station_id = i
    sessions.append([arrival_time, departure_time, requested_energy, station_id])

pickle.dump(sessions, open( file_name, "wb"))
print('Random test session generated and saved in the file: ' + file_name)


