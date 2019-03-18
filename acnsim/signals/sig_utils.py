import math


def periods_since_midnight(t, period):
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((t - midnight).total_seconds() / period)


def extended_schedule(schedule, start, length):
    tile_multiple = math.ceil((start + length) / len(schedule))
    tiled_schedule = schedule * tile_multiple
    return tiled_schedule[start:start + length]
