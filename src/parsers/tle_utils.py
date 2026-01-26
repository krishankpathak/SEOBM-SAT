import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime

def datetime_to_jday(dt: datetime):
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        dt.second + dt.microsecond * 1e-6
    )
    return jd, fr


def tle_to_eci(line1: str, line2: str, epoch: datetime):
    sat = Satrec.twoline2rv(line1, line2)
    jd, fr = datetime_to_jday(epoch)
    error, r, v = sat.sgp4(jd, fr)

    if error != 0:
        raise RuntimeError(f"SGP4 error code: {error}")

    return np.array(r), np.array(v)
