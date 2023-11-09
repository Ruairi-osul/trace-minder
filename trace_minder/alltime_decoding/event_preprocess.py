import numpy as np
from binit import which_bin


def event_timeseries_make(
    time_arr: np.ndarray,
    event_starts: np.ndarray,
    t_after: float,
    t_before: float = 0,
    in_event_name: str = "in_event",
    not_in_event_name: str = "not_in_event",
) -> np.ndarray:
    """
    Given a set of event starts, and a time array, return a new array that,
    for each time point, demarkates whether it was during the event or not.

    Parameters
    ----------
    time_arr : np.ndarray
        Array of time points.
    event_starts : np.ndarray
        Array of event start times.
    t_after : float
        Time after event start to include in event.
    t_before : float, optional
        Time before event start to include in event, by default 0.
    in_event_name : str, optional
        Name to give to time points that are in the event, by default "in_event".
    not_in_event_name : str, optional
        Name to give to time points that are not in the event, by default "not_in_event".

    Returns
    -------
    np.ndarray
        Array of same length as time_arr, with in_event_name or not_in_event_name

    """
    nan_mask = which_bin(
        arr=time_arr, bin_edges=event_starts, time_before=t_before, time_after=t_after
    )
    return np.where(np.isnan(nan_mask), not_in_event_name, in_event_name)


class EventTimeseriesMaker:
    def __init__(
        self,
        t_after: float,
        t_before: float = 0,
        in_event_name: str = "in_event",
        not_in_event_name: str = "not_in_event",
    ):
        self.t_after = t_after
        self.t_before = t_before
        self.in_event_name = in_event_name
        self.not_in_event_name = not_in_event_name

    def __call__(self, time_arr: np.ndarray, event_starts: np.ndarray) -> np.ndarray:
        return event_timeseries_make(
            time_arr=time_arr,
            event_starts=event_starts,
            t_after=self.t_after,
            t_before=self.t_before,
            in_event_name=self.in_event_name,
            not_in_event_name=self.not_in_event_name,
        )
