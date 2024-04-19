from collections import OrderedDict
import numpy as np


# Removed all the dynamic ones as I am currently interested in working with only static CBO methods
# Might need to revisit these methods
class PISHCAT_SEM:
    @staticmethod
    def static():

        P = lambda noise, sample: noise
        I = lambda noise, sample: noise
        S = lambda noise, sample: sample["I"] + noise
        H = lambda noise, sample: sample["P"] + noise
        C = lambda noise, sample: sample["H"] + noise
        A = lambda noise, sample: sample["I"] + sample["P"] + noise
        T = lambda noise, sample: sample["C"] + sample["A"] + noise
        return OrderedDict([("P", P), ("I", I), ("S", S), ("H", H), ("C", C), ("A", A), ("T", T)])


class StationaryDependentSEM:
    @staticmethod
    def static():

        X = lambda noise, sample: noise
        Z = lambda noise, sample: np.exp(-sample["X"]) + noise
        Y = lambda noise, sample: np.cos(sample["Z"]) - np.exp(-sample["Z"] / 20.0) + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class LinearMultipleChildrenSEM:
    """
    Test DAG for nodes within a slice that have more than one child _within_ the slice.

    Returns
    -------
        None
    """

    @staticmethod
    def static() -> OrderedDict:

        X = lambda noise, sample: 1 + noise
        Z = lambda noise, sample: 2 * sample["X"] + noise
        Y = lambda noise, sample: 2 * sample["Z"] - sample["X"] + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class StationaryDependentMultipleChildrenSEM:
    """
    Test DAG for nodes within a slice that have more than one child _within_ the slice.

    Returns
    -------
        None
    """

    @staticmethod
    def static() -> OrderedDict:

        X = lambda noise, sample: noise
        Z = lambda noise, sample: np.exp(-sample["X"][t]) + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"] / 20.0)
            - np.sin(sample["X"])
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class StationaryIndependentSEM:
    @staticmethod
    def static():
        X = lambda noise, sample: noise
        Z = lambda noise, sample: noise
        Y = (
            lambda noise, sample: -2 * np.exp(-((sample["X"] - 1) ** 2) - (sample["Z"] - 1) ** 2)
            - np.exp(-((sample["X"] + 1) ** 2) - sample["Z"] ** 2)
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryDependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.

    with: intervention_domain = {'X':[-4,1],'Z':[-3,3]}
    """

    def __init__(self, change_point):
        """
        Initialise change point(s).

        Parameters
        ----------
        cp : int
            The temporal index of the change point (cp).
        """
        self.cp = change_point

    @staticmethod
    def static():
        """
        noise: e
        sample: s
        time index: t
        """
        X = lambda e, t, s: e
        Z = lambda e, t, s: s["X"][t] + e
        Y = lambda e, t, s: np.sqrt(abs(36 - (s["Z"][t] - 1) ** 2)) + 1 + e
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryIndependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.
    """

    def __init__(self, change_point):
        self.change_point = change_point

    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -(
                2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
                + np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            )
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

