"""
This library to configure the application..
"""

import zoneinfo
from typing import Literal


class Config:  # pragma: no cover
    """
    Class to store the configuration of the application.
    """

    def __init__(
        self,
        *,
        map_elements: list | str | None = None,
        time_unit: Literal["ns", "us", "ms"] = "us",
    ) -> None:
        """
        Initializes the configuration of the application.

        :param struct_as_map: The list of columns to map. Default is None.

        :param time_zone: The time zone to use. Default is `Europe/Paris`.

        :param time_unit: The time unit to use. Default is "us".

        :return: None
        """
        if isinstance(map_elements, str):
            map_elements = [map_elements]
        self._check_map_elements(map_elements)
        self._map_elements = map_elements

        self._check_time_unit(time_unit)
        self._time_unit = time_unit

    @property
    def time_unit(self) -> Literal["ns", "us", "ms"]:
        """
        The time unit to get.

        :return: The time unit.
        """
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value: Literal["ns", "us", "ms"]) -> None:
        """
        The time unit to set.

        :param value: The time unit to set.

        :return: None
        """
        self._check_time_unit(value)
        self._time_unit = value

    @property
    def map_elements(self) -> None:
        """
        This property gets the map elements.

        :return: The map elements.
        """
        return self._map_elements

    @map_elements.setter
    def map_elements(self, value: list | str) -> None:
        """
        The map elements to set.

        :param value: The map elements to set.

        :return: None
        """
        if isinstance(value, str):
            value = [value]
        self._check_map_elements(value)
        self._map_elements = value

    @staticmethod
    def _check_time_zone(value: str | None) -> None:
        """
        Method to check the time zone.

        :param value: The value to check.

        :return: None
        """
        if value is None:
            return
        if not isinstance(value, str):
            msg = "The value must be a string."
            raise TypeError(msg)
        if value not in zoneinfo.available_timezones():
            msg = "The time zone is not valid. "
            "must be one of the available time zones.\n"
            "Check the documentation:\n"
            "https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Datetime.html"
            raise ValueError(msg)

    @staticmethod
    def _check_time_unit(value: str) -> None:
        """
        Method to check the time unit.

        :param value: The value to check.

        :return: None
        """
        if not isinstance(value, str):
            msg = "The value must be a string."
            raise TypeError(msg)
        if value not in {"us", "ns", "ms"}:
            msg = "The time unit is not valid. "
            "must be one of 'us', 'ns', 'ms'.\n"
            "Check the documentation:\n"
            "https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Datetime.html"
            raise ValueError(msg)

    @staticmethod
    def _check_map_elements(value: list) -> None:
        """
        This method checks the map elements.

        :param value: The value to check.

        :return: None
        """
        if value and not isinstance(value, list):
            msg = "The value must be a list."
            raise TypeError(msg)
