# DO NOT EDIT - AUTOMATICALLY GENERATED BY tests/make_test_stubs.py!
from typing import List
from typing import (
    Optional,
    Union,
)


def Nu_plate_Khan_Khan(Re: float, Pr: float, chevron_angle: float) -> float: ...


def Nu_plate_Kumar(
    Re: float,
    Pr: float,
    chevron_angle: float,
    mu: Optional[float] = ...,
    mu_wall: Optional[float] = ...
) -> float: ...


def Nu_plate_Martin(Re: float, Pr: float, plate_enlargement_factor: float, variant: str = ...) -> float: ...


def Nu_plate_Muley_Manglik(Re: float, Pr: float, chevron_angle: float, plate_enlargement_factor: float) -> float: ...

__all__: List[str]