from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConstants:
    MEASURE_RESOLUTION: int
    SIMULATION_RESOLUTION: int
