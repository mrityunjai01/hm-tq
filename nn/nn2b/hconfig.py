from dataclasses import dataclass


@dataclass
class HConfig:
    mult_factor: float = 1.5
    selector_prefix_len: int = 3
    min_non_blanks: int = 5
    max_blanks: int = 2
    span_start: int = 6
    min_total_occ: int = 4
