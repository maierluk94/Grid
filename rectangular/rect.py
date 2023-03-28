from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Point:
    """Representation of a point on the screen. Needed for drawing purposes."""

    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)
    
    def __truediv__(self, f: float) -> Point:
        return Point(self.x / f, self.y / f)

    def amount(self) -> int:
        return abs(self.x) + abs(self.y)
    

@dataclass
class Tile:
    """Representation of a tile in a grid."""

    row: int
    col: int