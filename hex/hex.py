"""Collection of classes needed to define hexagons and their edges and vertices
using cube coordinates for easy math operations.

Based on https://www.redblobgames.com/grids/hexagons/."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Union
import collections
import math

Orientation = collections.namedtuple("Orientation", \
              ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"])
Layout = collections.namedtuple("Layout", ["orientation", "size", "origin"])

ORIENTATION_POINTY = Orientation(math.sqrt(3.0), math.sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0, \
                                 math.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5)
ORIENTATION_FLAT = Orientation(3.0 / 2.0, 0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0), \
                               2.0 / 3.0, 0.0, -1.0 / 3.0, math.sqrt(3.0) / 3.0, 0.0)

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

    def to_hex(self, layout: Layout) -> Hex:
        """Returns the hex at the position of the point."""
        return self.to_fractional_hex(layout).round()

    def to_fractional_hex(self, layout: Layout) -> FractionalHex:
        """Returns the fractional hex at the position of the point."""
        M = layout.orientation
        size = layout.size
        origin = layout.origin
        pt = Point((self.x - origin.x) / size.x, (self.y - origin.y) / size.y)
        q = M.b0 * pt.x + M.b1 * pt.y
        r = M.b2 * pt.x + M.b3 * pt.y
        return FractionalHex(q, r, -q - r)
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Hex:
    """Represents a hex using cube coordinates."""

    q: int
    r: int
    s: int

    def __post_init__(self):
        assert self.q + self.r + self.s == 0, "q + r + s must be 0"

    def __eq__(self, other: Hex):
        return self.q == other.q and self.r == other.r and self.s == other.s

    def __add__(self, other: Hex) -> Hex:
        return Hex(self.q + other.q, self.r + other.r, self.s + other.s)
    
    def __sub__(self, other: Hex) -> Hex:
        return Hex(self.q - other.q, self.r - other.r, self.s - other.s)

    def __mul__(self, k: int) -> Hex:
        return Hex(self.q * k, self.r * k, self.s * k)
    
    def get_length(self) -> int:
        """Returns the distance from the origin."""
        return (abs(self.q) + abs(self.r) + abs(self.s)) // 2
    
    def distance_to(self, h: Hex) -> int:
        """Returns the distance to another hex."""
        return (self - h).get_length()
    
    def get_neighbor(self, hex_direction: Hex) -> Hex:
        """Returns the adjacent hex in the given direction."""
        assert hex_direction in HEX_DIRECTIONS, "hex_direction not from HEX_DIRECTIONS"
        return self + hex_direction
    
    def get_common_neighbors(self, h: Hex) -> list[Hex]:
        """Returns the common neighbors of this and another hex."""
        return [neighbor for neighbor in self.get_all_neighbors() if h.is_neighbor(neighbor)]
    
    def get_all_neighbors(self) -> list[Hex]:
        """Returns all neighboring hexagons as a list."""
        return [self.get_neighbor(direction) for direction in HEX_DIRECTIONS]
    
    def is_neighbor(self, h: Hex) -> bool:
        return self.distance_to(h) == 1
    
    def get_edges(self) -> list[Edge]:
        """Returns all adjacent edges as a list."""
        return [self.get_edge_to(h) for h in self.get_all_neighbors()]
    
    def get_edge_to(self, h: Hex) -> Edge:
        """Returns the edge to a neighboring hex."""
        assert self.is_neighbor(h), "Can only get the edge to an adjecent hex"
        return Edge(self, h)

    def get_vertices(self) -> list[Vertex]:
        """Returns all adjacent vertices as a list."""
        neighbors = self.get_all_neighbors()
        return [Vertex(self, neighbors[i], neighbors[(i + 1) % 6]) for i, _ in enumerate(neighbors)]
    
    def get_polygon_corner(self, layout: Layout, direction: int) -> Point:
        """Returns a corner position of a hex as a point."""
        center = self.to_pixel(layout)
        offset = self._corner_offset(layout, direction)
        corner = Point(center.x + offset.x, center.y + offset.y)
        return corner
    
    def get_all_polygon_corners(self, layout: Layout) -> list[tuple]:
        """Returns all corner positions of a hex. Useful for drawing."""
        return [self.get_polygon_corner(layout, i).to_tuple() for i in range(0, 6)]
    
    def to_pixel(self, layout: Layout) -> Point:
        """Returns the center position of the hex as a point."""
        M = layout.orientation
        size = layout.size
        origin = layout.origin
        x = (M.f0 * self.q + M.f1 * self.r) * size.x
        y = (M.f2 * self.q + M.f3 * self.r) * size.y
        return Point(x + origin.x, y + origin.y)    

    def draw_line_to(self, h: Hex) -> list[Hex]:
        """Returns all hexagons that are in the line from this to another hex."""
        N = self.distance_to(h)
        a_nudge = FractionalHex(self.q + 1e-06, self.r + 1e-06, self.s - 2e-06)
        b_nudge = FractionalHex(h.q + 1e-06, h.r + 1e-06, h.s - 2e-06)
        results = []
        step = 1.0 / max(N, 1)
        for i in range(0, N + 1):
            results.append(self._lerp(a_nudge.round(), b_nudge,round(), step * i))
        return results
    
    def _lerp(self, h: Hex, t: float) -> FractionalHex:
        return FractionalHex(self.q * (1.0 - t) + h.q * t, self.r * (1.0 - t) + \
                             h.r * t, self.s * (1.0 - t) + h.s * t)

    def _corner_offset(self, layout: Layout, corner: int) -> Point:
        M = layout.orientation
        size = layout.size
        angle = 2.0 * math.pi * (M.start_angle - corner) / 6.0
        return Point(size.x * math.cos(angle), size.y * math.sin(angle))


@dataclass
class FractionalHex:
    """A fractional hex may use float as positions.
    Useful for converting a the position of a point to the closest hex."""

    q: float
    r: float
    s: float

    def __post_init__(self):
        assert math.isclose(self.q + self.r + self.s, 0), "q + r + s must be 0"

    def round(self) -> Hex:
        """"Rounds to the nearest hex."""
        qi = int(round(self.q))
        ri = int(round(self.r))
        si = int(round(self.s))
        q_diff = abs(qi - self.q)
        r_diff = abs(ri - self.r)
        s_diff = abs(si - self.s)
        if q_diff > r_diff and q_diff > s_diff:
            qi = -ri - si
        else:
            if r_diff > s_diff:
                ri = -qi - si
            else:
                si = -qi - ri
        return Hex(qi, ri, si)


@dataclass
class Edge:
    """Represents the edge of two hexagons."""

    h1: Hex
    h2: Hex

    def __post_init__(self):
        assert self.h1.is_neighbor(self.h2), f"{self.h1} and {self.h2} are not neighbors"

    def __eq__(self, other: Edge) -> bool:
        for h in (self.h1, self.h2):
            if not h in (other.h1, other.h2):
                return False
        return True

    def get_vertices(self) -> tuple[Vertex, Vertex]:
        """Returns the two adjacent vertices as a tuple."""
        common_neighbors = self.h1.get_common_neighbors(self.h2)
        v = Vertex(self.h1, self.h2, common_neighbors[0])
        w = Vertex(self.h1, self.h2, common_neighbors[1])
        return (v, w)
    
    def get_neighbor_edges(self) -> list[Edge]:
        """Returns the four adjacent edges as a list."""
        edges = []
        for v in self.get_vertices():
            v_edges = v.get_edges()
            for e in v_edges:
                if e != self:
                    edges.append(e)
        return edges

    def vertices_to_pixel(self, layout) -> tuple[Point, Point]:
        """Convertes the position of the two adjacent vertices to points."""
        vertices = self.get_vertices()
        return (vertices[0].to_pixel(layout), vertices[1].to_pixel(layout))
    
    def center_to_pixel(self, layout) -> Point:
        """Returns the position of the center of the edge."""
        vertices = self.get_vertices()
        return ((vertices[0].to_pixel(layout) + vertices[1].to_pixel(layout)) / 2)
    
    @classmethod
    def from_center_pixel(self, layout: Layout, pixel: Point) -> Point:
        """Creates an edge from its approximate center position."""
        h = pixel.to_hex(layout)
        h_edges = h.get_edges()
        return min(h_edges, key=lambda e: (e.center_to_pixel(layout) - pixel).amount())
    
    def to_tuple(self) -> tuple[Hex, Hex]:
        return (self.h1, self.h2)


@dataclass
class Vertex:
    """Represents the vertex of three hexagons."""

    h1: Hex
    h2: Hex
    h3: Hex
    
    def __post_init__(self):
        t1 = self.h1.is_neighbor(self.h2)
        t2 = self.h2.is_neighbor(self.h3)
        t3 = self.h3.is_neighbor(self.h1)
        assert t1 and t2 and t3, "All hexagons must be adjacent to each other"

    def __eq__(self, other: Vertex) -> bool:
        for h in (self.h1, self.h2, self.h3):
            if not h in (other.h1, other.h2, other.h3):
                return False
        return True
    
    def __add__(self, other: Edge) -> Vertex:
        e_vertices = other.get_vertices()
        assert self in e_vertices, "Can only add an adjacent edge to a vertex"
        for vertex in e_vertices:
            if vertex != self:
                return vertex
    
    @classmethod
    def from_cube_coordinates(self, *args: Union[Sequence[int], tuple[int, int, int]]) -> Vertex:
        """Creates a vertex from cube coordinates. The sum of the coordinates must be 1 or 2."""
        if len(args) == 1:
            coord = args[0]
        elif len(args) == 3:
            coord = (args[0], args[1], args[2])

        # There is probably an easier way to do this...
        if sum(coord) == 1:
            h1 = Hex(coord[0] - 1, coord[1], coord[2])
            h2 = Hex(coord[0], coord[1], coord[2] - 1)
            h3 = Hex(coord[0], coord[1] - 1, coord[2])
        elif sum(coord) == 2:
            h1 = Hex(coord[0] - 1, coord[1] - 1, coord[2])
            h2 = Hex(coord[0], coord[1] - 1, coord[2] - 1)
            h3 = Hex(coord[0] - 1, coord[1], coord[2] - 1)

        return Vertex(h1, h2, h3)

    def get_edges(self) -> tuple[Edge, Edge, Edge]:
        """Returns the three adjacent edges."""
        hexagons = self.to_tuple()
        return tuple(hexagons[i].get_edge_to(hexagons[(i + 1) % 3]) for i in range(0, 3))
    
    def get_neighbor_vertices(self) -> tuple[Vertex, Vertex, Vertex]:
        """Returns the closest three vertices."""
        edges = self.get_edges()
        return tuple(self + e for e in edges)

    def distance_to(self, v: Vertex) -> int:
        """Calculate the distance to another vertex."""
        coord = self.to_cube_coordinates()
        v_coord = v.to_cube_coordinates()
        return abs(coord[0] - v_coord[0]) + abs(coord[1] + v_coord[1]) + abs(coord[2] + v_coord[2]) 

    def to_cube_coordinates(self) -> tuple[int, int, int]:
        '''Converting the position to a tuple of cube coordinates.
        Uses unit vectors q, r, s so that in the pointy top orientation,
        r points to the vertex at NE, q to S and s to NW. It's basically
        the same coordinate system that's used for the hexagons.'''
        self._sort_hexagons()
        if self.h3.q == self.h2.q:
            return (self.h1.q + 1, self.h1.r, self.h1.s)
        elif self.h3.q == self.h1.q:
            return (self.h1.q + 1, self.h1.r + 1, self.h1.s)

    def to_tuple(self) -> tuple[Hex, Hex, Hex]:
        return (self.h1, self.h2, self.h3)

    def to_pixel(self, layout) -> Point:
        """Returns the position of the vertex as a point."""
        x = 0
        y = 0
        for h in self.to_tuple():
            x += h.to_pixel(layout).x
            y += h.to_pixel(layout).y
        return Point(x / 3, y / 3)
    
    @classmethod
    def from_pixel(cls, layout: Layout, pixel: Point) -> Vertex:
        """Creates a vertex from its approximate position."""
        h = pixel.to_hex(layout)
        h_vertices = h.get_vertices()
        return min(h_vertices, key=lambda v: (v.to_pixel(layout) - pixel).amount())
    
    def _sort_hexagons(self):
        """Sort the hexagons in order h1, h2, h3, so that h1 + E = h2."""
        hexagons = self.to_tuple()
        h1 = min(hexagons, key=lambda hex: (hex.q, hex.r))
        h2 = h1 + E
        for h in hexagons:
            if h != h1 and h != h2:
                h3 = h
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3


# Directions in pointy top orientation
E = Hex(1, 0, -1)
NE = Hex(1, -1, 0)
NW = Hex(0, -1, 1)
W = Hex(-1, 0, 1)
SW = Hex(-1, 1, 0)
SE = Hex(0, 1, -1)
HEX_DIRECTIONS = [E, NE, NW, W, SW, SE]
