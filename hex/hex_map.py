from hex import Hex, Edge, Vertex
from typing import Optional

class HexMap:

    def __init__(self) -> None:
        self.hexes: list[Hex] = []

    def get_hex_at(self, h: Hex) -> Optional[Hex]:
        if h in self.hexes:
            return self.hexes[self.hexes.index(h)]
        else:
            return None
    
    def get_edge_at(self, e: Edge) -> Optional[Edge]:
        if e in self.edges:
            return self.edges[self.edges.index(e)]
        else:
            return None
    
    def get_vertex_at(self, v: Vertex) -> Optional[Vertex]:
        if v in self.vertices:
            return self.vertices[self.vertices.index(v)]
        else:
            return None

    def create_edges(self) -> None:
        self.edges: list[Edge] = []
        for h in self.hexes:
            for e in h.get_edges():
                if e not in self.edges:
                    self.edges.append(e)

    def create_vertices(self) -> None:
        self.vertices: list[Vertex] = []
        for h in self.hexes:
            for v in h.get_vertices():
                if v not in self.vertices:
                    self.vertices.append(v)

    def create_map_hexagon(self, size: int) -> None:
        for q in range(-size, size + 1):
            r1 = max(-size, -q - size)
            r2 = min(+size, -q + size)
            for r in range(r1, r2 + 1):
                self.hexes.append(Hex(q, r, -q - r))
