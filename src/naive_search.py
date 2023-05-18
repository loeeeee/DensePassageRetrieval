import numpy as np
import math

class NaiveSearch:
    def __init__(self, vectors, distance_method: str = "euclidean") -> None:
        """
        Distance method can be one of the following:
        "angular", "euclidean", "manhattan", "hamming", or "dot"
        """
        self.vectors = vectors
        if distance_method == "euclidean":
            self.distance_method = self._euclidean
        elif distance_method == "angular":
            self.distance_method = self._angular
        elif distance_method == "manhatten":
            self.distance_method = self._manhatten
        elif distance_method == "hamming":
            self.distance_method = self._hamming
        elif distance_method == "dot":
            self.distance_method = self._dot

    def search(self, query, top_n: int = 20) -> list:
        distances = []
        for vector in self.vectors:
            distances.append((self.distance_method(query, vector), vector))
        distances = sorted(distances, key= lambda x: x[0])
        return distances[1][:top_n]

    def _euclidean(self, point1, point2) -> float:
        # Convert points to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.linalg.norm(point2 - point1)

    def _angular(self, point1, point2) -> float:
        # Convert points to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Normalize the points
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        norm_point1 = point1 / norm1
        norm_point2 = point2 / norm2

        # Calculate the dot product
        dot_product = np.dot(norm_point1, norm_point2)

        # Calculate the angle between the points using arccosine
        angle = np.arccos(dot_product)

        # Convert angle from radians to degrees
        angle_deg = np.degrees(angle)

        return angle_deg
            
    def _manhatten(self, point1, point2) -> float:
        distance = sum(abs(x - y) for x, y in zip(point1, point2))
        return distance

    def _hamming(self, point1, point2) -> float:
        distance = sum(x != y for x, y in zip(point1, point2))
        return distance

    def _dot(self, point1, point2) -> float:

        point1 = np.array(point1)
        point2 = np.array(point2)

        distance = np.dot(point1, point2)
        return distance
