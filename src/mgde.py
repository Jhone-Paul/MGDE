import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import random
from abc import ABC, abstractmethod


class Problem(ABC):
    """Abstract base class for multi-objective optimization problems."""

    def __init__(self, num_variables: int, num_objectives: int,
                 lower_bounds: List[float], upper_bounds: List[float]):
        """
        Initialize the problem.

        Args:
            num_variables: Number of decision variables
            num_objectives: Number of objective functions
            lower_bounds: Lower bounds for decision variables
            upper_bounds: Upper bounds for decision variables
        """
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the objective functions for a solution.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of objective function values
        """
        pass

    def is_feasible(self, solution: np.ndarray) -> bool:
        """
        Check if a solution is within the feasible region.

        Args:
            solution: Decision variables vector

        Returns:
            True if the solution is feasible, False otherwise
        """
        for i in range(self.num_variables):
            if solution[i] < self.lower_bounds[i] or solution[i] > self.upper_bounds[i]:
                return False
        return True

    def generate_random_solution(self) -> np.ndarray:
        """
        Generate a random solution within the feasible region.

        Returns:
            Random solution vector
        """
        solution = np.zeros(self.num_variables)
        for i in range(self.num_variables):
            solution[i] = self.lower_bounds[i] + np.random.random() * (self.upper_bounds[i] - self.lower_bounds[i])
        return solution


#########################
# ZDT Family Problems
#########################

class ZDT1(Problem):
    """ZDT1 benchmark problem (Separable, Unimodal, Convex)."""

    def __init__(self, num_variables: int = 30):
        """
        Initialize the ZDT1 problem.

        Args:
            num_variables: Number of decision variables (default: 30)
        """
        super().__init__(num_variables, 2, [0.0] * num_variables, [1.0] * num_variables)

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the ZDT1 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of two objective function values [f1, f2]
        """
        f1 = solution[0]
        g = 1.0 + 9.0 * np.sum(solution[1:]) / (self.num_variables - 1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h

        return np.array([f1, f2])


class ZDT2(Problem):
    """ZDT2 benchmark problem (Separable, Unimodal, Concave)."""

    def __init__(self, num_variables: int = 30):
        """
        Initialize the ZDT2 problem.

        Args:
            num_variables: Number of decision variables (default: 30)
        """
        super().__init__(num_variables, 2, [0.0] * num_variables, [1.0] * num_variables)

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the ZDT2 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of two objective function values [f1, f2]
        """
        f1 = solution[0]
        g = 1.0 + 9.0 * np.sum(solution[1:]) / (self.num_variables - 1)
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h

        return np.array([f1, f2])


class ZDT3(Problem):
    """ZDT3 benchmark problem (Separable, Unimodal/multimodal, Disconnected)."""

    def __init__(self, num_variables: int = 30):
        """
        Initialize the ZDT3 problem.

        Args:
            num_variables: Number of decision variables (default: 30)
        """
        super().__init__(num_variables, 2, [0.0] * num_variables, [1.0] * num_variables)

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the ZDT3 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of two objective function values [f1, f2]
        """
        f1 = solution[0]
        g = 1.0 + 9.0 * np.sum(solution[1:]) / (self.num_variables - 1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h

        return np.array([f1, f2])


class ZDT4(Problem):
    """ZDT4 benchmark problem (Separable, Unimodal/multimodal, Convex)."""

    def __init__(self, num_variables: int = 10):
        """
        Initialize the ZDT4 problem.

        Args:
            num_variables: Number of decision variables (default: 10)
        """
        lower_bounds = [0.0] + [-5.0] * (num_variables - 1)
        upper_bounds = [1.0] + [5.0] * (num_variables - 1)
        super().__init__(num_variables, 2, lower_bounds, upper_bounds)

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the ZDT4 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of two objective function values [f1, f2]
        """
        f1 = solution[0]

        g = 1.0 + 10.0 * (self.num_variables - 1)
        for i in range(1, self.num_variables):
            g += solution[i] ** 2 - 10.0 * np.cos(4.0 * np.pi * solution[i])

        h = 1.0 - np.sqrt(f1 / g)

        f2 = g * h

        return np.array([f1, f2])




class ZDT6(Problem):
    """ZDT6 benchmark problem (Separable, Multimodal, Concave)."""

    def __init__(self, num_variables: int = 10):
        """
        Initialize the ZDT6 problem.

        Args:
            num_variables: Number of decision variables (default: 10)
        """
        super().__init__(num_variables, 2, [0.0] * num_variables, [1.0] * num_variables)

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the ZDT6 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of two objective function values [f1, f2]
        """
        f1 = 1.0 - np.exp(-4.0 * solution[0]) * (np.sin(6.0 * np.pi * solution[0])) ** 6

        g = 1.0 + 9.0 * ((np.sum(solution[1:]) / (self.num_variables - 1)) ** 0.25)
        h = 1.0 - (f1 / g) ** 2

        f2 = g * h

        return np.array([f1, f2])


#########################
# WFG Family Problems
#########################
import numpy as np


class WFGProblem(Problem):
    """Base class for WFG benchmark problems."""

    def __init__(self, num_objectives: int = 2, k: int = 4, l: int = 20):
        """
        Initialize a WFG problem.

        Args:
            num_objectives: Number of objectives (default: 2)
            k: Number of position parameters (default: 4)
            l: Number of distance parameters (default: 20)
        """
        self.k = k
        self.l = l
        num_variables = k + l

        lower_bounds = [0.0] * num_variables
        upper_bounds = [2.0 * (i + 1) for i in range(num_variables)]

        super().__init__(num_variables, num_objectives, lower_bounds, upper_bounds)

        self.M = num_objectives
        self.S = np.array([2.0 * i for i in range(1, self.M + 1)])

    def _normalize_z(self, z: np.ndarray) -> np.ndarray:
        """
        Normalize decision variables to [0,1].

        Args:
            z: Decision variables vector

        Returns:
            Normalized variables
        """
        normalized = np.zeros(self.num_variables)
        for i in range(self.num_variables):
            normalized[i] = z[i] / (2.0 * (i + 1))
        return normalized

    def _calculate_x(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate x vector from t vector.

        Args:
            t: Transition vector

        Returns:
            x vector
        """
        x = np.zeros(self.M)
        for i in range(self.M):
            x[i] = t[i] + 1.0
        return x

    def _s_linear(self, y: float, A: float) -> float:
        """
        Linear shift function.

        Args:
            y: Value to shift
            A: Parameter

        Returns:
            Shifted value
        """
        return abs(y - A) / abs(np.floor(A - y) + A)

    def _s_decept(self, y: float, A: float, B: float, C: float) -> float:
        """
        Deceptive shift function.

        Args:
            y: Value to shift
            A, B, C: Parameters

        Returns:
            Shifted value
        """
        return 1.0 + (abs(y - A) - B) * (
                    np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B) + np.floor(A + B - y) * (
                        1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B) + 1.0 / B)

    def _s_multi(self, y: float, A: float, B: float, C: float) -> float:
        """
        Multi-modal shift function.

        Args:
            y: Value to shift
            A, B, C: Parameters

        Returns:
            Shifted value
        """
        return (1.0 + np.cos((4.0 * A + 2.0) * np.pi * (0.5 - abs(y - C) / (2.0 * (np.floor(C - y) + C)))) + 4.0 * B * (
                    abs(y - C) / (2.0 * (np.floor(C - y) + C))) ** 2) / (B + 2.0)

    def _b_flat(self, y: float, A: float, B: float, C: float) -> float:
        """
        Flat bias function.

        Args:
            y: Value to bias
            A, B, C: Parameters

        Returns:
            Biased value
        """
        return A + min(0.0, np.floor(y - B)) * A * (B - y) / B - min(0.0, np.floor(C - y)) * (1.0 - A) * (y - C) / (
                    1.0 - C)

    def _b_poly(self, y: float, alpha: float) -> float:
        """
        Polynomial bias function.

        Args:
            y: Value to bias
            alpha: Parameter

        Returns:
            Biased value
        """
        return y ** alpha

    def _r_sum(self, y: np.ndarray, w: np.ndarray) -> float:
        """
        Weighted sum reduction function.

        Args:
            y: Vector to reduce
            w: Weight vector

        Returns:
            Reduced value
        """
        return np.sum(y * w) / np.sum(w)

    def _r_nonsep(self, y: np.ndarray, A: int) -> float:
        """
        Non-separable reduction function.

        Args:
            y: Vector to reduce
            A: Parameter

        Returns:
            Reduced value
        """
        n = len(y)
        result = 0.0

        for j in range(n):
            result += y[j]
            for k in range(0, A - 1):
                result += abs(y[j] - y[(j + k + 1) % n])

        return result / (n * A)


class WFG1(WFGProblem):
    """WFG1 benchmark problem (Separable, Unimodal, Convex, mixed)."""

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the WFG1 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of objective function values
        """
        z = self._normalize_z(solution)

        t1 = np.zeros(self.num_variables)

        for i in range(self.k):
            t1[i] = z[i]

        for i in range(self.k, self.num_variables):
            t1[i] = self._s_linear(z[i], 0.35)

        t2 = np.zeros(self.num_variables)

        for i in range(self.k):
            t2[i] = t1[i]

        for i in range(self.k, self.num_variables):
            t2[i] = self._b_flat(t1[i], 0.8, 0.75, 0.85)

        t3 = np.zeros(self.num_variables)

        for i in range(self.num_variables):
            t3[i] = self._b_poly(t2[i], 0.02)

        t4 = np.zeros(self.M)

        for i in range(self.M - 1):
            start_idx = i * self.k // (self.M - 1)
            end_idx = (i + 1) * self.k // (self.M - 1)

            weights = np.ones(end_idx - start_idx)

            sub_t3 = t3[start_idx:end_idx]

            t4[i] = self._r_sum(sub_t3, weights)

        distance_weights = np.ones(self.num_variables - self.k)
        t4[self.M - 1] = self._r_sum(t3[self.k:], distance_weights)

        h = np.zeros(self.M)

        for i in range(1, self.M):
            h[i - 1] = self._convex(t4, i)

        h[self.M - 1] = self._mixed(t4)

        x = self._calculate_x(t4)
        objectives = np.zeros(self.M)

        for i in range(self.M):
            objectives[i] = x[self.M - 1] * h[i]

        objectives = objectives * self.S

        return objectives

    def _convex(self, t: np.ndarray, m: int) -> float:
        """
        Convex shape function.
        """
        if m == 1:
            return 1.0 - np.cos(t[0] * np.pi / 2.0)
        else:
            return (1.0 - np.cos(t[m - 1] * np.pi / 2.0)) * np.prod(
                [1.0 - np.sin(t[i] * np.pi / 2.0) for i in range(m - 1)])

    def _mixed(self, t: np.ndarray) -> float:
        """
        Mixed shape function.
        """
        return 1.0 - t[0] - np.cos(10.0 * np.pi * t[0] + np.pi / 2.0) / (10.0 * np.pi)


class WFG2(WFGProblem):
    """WFG2 benchmark problem (Non-separable, Unimodal/multimodal, Convex, disconnected)."""

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the WFG2 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of objective function values
        """
        z = self._normalize_z(solution)

        t1 = np.zeros(self.num_variables)

        for i in range(self.k):
            t1[i] = z[i]

        for i in range(self.k, self.num_variables):
            t1[i] = self._s_linear(z[i], 0.35)

        t2_size = self.k + self.l // 2
        t2 = np.zeros(t2_size)

        for i in range(self.k):
            t2[i] = t1[i]

        for i in range(self.l // 2):
            idx = self.k + 2 * i
            if idx + 1 < self.num_variables:
                t2[self.k + i] = self._r_nonsep(np.array([t1[idx], t1[idx + 1]]), 2)

        t3 = np.zeros(self.M)

        for i in range(self.M - 1):
            start_idx = i * self.k // (self.M - 1)
            end_idx = (i + 1) * self.k // (self.M - 1)

            weights = np.ones(end_idx - start_idx)

            sub_t2 = t2[start_idx:end_idx]

            t3[i] = self._r_sum(sub_t2, weights)

        distance_weights = np.ones(self.l // 2)
        t3[self.M - 1] = self._r_sum(t2[self.k:], distance_weights)

        h = np.zeros(self.M)

        for i in range(1, self.M):
            h[i - 1] = self._convex(t3, i)

        h[self.M - 1] = self._disconnected(t3)

        x = self._calculate_x(t3)
        objectives = np.zeros(self.M)

        for i in range(self.M):
            objectives[i] = x[self.M - 1] * h[i]

        objectives = objectives * self.S

        return objectives

    def _convex(self, t: np.ndarray, m: int) -> float:
        """
        Convex shape function.
        """
        if m == 1:
            return 1.0 - np.cos(t[0] * np.pi / 2.0)
        else:
            return (1.0 - np.cos(t[m - 1] * np.pi / 2.0)) * np.prod(
                [1.0 - np.sin(t[i] * np.pi / 2.0) for i in range(m - 1)])

    def _disconnected(self, t: np.ndarray) -> float:
        """
        Disconnected shape function.
        """
        alpha = 1.0
        beta = 1.0
        A = 5.0

        return 1.0 - (t[0] ** alpha) * np.cos(A * np.pi * t[0] ** beta) ** 2


class WFG3(WFGProblem):
    """WFG3 benchmark problem (Non-separable, Unimodal, Linear, degenerate)."""

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the WFG3 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of objective function values
        """
        z = self._normalize_z(solution)

        t1 = np.zeros(self.num_variables)

        for i in range(self.k):
            t1[i] = z[i]

        for i in range(self.k, self.num_variables):
            t1[i] = self._s_linear(z[i], 0.35)

        t2_size = self.k + self.l // 2
        t2 = np.zeros(t2_size)

        for i in range(self.k):
            t2[i] = t1[i]

        for i in range(self.l // 2):
            idx = self.k + 2 * i
            if idx + 1 < self.num_variables:
                t2[self.k + i] = self._r_nonsep(np.array([t1[idx], t1[idx + 1]]), 2)

        t3 = np.zeros(self.M)

        for i in range(self.M - 1):
            start_idx = i * self.k // (self.M - 1)
            end_idx = (i + 1) * self.k // (self.M - 1)

            weights = np.ones(end_idx - start_idx)

            sub_t2 = t2[start_idx:end_idx]

            t3[i] = self._r_sum(sub_t2, weights)

        distance_weights = np.ones(self.l // 2)
        t3[self.M - 1] = self._r_sum(t2[self.k:], distance_weights)

        h = np.zeros(self.M)

        for i in range(self.M):
            h[i] = self._linear(t3, i + 1)

        x = self._calculate_x(t3)
        objectives = np.zeros(self.M)

        for i in range(self.M):
            objectives[i] = x[self.M - 1] * h[i]

        objectives = objectives * self.S

        return objectives

    def _linear(self, t: np.ndarray, m: int) -> float:
        """
        Linear shape function.
        """
        if m == 1:
            return t[0]
        else:
            return np.prod([t[i] for i in range(m - 1)]) * (1.0 - t[m - 1])


class WFG4(WFGProblem):
    """WFG4 benchmark problem (Separable, Multimodal, Concave)."""

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the WFG4 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of objective function values
        """
        z = self._normalize_z(solution)

        t1 = np.zeros(self.num_variables)

        for i in range(self.num_variables):
            t1[i] = self._s_multi(z[i], 30.0, 10.0, 0.35)

        t2 = np.zeros(self.M)

        for i in range(self.M - 1):
            start_idx = i * self.k // (self.M - 1)
            end_idx = (i + 1) * self.k // (self.M - 1)

            weights = np.ones(end_idx - start_idx)

            sub_t1 = t1[start_idx:end_idx]

            t2[i] = self._r_sum(sub_t1, weights)

        distance_weights = np.ones(self.num_variables - self.k)
        t2[self.M - 1] = self._r_sum(t1[self.k:], distance_weights)

        h = np.zeros(self.M)

        for i in range(self.M):
            h[i] = self._concave(t2, i + 1)

        x = self._calculate_x(t2)
        objectives = np.zeros(self.M)

        for i in range(self.M):
            objectives[i] = x[self.M - 1] * h[i]

        objectives = objectives * self.S

        return objectives

    def _concave(self, t: np.ndarray, m: int) -> float:
        """
        Concave shape function.
        """
        if m == 1:
            return np.sin(t[0] * np.pi / 2.0)
        else:
            return np.sin(t[m - 1] * np.pi / 2.0) * np.prod(
                [np.cos(t[i] * np.pi / 2.0) for i in range(m - 1)])


class WFG5(WFGProblem):
    """WFG5 benchmark problem (Separable, Multimodal, Concave)."""

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate the WFG5 objective functions.

        Args:
            solution: Decision variables vector

        Returns:
            Vector of objective function values
        """
        z = self._normalize_z(solution)

        t1 = np.zeros(self.num_variables)

        for i in range(self.num_variables):
            t1[i] = self._s_decept(z[i], 0.35, 0.001, 0.05)

        t2 = np.zeros(self.M)

        for i in range(self.M - 1):
            start_idx = i * self.k // (self.M - 1)
            end_idx = (i + 1) * self.k // (self.M - 1)

            weights = np.ones(end_idx - start_idx)

            sub_t1 = t1[start_idx:end_idx]

            t2[i] = self._r_sum(sub_t1, weights)

        distance_weights = np.ones(self.num_variables - self.k)
        t2[self.M - 1] = self._r_sum(t1[self.k:], distance_weights)

        h = np.zeros(self.M)

        for i in range(self.M):
            h[i] = self._concave(t2, i + 1)

        x = self._calculate_x(t2)
        objectives = np.zeros(self.M)

        for i in range(self.M):
            objectives[i] = x[self.M - 1] * h[i]

        objectives = objectives * self.S

        return objectives

    def _concave(self, t: np.ndarray, m: int) -> float:
        """
        Concave shape function.
        """
        if m == 1:
            return np.sin(t[0] * np.pi / 2.0)
        else:
            return np.sin(t[m - 1] * np.pi / 2.0) * np.prod(
                [np.cos(t[i] * np.pi / 2.0) for i in range(m - 1)])
class Solution:
    """Class representing a solution in the optimization process."""

    def __init__(self, variables: np.ndarray, objectives: np.ndarray = None):
        """
        Initialize a solution.

        Args:
            variables: Decision variables vector
            objectives: Objective function values vector (optional)
        """
        self.variables = variables
        self.objectives = objectives
        self.rank = 0
        self.crowding_distance = 0.0

    def dominates(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another solution.

        Args:
            other: Another solution to compare with

        Returns:
            True if this solution dominates the other, False otherwise
        """
        at_least_one_better = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            elif self.objectives[i] < other.objectives[i]:
                at_least_one_better = True
        return at_least_one_better


class Archive:
    """Archive of non-dominated solutions."""

    def __init__(self, max_size: int = 100):
        """
        Initialize the archive.

        Args:
            max_size: Maximum archive size (default: 100)
        """
        self.solutions = []
        self.max_size = max_size

    def add(self, solution: Solution) -> bool:
        """
        Add a solution to the archive if it is non-dominated.

        Args:
            solution: Solution to add

        Returns:
            True if the solution was added, False otherwise
        """
        for archived_solution in self.solutions:
            if archived_solution.dominates(solution):
                return False

        self.solutions = [s for s in self.solutions if not solution.dominates(s)]

        self.solutions.append(solution)

        if len(self.solutions) > self.max_size:
            self._update_crowding_distances()
            self.solutions.sort(key=lambda s: s.crowding_distance)
            self.solutions = self.solutions[1:]

        return True

    def _update_crowding_distances(self):
        """Update crowding distances for all solutions in the archive."""
        n = len(self.solutions)
        if n <= 2:
            for solution in self.solutions:
                solution.crowding_distance = float('inf')
            return

        for solution in self.solutions:
            solution.crowding_distance = 0.0

        m = len(self.solutions[0].objectives)

        for obj_index in range(m):
            self.solutions.sort(key=lambda s: s.objectives[obj_index])

            self.solutions[0].crowding_distance = float('inf')
            self.solutions[-1].crowding_distance = float('inf')

            obj_range = self.solutions[-1].objectives[obj_index] - self.solutions[0].objectives[obj_index]
            if obj_range > 0:
                for i in range(1, n - 1):
                    self.solutions[i].crowding_distance += (
                                                                   self.solutions[i + 1].objectives[obj_index] -
                                                                   self.solutions[i - 1].objectives[obj_index]
                                                           ) / obj_range

    def get_random_solution(self) -> Solution:
        """
        Get a random solution from the archive.

        Returns:
            Random solution from the archive
        """
        if not self.solutions:
            return None
        return random.choice(self.solutions)

    def tournament_selection(self, tournament_size: int = 2) -> Solution:
        """
        Select a solution from the archive using tournament selection.

        Args:
            tournament_size: Number of solutions in the tournament (default: 2)

        Returns:
            Selected solution
        """
        if not self.solutions:
            return None

        self._update_crowding_distances()

        candidates = random.sample(self.solutions, min(tournament_size, len(self.solutions)))
        candidates.sort(key=lambda s: s.crowding_distance, reverse=True)

        return candidates[0]


class SubPopulation:
    """Sub-population optimizing a single objective function."""

    def __init__(self, problem: Problem, objective_index: int, population_size: int, archive: Archive):
        """
        Initialize the sub-population.

        Args:
            problem: Optimization problem
            objective_index: Index of the objective function to optimize
            population_size: Size of the sub-population
            archive: Archive of non-dominated solutions
        """
        self.problem = problem
        self.objective_index = objective_index
        self.population_size = population_size
        self.archive = archive

        self.population = []
        for _ in range(population_size):
            variables = problem.generate_random_solution()
            objectives = problem.evaluate(variables)
            self.population.append(Solution(variables, objectives))

    def evolve(self, F: float, CR: float, use_tournament: bool = False):
        """
        Evolve the sub-population for one generation.

        Args:
            F: Differential weight (scaling factor)
            CR: Crossover probability
            use_tournament: Whether to use tournament selection for archive guide (default: False)
        """
        new_population = []

        for i in range(self.population_size):
            target = self.population[i]

            indices = list(range(self.population_size))
            indices.remove(i)
            random.shuffle(indices)
            r1, r2 = indices[0], indices[1]

            if use_tournament:
                archive_guide = self.archive.tournament_selection()
            else:
                archive_guide = self.archive.get_random_solution()

            if archive_guide is None:
                archive_indices = list(range(self.population_size))
                archive_indices.remove(i)
                if r1 in archive_indices:
                    archive_indices.remove(r1)
                if r2 in archive_indices:
                    archive_indices.remove(r2)
                if archive_indices:
                    archive_guide_idx = random.choice(archive_indices)
                    archive_guide = self.population[archive_guide_idx]
                else:
                    archive_guide = self.population[r1]

            mutation_vector = (
                    self.population[r1].variables +
                    F * (self.population[r2].variables - target.variables) +
                    F * (archive_guide.variables - target.variables)
            )

            for j in range(self.problem.num_variables):
                mutation_vector[j] = max(self.problem.lower_bounds[j],
                                         min(self.problem.upper_bounds[j], mutation_vector[j]))

            trial_vector = np.zeros(self.problem.num_variables)

            for j in range(self.problem.num_variables):
                if random.random() < CR or j == random.randint(0, self.problem.num_variables - 1):
                    if random.random() < 0.5:
                        trial_vector[j] = mutation_vector[j]
                    else:
                        trial_vector[j] = archive_guide.variables[j]
                else:
                    trial_vector[j] = target.variables[j]

            trial_objectives = self.problem.evaluate(trial_vector)
            trial_solution = Solution(trial_vector, trial_objectives)

            if trial_objectives[self.objective_index] <= target.objectives[self.objective_index]:
                new_population.append(trial_solution)
                self.archive.add(trial_solution)
            else:
                new_population.append(target)

        self.population = new_population


class MGDE:
    """Multi-Guide Differential Evolution algorithm."""

    def __init__(self, problem: Problem, population_size: int = 100, archive_size: int = 100):
        """
        Initialize the MGDE algorithm.

        Args:
            problem: Multi-objective optimization problem
            population_size: Size of each sub-population (default: 100)
            archive_size: Size of the archive (default: 100)
        """
        self.problem = problem
        self.population_size = population_size

        self.archive = Archive(archive_size)

        self.sub_populations = []
        for i in range(problem.num_objectives):
            self.sub_populations.append(SubPopulation(problem, i, population_size, self.archive))

        for sub_pop in self.sub_populations:
            for solution in sub_pop.population:
                self.archive.add(solution)

    def run(self, max_generations: int, F: float = 0.5, CR: float = 0.9,
            use_tournament: bool = False) -> Tuple[List[Solution], List[float]]:
        """
        Run the MGDE algorithm.

        Args:
            max_generations: Maximum number of generations
            F: Differential weight (scaling factor)
            CR: Crossover probability
            use_tournament: Whether to use tournament selection for archive guide

        Returns:
            Tuple of (final archive solutions, hypervolume history)
        """
        hypervolume_history = []

        for generation in range(max_generations):
            for sub_pop in self.sub_populations:
                sub_pop.evolve(F, CR, use_tournament)

            if generation % 10 == 0:
                hv = self._calculate_hypervolume()
                hypervolume_history.append(hv)
                print(f"Generation {generation}, Hypervolume: {hv:.4f}, Archive size: {len(self.archive.solutions)}")

        return self.archive.solutions, hypervolume_history

    def _calculate_hypervolume(self, reference_point=None) -> float:
        """
        Calculate the hypervolume indicator (approximated).

        Args:
            reference_point: Reference point for hypervolume calculation. If None, a default is used.

        Returns:
            Hypervolume value
        """
        if not self.archive.solutions:
            return 0.0

        if reference_point is None:
            max_obj = np.max([solution.objectives for solution in self.archive.solutions], axis=0)
            reference_point = max_obj + 0.1 * max_obj

        if self.problem.num_objectives == 2:
            points = np.array([solution.objectives for solution in self.archive.solutions])
            points = points[points[:, 0].argsort()]

            area = 0.0
            prev_point = np.array([0.0, reference_point[1]])

            for point in points:
                area += (point[0] - prev_point[0]) * (reference_point[1] - point[1])
                prev_point = point

            area += (reference_point[0] - prev_point[0]) * (reference_point[1] - prev_point[1])

            return area
        else:
            points = np.array([solution.objectives for solution in self.archive.solutions])

            dominated_space = 0.0
            for point in points:
                vol = np.prod(reference_point - point)
                dominated_space += vol

            return dominated_space

    def plot_pareto_front(self, show_analytical: bool = True):
        """
        Plot the Pareto front approximation.

        Args:
            show_analytical: Whether to show the analytical Pareto front for ZDT1
        """
        if not self.archive.solutions:
            print("Archive is empty, nothing to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        points = np.array([solution.objectives for solution in self.archive.solutions])
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=30, label='MGDE Solutions')

        if show_analytical and isinstance(self.problem, ZDT1):
            f1 = np.linspace(0, 1, 100)
            f2 = 1 - np.sqrt(f1)
            ax.plot(f1, f2, 'r-', label='Analytical Pareto Front')

        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_title('Pareto Front Approximation')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_hypervolume_hist(hypervolume_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, 100, 10), hypervolume_history)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume Convergence')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    zdt1 = ZDT1(num_variables=30)

    mgde = MGDE(zdt1, population_size=50, archive_size=100)

    final_archive, hypervolume_history = mgde.run(max_generations=100,
                                                  F=0.5, CR=0.9,
                                                  use_tournament=True)

    mgde.plot_pareto_front()
    plot_hypervolume_hist(hypervolume_history)
