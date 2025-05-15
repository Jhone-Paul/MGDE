from mgde import *

class Particle:
    """Class representing a particle in the PSO algorithm."""

    def __init__(self, problem: Problem, objective_index: int = None):
        """
        Initialize a particle.

        Args:
            problem: Optimization problem
            objective_index: Index of the objective function to optimize (or None for all)
        """
        self.problem = problem
        self.objective_index = objective_index

        self.position = problem.generate_random_solution()
        self.velocity = np.zeros(problem.num_variables)

        self.objectives = problem.evaluate(self.position)

        self.best_position = self.position.copy()
        self.best_objectives = self.objectives.copy()

        self.solution = Solution(self.position, self.objectives)
        self.best_solution = Solution(self.best_position, self.best_objectives)

    def update_velocity(self, global_best: np.ndarray, archive_guide: np.ndarray,
                        w: float, c1: float, c2: float, c3: float):
        """
        Update the particle's velocity.

        Args:
            global_best: Global best position
            archive_guide: Position from the archive
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            c3: Archive guide coefficient
        """
        r1 = np.random.random(self.problem.num_variables)
        r2 = np.random.random(self.problem.num_variables)
        r3 = np.random.random(self.problem.num_variables)

        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best - self.position) +
                         c3 * r3 * (archive_guide - self.position))

    def update_position(self):
        """Update the particle's position based on its velocity."""
        self.position = self.position + self.velocity

        for i in range(self.problem.num_variables):
            self.position[i] = max(self.problem.lower_bounds[i],
                                   min(self.problem.upper_bounds[i], self.position[i]))

        self.objectives = self.problem.evaluate(self.position)
        self.solution = Solution(self.position, self.objectives)

        self._update_personal_best()

    def _update_personal_best(self):
        """Update the particle's personal best if current position is better."""
        if self.objective_index is not None:
            if self.objectives[self.objective_index] <= self.best_objectives[self.objective_index]:
                self.best_position = self.position.copy()
                self.best_objectives = self.objectives.copy()
                self.best_solution = Solution(self.best_position, self.best_objectives)
        else:
            current_solution = Solution(self.position, self.objectives)
            best_solution = Solution(self.best_position, self.best_objectives)

            if current_solution.dominates(best_solution) or not best_solution.dominates(current_solution):
                self.best_position = self.position.copy()
                self.best_objectives = self.objectives.copy()
                self.best_solution = Solution(self.best_position, self.best_objectives)


class SubSwarm:
    """Sub-swarm optimizing a single objective function."""

    def __init__(self, problem: Problem, objective_index: int, swarm_size: int, archive: Archive):
        """
        Initialize the sub-swarm.

        Args:
            problem: Optimization problem
            objective_index: Index of the objective function to optimize
            swarm_size: Size of the sub-swarm
            archive: Archive of non-dominated solutions
        """
        self.problem = problem
        self.objective_index = objective_index
        self.swarm_size = swarm_size
        self.archive = archive

        self.particles = []
        for _ in range(swarm_size):
            particle = Particle(problem, objective_index)
            self.particles.append(particle)

            self.archive.add(particle.solution)

        self.global_best_position = None
        self.global_best_objective = float('inf')
        self._update_global_best()

    def _update_global_best(self):
        """Update the global best position based on the specific objective."""
        for particle in self.particles:
            if particle.best_objectives[self.objective_index] < self.global_best_objective:
                self.global_best_objective = particle.best_objectives[self.objective_index]
                self.global_best_position = particle.best_position.copy()

    def evolve(self, w: float, c1: float, c2: float, c3: float, use_tournament: bool = False):
        """
        Evolve the sub-swarm for one iteration.

        Args:
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            c3: Archive guide coefficient
            use_tournament: Whether to use tournament selection for archive guide
        """
        for particle in self.particles:
            if use_tournament:
                archive_guide_solution = self.archive.tournament_selection()
            else:
                archive_guide_solution = self.archive.get_random_solution()

            if archive_guide_solution is None:
                random_particle = random.choice(self.particles)
                archive_guide = random_particle.best_position
            else:
                archive_guide = archive_guide_solution.variables

            particle.update_velocity(self.global_best_position, archive_guide, w, c1, c2, c3)
            particle.update_position()
            self.archive.add(particle.solution)

        self._update_global_best()


class MGPSO:
    """Multi-Guide Particle Swarm Optimization algorithm."""

    def __init__(self, problem: Problem, swarm_size: int = 50, archive_size: int = 100):
        """
        Initialize the MGPSO algorithm.

        Args:
            problem: Multi-objective optimization problem
            swarm_size: Size of each sub-swarm (default: 50)
            archive_size: Size of the archive (default: 100)
        """
        self.problem = problem
        self.swarm_size = swarm_size

        self.archive = Archive(archive_size)

        self.sub_swarms = []
        for i in range(problem.num_objectives):
            self.sub_swarms.append(SubSwarm(problem, i, swarm_size, self.archive))

    def run(self, max_iterations: int, w_start: float = 0.9, w_end: float = 0.4,
            c1: float = 2.0, c2: float = 2.0, c3: float = 1.5,
            use_tournament: bool = True) -> Tuple[List[Solution], List[float]]:
        """
        Run the MGPSO algorithm.

        Args:
            max_iterations: Maximum number of iterations
            w_start: Initial inertia weight
            w_end: Final inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            c3: Archive guide coefficient
            use_tournament: Whether to use tournament selection for archive guide

        Returns:
            Tuple of (final archive solutions, hypervolume history)
        """
        hypervolume_history = []

        for iteration in range(max_iterations):
            w = w_start - (w_start - w_end) * iteration / max_iterations

            for sub_swarm in self.sub_swarms:
                sub_swarm.evolve(w, c1, c2, c3, use_tournament)

            if iteration % 10 == 0:
                hv = self._calculate_hypervolume()
                hypervolume_history.append(hv)
                print(f"Iteration {iteration}, Hypervolume: {hv:.4f}, Archive size: {len(self.archive.solutions)}")

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
            show_analytical: Whether to show the analytical Pareto front for ZDT problems
        """
        if not self.archive.solutions:
            print("Archive is empty, nothing to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        points = np.array([solution.objectives for solution in self.archive.solutions])
        ax.scatter(points[:, 0], points[:, 1], c='green', s=30, label='MGPSO Solutions')

        if show_analytical:
            if isinstance(self.problem, ZDT1):
                f1 = np.linspace(0, 1, 100)
                f2 = 1 - np.sqrt(f1)
                ax.plot(f1, f2, 'r-', label='Analytical Pareto Front')
            elif isinstance(self.problem, ZDT2):
                f1 = np.linspace(0, 1, 100)
                f2 = 1 - (f1 ** 2)
                ax.plot(f1, f2, 'r-', label='Analytical Pareto Front')
            elif isinstance(self.problem, ZDT3):
                f1 = np.linspace(0, 1, 1000)
                f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
                valid = (f2 <= 1) & (f2 >= -1)
                ax.plot(f1[valid], f2[valid], 'r-', label='Analytical Pareto Front')

        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_title('Pareto Front Approximation')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_mgpso_hypervolume_hist(hypervolume_history):
    """
    Plot hypervolume history for MGPSO algorithm.

    Args:
        hypervolume_history: List of hypervolume values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(hypervolume_history) * 10, 10), hypervolume_history)
    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.title('MGPSO Hypervolume Convergence')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    problem = ZDT1(num_variables=30)

    mgpso = MGPSO(problem, swarm_size=30, archive_size=100)

    final_archive, hypervolume_history = mgpso.run(
        max_iterations=100,
        w_start=0.9,
        w_end=0.4,
        c1=2.0,
        c2=2.0,
        c3=1.5,
        use_tournament=True
    )

    mgpso.plot_pareto_front()
    plot_mgpso_hypervolume_hist(hypervolume_history)


def compare_algorithms(problem, max_iterations=100):
    """
    Compare MGDE and MGPSO on the same problem.

    Args:
        problem: Multi-objective optimization problem
        max_iterations: Maximum number of iterations/generations
    """
    mgde = MGDE(problem, population_size=50, archive_size=100)
    mgpso = MGPSO(problem, swarm_size=50, archive_size=100)

    print("Running MGDE...")
    mgde_archive, mgde_hv = mgde.run(max_generations=max_iterations, F=0.5, CR=0.9, use_tournament=True)

    print("Running MGPSO...")
    mgpso_archive, mgpso_hv = mgpso.run(max_iterations=max_iterations, use_tournament=True)

    plt.figure(figsize=(12, 6))

    mgde_points = np.array([solution.objectives for solution in mgde_archive])
    mgpso_points = np.array([solution.objectives for solution in mgpso_archive])

    plt.scatter(mgde_points[:, 0], mgde_points[:, 1], c='blue', s=30, label='MGDE Solutions')
    plt.scatter(mgpso_points[:, 0], mgpso_points[:, 1], c='green', s=30, label='MGPSO Solutions')

    if isinstance(problem, ZDT1):
        f1 = np.linspace(0, 1, 100)
        f2 = 1 - np.sqrt(f1)
        plt.plot(f1, f2, 'r-', label='Analytical Pareto Front')

    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.title(f'Pareto Front Comparison on {problem.__class__.__name__}')
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(10, 6))
    x_mgde = range(0, max_iterations, 10)
    x_mgpso = range(0, max_iterations, 10)

    plt.plot(x_mgde, mgde_hv, 'b-', label='MGDE')
    plt.plot(x_mgpso, mgpso_hv, 'g-', label='MGPSO')

    plt.xlabel('Iteration/Generation')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume Convergence Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Final MGDE Hypervolume: {mgde_hv[-1]:.4f}")
    print(f"Final MGPSO Hypervolume: {mgpso_hv[-1]:.4f}")

    return mgde_archive, mgpso_archive, mgde_hv, mgpso_hv