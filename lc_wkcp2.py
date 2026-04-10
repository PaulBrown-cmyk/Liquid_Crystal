#------------------------------------------------#
#               Import libararies                #
#------------------------------------------------#
import io
import os
import sys
import pstats
import signal
import warnings
import imageio
import logging
import cProfile
import numpy as np

# Keep Matplotlib's cache in a writable location when running on a headless
# shell or CI host.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "codex_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join("/tmp", "codex_cache"))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count, Array, Manager, Lock
from lc_fem import run_fem_solver
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    #"font.sans-serif": "Helvetica",
})

# Global flag to indicate if termination signal is received
terminate = False

# Signal handler
def signal_handler(sig, frame):
    global terminate
    print('Termination signal received. Saving current state...')
    terminate = True

signal.signal(signal.SIGINT, signal_handler)   # For interrupt signal (Ctrl+C)
signal.signal(signal.SIGTERM, signal_handler)  # For termination signal
#------------------------------------------------#
# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
#------------------------------------------------#
n_x_shared = None
n_y_shared = None
n_z_shared = None
lock = None
#------------------------------------------------#

# Load balancing methods for Class: LiquidCrystalCylinder
def array_to_shared_memory(arr):
    shared_array = Array('d', arr.flatten(), lock=True)
    return shared_array

def shared_memory_to_array(shared_array, shape):
    arr = np.frombuffer(shared_array.get_obj())
    return arr.reshape(shape)

def init_shared_memory(n_x, n_y, n_z):
    global n_x_shared, n_y_shared, n_z_shared, lock
    n_x_shared = array_to_shared_memory(n_x)
    n_y_shared = array_to_shared_memory(n_y)
    n_z_shared = array_to_shared_memory(n_z)
    lock = Lock()

def profile(func):
    """A decorator that profiles a function."""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

class LiquidCrystalCylinder:
    """Legacy Monte Carlo cylinder solver.

    This implementation is quarantined for comparison and historical reference
    only. The FEM continuum solver in lc_fem.py is the supported path.
    """
    def __init__(self, coordinates_file, A=1.0e5, U=3.5, S=0.73, W=1.0e-5, kT=4.11e-21):
        warnings.warn(
            "LiquidCrystalCylinder is legacy/quarantined; use the FEM solver in lc_fem.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.A = A
        self.U = U
        self.S = S
        self.W = W  # Rapini-Papoular surface anchoring strength
        self.kT = kT
        self.coordinates_file = coordinates_file
        self._geometry_prepared = False

        if os.path.exists('optimized_director_field.txt'):
            # Restart from the last optimized field if it exists.
            self.load_director_field('optimized_director_field.txt')
        else:
            self.vertices = self.load_coordinates()
            self.n_x, self.n_y, self.n_z = self.initialize_directors()
            self._prepare_geometry()

    def load_coordinates(self):
        vertices = np.loadtxt(self.coordinates_file)
        vertices *= 1e-6  # Convert micrometers to meters
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        print(f'Coordinates dimensions: {max_coords - min_coords} meters')
        return vertices

    def initialize_directors(self):
        theta = np.random.uniform(0, 2 * np.pi, len(self.vertices))
        phi = np.random.uniform(0, np.pi, len(self.vertices))
        n_x = np.cos(theta) * np.sin(phi)
        n_y = np.sin(theta) * np.sin(phi)
        n_z = np.cos(phi)
        return self.normalize_directors(n_x, n_y, n_z)

    def normalize_directors(self, n_x, n_y, n_z):
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        # Guard against the rare case where a proposal is numerically zero.
        norm = np.where(norm == 0, 1.0, norm)
        return n_x / norm, n_y / norm, n_z / norm

    def _prepare_geometry(self):
        """Cache geometry-dependent quantities for the cylinder point cloud."""
        if self._geometry_prepared:
            return

        self._spacing = self._estimate_spacing()
        self._vertex_volume = self._spacing ** 3
        self._surface_area_weight = self._spacing ** 2
        self._boundary_mask, self._surface_normals = self._identify_boundary_vertices()
        self._neighbor_pairs, self._neighbor_vectors, self._neighbor_distances = self._build_neighbor_graph()
        self._geometry_prepared = True

    def _estimate_spacing(self):
        """Estimate the characteristic point spacing from nearest-neighbor distances."""
        if len(self.vertices) < 2:
            return 1.0

        tree = cKDTree(self.vertices)
        distances, _ = tree.query(self.vertices, k=min(2, len(self.vertices)))
        nearest = distances[:, 1] if distances.ndim == 2 else distances[1:]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if nearest.size == 0:
            return 1.0
        return float(np.median(nearest))

    def _identify_boundary_vertices(self):
        """Identify lateral wall and end-cap nodes and assign outward normals.

        The geometry in this project is a cylinder aligned with the z-axis.
        If the grid is later rotated, this method should be updated accordingly.
        """
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.vertices[:, 2]
        radial_distance = np.sqrt(x**2 + y**2)

        radial_tol = max(0.25 * self._spacing, 1e-12)
        z_tol = max(0.25 * self._spacing, 1e-12)

        side_mask = np.isclose(radial_distance, radial_distance.max(), atol=radial_tol)
        z_min_mask = np.isclose(z, z.min(), atol=z_tol)
        z_max_mask = np.isclose(z, z.max(), atol=z_tol)
        cap_mask = z_min_mask | z_max_mask
        boundary_mask = side_mask | cap_mask

        normals = np.zeros_like(self.vertices)
        for i in np.where(boundary_mask)[0]:
            contributions = []

            if side_mask[i] and radial_distance[i] > 0:
                contributions.append(np.array([x[i], y[i], 0.0]) / radial_distance[i])

            if z_min_mask[i]:
                contributions.append(np.array([0.0, 0.0, -1.0]))

            if z_max_mask[i]:
                contributions.append(np.array([0.0, 0.0, 1.0]))

            normal = np.sum(contributions, axis=0)
            norm = np.linalg.norm(normal)
            if norm == 0:
                # Fallback for degenerate corner points.
                normal = np.array([0.0, 0.0, 1.0])
                norm = 1.0
            normals[i] = normal / norm

        return boundary_mask, normals

    def _build_neighbor_graph(self, max_neighbors=6):
        """Build a symmetric nearest-neighbor graph over the point cloud.

        The elastic free energy is evaluated on this graph rather than on array
        order, because the grid is an unstructured point cloud in physical space.
        """
        n_points = len(self.vertices)
        if n_points < 2:
            return np.empty((0, 2), dtype=int), np.empty((0, 3)), np.empty(0)

        tree = cKDTree(self.vertices)
        k = min(max_neighbors + 1, n_points)
        distances, indices = tree.query(self.vertices, k=k)

        edge_map = {}
        cutoff = 1.75 * self._spacing

        for i in range(n_points):
            for j_idx in range(1, k):
                j = int(indices[i, j_idx])
                distance = float(distances[i, j_idx])
                if not np.isfinite(distance) or distance <= 0 or distance > cutoff:
                    continue

                a, b = sorted((i, j))
                if a == b:
                    continue

                # Keep the shortest available bond if the same edge is discovered twice.
                current = edge_map.get((a, b))
                if current is None or distance < current:
                    edge_map[(a, b)] = distance

        if not edge_map:
            return np.empty((0, 2), dtype=int), np.empty((0, 3)), np.empty(0)

        pairs = np.array(list(edge_map.keys()), dtype=int)
        distances = np.array([edge_map[tuple(pair)] for pair in pairs], dtype=float)
        vectors = self.vertices[pairs[:, 1]] - self.vertices[pairs[:, 0]]
        return pairs, vectors, distances

    @staticmethod
    def _rotate_about_axis(vector, axis, angle):
        """Rotate a vector about an axis using Rodrigues' formula."""
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return vector

        axis = axis / axis_norm
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return (
            vector * cos_angle
            + np.cross(axis, vector) * sin_angle
            + axis * np.dot(axis, vector) * (1.0 - cos_angle)
        )

    @staticmethod
    def _rotate_about_axis_matrix(matrix, axis, angle):
        """Rotate a 3x3 tensor with the same Rodrigues rotation used for directors."""
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return matrix

        axis = axis / axis_norm
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        R = np.eye(3) + sin_angle * K + (1.0 - cos_angle) * (K @ K)
        return R @ matrix @ R.T

    def compute_Q_tensor(self, n_x, n_y, n_z):
        S = self.S
        Q_xx = S * (n_x**2 - 1/3)   # Diagonal terms
        Q_yy = S * (n_y**2 - 1/3)
        Q_zz = S * (n_z**2 - 1/3)
        Q_xy = S * (n_x * n_y)      # Off diagonal terms
        Q_xz = S * (n_x * n_z)
        Q_yz = S * (n_y * n_z)
        return Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz

    def compute_elastic_free_energy(self, n_x, n_y, n_z):
        self._prepare_geometry()
        A, U, W, S = self.A, self.U, self.W, self.S
        Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz = self.compute_Q_tensor(n_x, n_y, n_z)

        # Compute Tr(Q^2)
        Tr_Q2 = Q_xx**2 + Q_yy**2 + Q_zz**2 + 2*(Q_xy**2 + Q_xz**2 + Q_yz**2)

        # Compute Tr(Q^3) using vectorized operations
        Tr_Q3 = (Q_xx * (Q_xx * Q_xx + 2 * Q_xy * Q_xy + 2 * Q_xz * Q_xz) +
                Q_yy * (Q_yy * Q_yy + 2 * Q_xy * Q_xy + 2 * Q_yz * Q_yz) +
                Q_zz * (Q_zz * Q_zz + 2 * Q_xz * Q_xz + 2 * Q_yz * Q_yz) +
                2 * (Q_xy * (Q_xx * Q_xy + Q_yy * Q_yz + Q_yz * Q_zz) +
                    Q_xz * (Q_xx * Q_xz + Q_yy * Q_yz + Q_yz * Q_zz) +
                    Q_yz * (Q_yy * Q_yz + Q_xx * Q_xz + Q_zz * Q_yz)))

        # Landau-de Gennes bulk free energy density.
        bulk_energy_density = (
            0.5 * A * (1.0 - U / 3.0) * Tr_Q2
            - (A * U / 3.0) * Tr_Q3
            + 0.25 * A * U * Tr_Q2**2
        )
        bulk_energy = np.sum(bulk_energy_density) * self._vertex_volume

        # Discrete cholesteric / elastic coupling on the nearest-neighbor graph.
        # Each bond compares Q_j to Q_i rotated by the preferred helical twist
        # about the bond axis. This is a graph-based approximation to the
        # continuum gradient terms that respects the unstructured cylinder grid.
        L = 6.0E-12                 # Elastic constant in N
        q_0 = 2 * np.pi / (0.5e-6)  # Preferred helical wave number in m^-1

        Q_matrices = np.empty((len(n_x), 3, 3))
        Q_matrices[:, 0, 0] = Q_xx
        Q_matrices[:, 0, 1] = Q_matrices[:, 1, 0] = Q_xy
        Q_matrices[:, 0, 2] = Q_matrices[:, 2, 0] = Q_xz
        Q_matrices[:, 1, 1] = Q_yy
        Q_matrices[:, 1, 2] = Q_matrices[:, 2, 1] = Q_yz
        Q_matrices[:, 2, 2] = Q_zz

        elastic_energy = 0.0
        for (i, j), bond_vector, bond_length in zip(self._neighbor_pairs, self._neighbor_vectors, self._neighbor_distances):
            if bond_length <= 0:
                continue

            bond_axis = bond_vector / bond_length
            preferred_rotation = q_0 * bond_length
            Q_i_rot = self._rotate_about_axis_matrix(Q_matrices[i], bond_axis, preferred_rotation)
            delta_Q = Q_matrices[j] - Q_i_rot

            # Bond energy approximates \int (L/2)|∇Q|^2 dV on a point cloud.
            bond_weight = self._vertex_volume / (bond_length**2)
            elastic_energy += 0.5 * L * bond_weight * np.sum(delta_Q**2)

        # Rapini-Papoular anchoring on the actual cylinder boundary only.
        # We use planar anchoring: the director prefers to lie tangent to the surface.
        surface_energy = 0.0
        if np.any(self._boundary_mask):
            boundary_n = np.column_stack((n_x, n_y, n_z))
            alignment = np.sum(boundary_n * self._surface_normals, axis=1)
            surface_energy = 0.5 * W * self._surface_area_weight * np.sum(alignment[self._boundary_mask] ** 2)

        total_energy = bulk_energy + elastic_energy + surface_energy

        if np.isnan(total_energy):
            print('NaN detected in total energy')

        return total_energy

    def apply_periodic_boundary_conditions(self):
        # Legacy no-op: a finite cylinder is not periodic. Rapini-Papoular
        # anchoring is applied explicitly in the surface-energy term, so there
        # is no periodic wraparound to enforce here.
        return self.n_x, self.n_y, self.n_z

    def plot_director_field(self, ax, iteration=None):
        ax.clear()
        norm = np.sqrt(self.n_x**2 + self.n_y**2 + self.n_z**2)
        norm = np.where(norm == 0, 1.0, norm)
        # Adjust the scaling factor to ensure visibility
        scaling_factor = 10  # Increase or adjust this factor to make changes more visible
        ax.quiver((self.vertices[:, 0])*1e6, (self.vertices[:, 1])*1e6, (self.vertices[:, 2])*1e6,
                (self.n_x / norm)*scaling_factor, (self.n_y / norm)*scaling_factor, (self.n_z / norm)*scaling_factor,
                length=0.1, normalize=True, pivot='middle')
        title = 'Relaxed Liquid Crystal Director Field'
        if iteration is not None:
            title += f' - Iteration {iteration}'
        ax.set_title(title)
        ax.set_xlim3d(np.min(self.vertices[:, 0])*1e6, np.max(self.vertices[:, 0])*1e6)
        ax.set_ylim3d(np.min(self.vertices[:, 1])*1e6, np.max(self.vertices[:, 1])*1e6)
        ax.set_zlim3d(np.min(self.vertices[:, 2])*1e6, np.max(self.vertices[:, 2])*1e6)
        plt.draw()


    def update_animation(self, ax, iteration):
        self.plot_director_field(ax, iteration[0])
        energy = self.compute_elastic_free_energy(self.n_x, self.n_y, self.n_z)
        print(f'Iteration {iteration[0]}: Elastic free energy = {energy}')
        iteration[0] += 1

    def save_director_field(self, filename):
        with open(filename, 'w') as f:
            f.write('# x (m)     y (m)     z (m)     n_x     n_y     n_z\n')
            for i in range(len(self.vertices)):
                f.write(f'{self.vertices[i, 0]:.6e}     {self.vertices[i, 1]:.6e}     {self.vertices[i, 2]:.6e}     {self.n_x[i]:.6e}     {self.n_y[i]:.6e}     {self.n_z[i]:.6e}\n')

    def create_movie_from_checkpoints(self, output_filename='simulation.mp4', frame_rate=5):
        import re

        checkpoint_files = sorted(
            [file for file in os.listdir() if file.startswith('checkpoint_iter_')],
            key=lambda filename: int(re.findall(r'\d+', filename)[0]),
        )
        if not checkpoint_files:
            logger.info('No checkpoint files were found, so no movie was created.')
            return

        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        def update_plot(filename):
            self.load_director_field(filename)
            self.plot_director_field(ax)
            ax.set_title(f'Checkpoint: {filename}')

        def save_frame(filename):
            update_plot(filename)
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(height, width, 4)  # RGBA has 4 channels
            return image

        frames = [save_frame(file) for file in checkpoint_files]
        imageio.mimsave(output_filename, frames, fps=frame_rate)
        print(f'Movie saved as {output_filename}')
        logger.info(f'Movie created from checkpoints and saved as {output_filename}')


    def _monte_carlo_worker_dynamic(self, iterations, seed):
        global n_x_shared, n_y_shared, n_z_shared, lock, terminate

        np.random.seed(seed)
        n_x = shared_memory_to_array(n_x_shared, self.n_x.shape)
        n_y = shared_memory_to_array(n_y_shared, self.n_y.shape)
        n_z = shared_memory_to_array(n_z_shared, self.n_z.shape)
        current_energy = self.compute_elastic_free_energy(n_x, n_y, n_z)
        energies = [current_energy]

        changes_x = np.zeros_like(n_x)
        changes_y = np.zeros_like(n_y)
        changes_z = np.zeros_like(n_z)

        for i in range(iterations):
            if terminate:
                break
            n_x_new, n_y_new, n_z_new, current_energy = self._monte_carlo_step(n_x, n_y, n_z, current_energy)
            energies.append(current_energy)

            changes_x += n_x_new - n_x
            changes_y += n_y_new - n_y
            changes_z += n_z_new - n_z

            n_x, n_y, n_z = n_x_new, n_y_new, n_z_new

            # Log progress
            if i % 100 == 0:
                logger.info(f'Worker with seed {seed}: Iteration {i}, Energy = {current_energy}')

        with lock:
            n_x_shared[:] = (shared_memory_to_array(n_x_shared, self.n_x.shape) + changes_x).flatten()
            n_y_shared[:] = (shared_memory_to_array(n_y_shared, self.n_y.shape) + changes_y).flatten()
            n_z_shared[:] = (shared_memory_to_array(n_z_shared, self.n_z.shape) + changes_z).flatten()

        return changes_x, changes_y, changes_z, energies


    def minimize_free_energy_monte_carlo(self, ax=None, iterations=10000, parallel=False, checkpoint_interval=1000, checkpoint_file=None):
        global n_x_shared, n_y_shared, n_z_shared, lock, terminate

        if checkpoint_file is not None:
            self.load_director_field(checkpoint_file)
            logger.info(f'Restarted from checkpoint: {checkpoint_file}')
            # The sequential path reads directly from the loaded arrays.

        if parallel:
            # The old parallel update scheme combined independent trajectories
            # into one shared state, which is not a valid Monte Carlo update for
            # a coupled elastic system. Keep the API, but run sequentially.
            logger.warning('parallel=True is disabled for coupled Monte Carlo updates; running sequentially instead.')

        current_energy = self.compute_elastic_free_energy(self.n_x, self.n_y, self.n_z)
        logger.info(f'Initial energy: {current_energy}')
        iteration = [0]
        energies = [current_energy]

        for i in range(iterations):
            if terminate:
                break
            self.n_x, self.n_y, self.n_z, current_energy = self._monte_carlo_step(self.n_x, self.n_y, self.n_z, current_energy)
            energies.append(current_energy)

            if ax is not None and i % 100 == 0:
                self.update_animation(ax, iteration)
                logger.info(f'Step {i}: Elastic free energy = {current_energy}')

            if i % checkpoint_interval == 0 and i > 0:
                checkpoint_filename = f'checkpoint_iter_{i}.txt'
                self.save_director_field(checkpoint_filename)
                logger.info(f'Checkpoint saved at iteration {i}')

        self.plot_energy_per_iteration(energies)

        if not terminate:
            logger.info('Minimization complete!')
            self.save_director_field('optimized_director_field.txt')

        return energies



    def _monte_carlo_step(self, n_x, n_y, n_z, current_energy):
        n_x_new, n_y_new, n_z_new = self._perturb_directors(n_x.copy(), n_y.copy(), n_z.copy())
        n_x_new, n_y_new, n_z_new = self.normalize_directors(n_x_new, n_y_new, n_z_new)
        new_energy = self.compute_elastic_free_energy(n_x_new, n_y_new, n_z_new)

        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            probability = 1.0
        else:
            probability = np.exp(-delta_energy / self.kT)

        if np.isinf(probability) or np.isnan(probability):
            probability = 0.0  

        if new_energy < current_energy or np.random.rand() < probability:
            return n_x_new, n_y_new, n_z_new, new_energy
        else:
            return n_x, n_y, n_z, current_energy

    def _perturb_directors(self, n_x, n_y, n_z):
        # Use a small local rotation so the Metropolis step explores the energy
        # landscape gradually instead of replacing a director with a random vector.
        num_vertices = len(n_x)
        i = np.random.randint(0, num_vertices)
        current = np.array([n_x[i], n_y[i], n_z[i]])

        random_vector = np.random.normal(size=3)
        random_vector -= np.dot(random_vector, current) * current
        norm = np.linalg.norm(random_vector)
        if norm == 0:
            random_vector = np.array([1.0, 0.0, 0.0])
            random_vector -= np.dot(random_vector, current) * current
            norm = np.linalg.norm(random_vector)

        rotation_axis = random_vector / norm
        rotation_angle = np.deg2rad(10.0) * (2.0 * np.random.rand() - 1.0)
        rotated = self._rotate_about_axis(current, rotation_axis, rotation_angle)

        n_x[i], n_y[i], n_z[i] = rotated
        return n_x, n_y, n_z


    def minimize_free_energy_conjugate_gradient(self, ax):
        callback_counter = [0]

        def objective_function(params):
            n_x = params[:len(self.vertices)]
            n_y = params[len(self.vertices):2*len(self.vertices)]
            n_z = params[2*len(self.vertices):]
            n_x, n_y, n_z = self.normalize_directors(n_x, n_y, n_z)
            return self.compute_elastic_free_energy(n_x, n_y, n_z)

        def callback(params):
            n_x = params[:len(self.vertices)]
            n_y = params[len(self.vertices):2*len(self.vertices)]
            n_z = params[2*len(self.vertices):]
            n_x, n_y, n_z = self.normalize_directors(n_x, n_y, n_z)
            self.n_x, self.n_y, self.n_z = n_x, n_y, n_z
            iteration = [callback_counter[0]]
            self.update_animation(ax, iteration)
            callback_counter[0] += 1

        params_initial = np.concatenate([self.n_x, self.n_y, self.n_z])
        result = minimize(objective_function, params_initial, method='CG', callback=callback, tol=1e-6)

        self.n_x, self.n_y, self.n_z = result.x[:len(self.vertices)], result.x[len(self.vertices):2*len(self.vertices)], result.x[2*len(self.vertices):]
        self.n_x, self.n_y, self.n_z = self.normalize_directors(self.n_x, self.n_y, self.n_z)

        return self.n_x, self.n_y, self.n_z

    def load_director_field(self, filename):
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[1] < 6:
            raise ValueError(f'{filename} does not contain x, y, z, n_x, n_y, n_z columns')

        self.vertices = data[:, :3]
        self.n_x = data[:, 3]
        self.n_y = data[:, 4]
        self.n_z = data[:, 5]
        self._geometry_prepared = False
        self._prepare_geometry()
        logger.info(f'Loaded director field from {filename}')
        return self.vertices, self.n_x, self.n_y, self.n_z

    def plot_energy_per_iteration(self, energies):
        if not energies:
            logger.warning('No energies were recorded, so no energy plot was created.')
            return
        plt.figure(figsize=(10, 6))
        plt.plot(energies, label='Elastic Free Energy')
        plt.xlabel('Iteration', size=20)
        plt.ylabel('F (J)', size=20)
        plt.title('Free Energy Per Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig('energy_per_iteration.jpg', dpi=600)
        #plt.show()

    def plot_angle_histogram(self):
        # Calculate the angles of the directors relative to the z-axis
        norm = np.sqrt(self.n_x**2 + self.n_y**2 + self.n_z**2)
        norm = np.where(norm == 0, 1.0, norm)
        cos_theta = np.clip(self.n_z / norm, -1.0, 1.0)
        angles = np.arccos(cos_theta)
        angles_degrees = np.degrees(angles)

        with open('Director_Angles.txt', mode='w') as f:
            f.write('#  Angle  \n')
            for angle in angles_degrees:
                f.write(f'{angle}\n')

        plt.figure(figsize=(8, 6))
        plt.hist(angles_degrees, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel(r'Angle ($^{\circ}$)', size=20)
        plt.ylabel('Frequency', size=20)
        plt.title('Histogram of Director Angles Relative to the z-axis')
        plt.savefig('histogram.jpg', dpi=600)
        #plt.show()


def ensure_default_cylinder_grid(filename: str) -> str:
    """Create the default straight-cylinder grid if it is missing.

    The continuum solver expects a real point cloud on disk. Generating the
    canonical geometry here keeps a fresh checkout runnable without requiring a
    separately saved data artifact.
    """
    if os.path.exists(filename):
        return filename

    from grid1 import CylinderGrid

    cylinder = CylinderGrid(
        diameter_um=5,
        length_um=20,
        num_boundary_points_per_z=20,
        num_z_levels=20,
        num_inner_points=20,
        min_distance_um=1.0,
    )
    cylinder.generate_straight_cylinder_with_grid()
    cylinder.save_grid_to_file(filename)
    logger.info(f"Generated default cylinder grid at {filename}")
    return filename

if __name__ == "__main__":
    import os
    import re

    # The FEM solver is now the main path for production runs. The legacy
    # Monte Carlo implementation above is kept for comparison and historical
    # reference, but the entry point uses the continuum formulation.
    run_time = 500
    coordinates_file = ensure_default_cylinder_grid("straight_cylinder_grid_with_grid.txt")
    checkpoint_file = None

    checkpoint_files = [file for file in os.listdir() if file.startswith("checkpoint_iter_")]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda f: int(re.findall(r"\d+", f)[0]))
        checkpoint_file = checkpoint_files[-1]
        logger.info(f"Found checkpoint {checkpoint_file}; resuming the FEM solver from that state.")

    solver, energies = run_fem_solver(
        coordinates_file=coordinates_file,
        checkpoint_file=checkpoint_file,
        run_time=run_time,
        output_prefix="cholesteric_fem",
    )

    final_energy = energies[-1]
    logger.info(f"Final continuum free energy: {final_energy / 1000.0} kJ")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    solver.plot_director_field(ax, title="Relaxed FEM Director Field")
