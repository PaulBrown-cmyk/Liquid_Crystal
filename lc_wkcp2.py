#------------------------------------------------#
#               Import libararies                #
#------------------------------------------------#
import io
import os
import sys
import pstats
import signal
import imageio
import logging
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count, Array, Manager, Lock
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
    def __init__(self, coordinates_file, A=1.0e5, U=3.5, S=0.73, W=1.0e-5, kT=4.11e-21):
        self.A = A
        self.U = U
        self.S = S
        self.W = W  # Rapini-Papoular surface anchoring strength
        self.kT = kT
        self.coordinates_file = coordinates_file

        if os.path.exists('optimized_director_field.txt'):
            self.vertices, self.n_x, self.n_y, self.n_z = self.load_director_field('optimized_director_field.txt')
        else:
            self.vertices = self.load_coordinates()
            self.n_x, self.n_y, self.n_z = self.initialize_directors()

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
        return n_x / norm, n_y / norm, n_z / norm

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

        bulk_energy_density = 0.5 * A * (1 - 0.3333 * U) * Tr_Q2 - 0.3333 * A * U * Tr_Q3 + 0.25 * A * U * Tr_Q2**2

        # Cholesteric energy density
        L = 6.0E-12                 # Elastic constant in N
        #q_0 = 2 * np.pi / (1e-6)   # Pitch wave number in units of m^-1
        q_0 = 2 * np.pi / (0.5e-6)  # Pitch wave number in units of m^-1

        # Compute derivatives of Q-tensor components
        dQ_dx = np.gradient(Q_xx, 0.1, edge_order=2, axis=0)
        dQ_dy = np.gradient(Q_yy, 0.1, edge_order=2, axis=0)
        dQ_dz = np.gradient(Q_zz, 0.1, edge_order=2, axis=0)

        gradient_energy_density = 0.5 * L * (dQ_dx**2 + dQ_dy**2 + dQ_dz**2)

        levi_civita = np.zeros((3, 3, 3))
        levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1
        levi_civita[0, 2, 1] = levi_civita[1, 0, 2] = levi_civita[2, 1, 0] = -1

        cholesteric_energy_density = 2 * q_0 * L * (
            levi_civita[0, 1, 2] * Q_xx * dQ_dx +
            levi_civita[1, 2, 0] * Q_yy * dQ_dy +
            levi_civita[2, 0, 1] * Q_zz * dQ_dz
        )

        total_energy_density = bulk_energy_density + gradient_energy_density + cholesteric_energy_density

        # Rapini-Papoular surface energy term using vectorized operations
        surface_energy_density = np.zeros_like(bulk_energy_density)
        mask = np.logical_or(np.isclose(self.vertices[:, 2], self.vertices[:, 2].max()),
                            np.isclose(self.vertices[:, 2], self.vertices[:, 2].min()))

        n = np.column_stack((self.vertices[:, 0], self.vertices[:, 1], np.zeros(len(self.vertices))))
        norm_n = np.linalg.norm(n, axis=1)
        valid_indices = norm_n != 0
        n[valid_indices] /= norm_n[valid_indices][:, np.newaxis]

        delta_ij = np.eye(3)
        p_ij = delta_ij - np.einsum('ij,ik->ijk', n, n)

        for i in range(len(n_x)):
            R_xx = Q_xx[i] + 0.3333 * S
            R_xy = Q_xy[i]
            R_xz = Q_xz[i]
            R_yy = Q_yy[i] + 0.3333 * S
            R_yz = Q_yz[i]
            R_zz = Q_zz[i] + 0.3333 * S
            R = np.array([[R_xx, R_xy, R_xz], [R_xy, R_yy, R_yz], [R_xz, R_yz, R_zz]])

            K = np.einsum('ij,jk,kl->il', p_ij[i], R, p_ij[i])

            energy_rp = 0.5 * W * np.sum((R - K)**2)

            surface_energy_density[i] = energy_rp

            if np.isnan(energy_rp):
                print(f'NaN detected in surface energy at index {i}: R={R}, K={K}, energy_rp={energy_rp}')

        total_energy = np.sum(total_energy_density) + np.sum(surface_energy_density)

        if np.isnan(total_energy):
            print('NaN detected in total energy')

        return total_energy

    def apply_periodic_boundary_conditions(self):
        self.n_x[0] = self.n_x[-1]
        self.n_y[0] = self.n_y[-1]
        self.n_z[0] = self.n_z[-1]
        return self.n_x, self.n_y, self.n_z

    def plot_director_field(self, ax, iteration=None):
        ax.clear()
        norm = np.sqrt(self.n_x**2 + self.n_y**2 + self.n_z**2)
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

    def load_director_field(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Skip header line
            data = np.array([list(map(float, line.split())) for line in lines])
            self.vertices = data[:, :3]
            self.n_x = data[:, 3]
            self.n_y = data[:, 4]
            self.n_z = data[:, 5]

    def create_movie_from_checkpoints(self, output_filename='simulation.mp4', frame_rate=5):
        checkpoint_files = sorted([file for file in os.listdir() if file.startswith('checkpoint_iter_')])

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
            init_shared_memory(self.n_x, self.n_y, self.n_z)  # Initialize shared memory with loaded data

        if parallel:
            num_processes = cpu_count()
            iterations_per_process = iterations // num_processes
            seeds = np.random.randint(0, 10000, num_processes)

            if checkpoint_file is None:
                init_shared_memory(self.n_x, self.n_y, self.n_z)

            with Manager() as manager:
                results = manager.list()
                with Pool(processes=num_processes, initializer=init_shared_memory, initargs=(self.n_x, self.n_y, self.n_z)) as pool:
                    combined_energies = []

                    for start in range(0, iterations, checkpoint_interval):
                        if terminate:
                            break
                        remaining_iterations = min(checkpoint_interval, iterations - start)
                        result_objects = [pool.apply_async(self._monte_carlo_worker_dynamic, args=(remaining_iterations, seed)) for seed in seeds]

                        changes_x = np.zeros_like(self.n_x)
                        changes_y = np.zeros_like(self.n_y)
                        changes_z = np.zeros_like(self.n_z)

                        for result in result_objects:
                            changes_x_worker, changes_y_worker, changes_z_worker, energies = result.get()
                            changes_x += changes_x_worker
                            changes_y += changes_y_worker
                            changes_z += changes_z_worker
                            combined_energies.extend(energies)

                        with lock:
                            n_x_shared[:] = (shared_memory_to_array(n_x_shared, self.n_x.shape) + changes_x).flatten()
                            n_y_shared[:] = (shared_memory_to_array(n_y_shared, self.n_y.shape) + changes_y).flatten()
                            n_z_shared[:] = (shared_memory_to_array(n_z_shared, self.n_z.shape) + changes_z).flatten()

                        self.n_x = shared_memory_to_array(n_x_shared, self.n_x.shape)
                        self.n_y = shared_memory_to_array(n_y_shared, self.n_y.shape)
                        self.n_z = shared_memory_to_array(n_z_shared, self.n_z.shape)
                        checkpoint_filename = f'checkpoint_iter_{start + remaining_iterations}.txt'
                        self.save_director_field(checkpoint_filename)
                        logger.info(f'Checkpoint saved at iteration {start + remaining_iterations}')

                    self.plot_energy_per_iteration(combined_energies)

                return combined_energies

        current_energy = self.compute_elastic_free_energy(self.n_x, self.n_y, self.n_z)
        logger.info(f'Initial energy: {current_energy}')
        iteration = [0]
        energies = [current_energy]

        for i in range(iterations):
            if terminate:
                break
            self.n_x, self.n_y, self.n_z, current_energy = self._monte_carlo_step(self.n_x, self.n_y, self.n_z, current_energy)
            energies.append(current_energy)

            if ax and i % 100 == 0:
                self.update_animation(ax, iteration)
                logger.info(f'Step {i}: Elastic free energy = {current_energy}')

            if i % checkpoint_interval == 0 and i > 0:
                checkpoint_filename = f'checkpoint_iter_{i}.txt'
                self.save_director_field(checkpoint_filename)
                logger.info(f'Checkpoint saved at iteration {i}')

            if i == iterations:
                logger.info('Minimization complete!')
                self.save_director_field('optimized_director_field.txt')
                break

        self.plot_energy_per_iteration(energies)
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
        num_vertices = len(n_x)
        i = np.random.randint(0, num_vertices)
        theta = np.random.rand() * 2 * np.pi
        phi = np.random.rand() * np.pi
        n_x[i] = np.cos(theta) * np.sin(phi)
        n_y[i] = np.sin(theta) * np.sin(phi)
        n_z[i] = np.cos(phi)
        return n_x, n_y, n_z


    def minimize_free_energy_conjugate_gradient(self, ax):
        def objective_function(params):
            n_x, n_y, n_z = params[:len(self.vertices)], params[len(self.vertices):2*len(self.vertices)], params[2*len(self.vertices):]
            return self.compute_elastic_free_energy(n_x, n_y, n_z)

        def callback(params):
            n_x, n_y, n_z = params[:len(self.vertices)], params[len(self.vertices):2*len(self.vertices)], params[2*len(self.vertices):]
            n_x, n_y, n_z = self.normalize_directors(n_x, n_y, n_z)
            self.n_x, self.n_y, self.n_z = n_x, n_y, n_z
            iteration = [0]
            self.update_animation(ax, iteration)

        params_initial = np.concatenate([self.n_x, self.n_y, self.n_z])
        result = minimize(objective_function, params_initial, method='CG', callback=callback, tol=1e-6)

        self.n_x, self.n_y, self.n_z = result.x[:len(self.vertices)], result.x[len(self.vertices):2*len(self.vertices)], result.x[2*len(self.vertices):]
        self.n_x, self.n_y, self.n_z = self.normalize_directors(self.n_x, self.n_y, self.n_z)

        return self.n_x, self.n_y, self.n_z

    def load_director_field(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Skip header line
            data = np.array([list(map(float, line.split())) for line in lines])
            self.vertices = data[:, :3]
            self.n_x = data[:, 3]
            self.n_y = data[:, 4]
            self.n_z = data[:, 5]
        logger.info(f'Loaded director field from {filename}')

    def load_director_field(self, filename):
        data = np.loadtxt(filename)
        print(f"Loaded data shape: {data.shape}")  # Debugging print
        vertices = data[:, :3]
        n_x = data[:, 3]
        n_y = data[:, 4]
        n_z = data[:, 5]
        print(f"Loaded vertices shape: {vertices.shape}, n_x shape: {n_x.shape}, n_y shape: {n_y.shape}, n_z shape: {n_z.shape}")  # Debugging print
        return vertices, n_x, n_y, n_z

    def plot_energy_per_iteration(self, energies):
        plt.figure(figsize=(10, 6))
        plt.plot(energies, label='Elastic Free Energy')
        plt.xlabel('Iteration', size=20)
        plt.ylabel('F (kJ)', size=20)
        plt.title('Free Energy Per Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig('energy_per_iteration.jpg', dpi=600)
        #plt.show()

    def plot_angle_histogram(self):
        # Calculate the angles of the directors relative to the z-axis
        angles = np.arccos(self.n_z / np.sqrt(self.n_x**2 + self.n_y**2 + self.n_z**2))
        angles_degrees = np.degrees(angles)

        with open('Director_Angles.txt', mode='w') as f:
            f.write('#  Angle  \n')
            for angle in angles_degrees:
                f.write(f'{angle}\n')

        plt.figure(figsize=(8, 6))
        plt.hist(angles_degrees, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel('Angle ($^{\circ}$)', size=20)
        plt.ylabel('Frequency', size=20)
        plt.title('Histogram of Director Angles Relative to the z-axis')
        plt.savefig('histogram.jpg', dpi=600)
        #plt.show()

if __name__ == "__main__":
    import os
    import re

    run_time = 20_000

    coordinates_file = 'straight_cylinder_grid_with_grid.txt'  # Path to your coordinates file
    checkpoint_file = None

    checkpoint_files = [file for file in os.listdir() if file.startswith('checkpoint_iter_')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))  # Sort by iteration number
        checkpoint_file = checkpoint_files[-1]

    cylinder = LiquidCrystalCylinder(coordinates_file)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if checkpoint_file:
        cylinder.load_director_field(checkpoint_file)
        last_iteration = int(re.findall(r'\d+', checkpoint_file)[0])    # sort for the last iteration
        remaining_iterations = max(0, run_time - last_iteration)
        logger.info(f'Resuming from checkpoint {checkpoint_file}, remaining iterations: {remaining_iterations}')
    else:
        initial_energy = cylinder.compute_elastic_free_energy(cylinder.n_x, cylinder.n_y, cylinder.n_z)
        logger.info(f'Initial elastic free energy: {initial_energy}')

        cylinder.apply_periodic_boundary_conditions()

        remaining_iterations = run_time

    try:
        energies = cylinder.minimize_free_energy_monte_carlo(ax, iterations=remaining_iterations, parallel=True, checkpoint_file=checkpoint_file)
    except KeyboardInterrupt:
        logger.info('Execution interrupted by user')

    final_energy_mc = energies[-1]
    logger.info(f'Final elastic free energy (Monte Carlo): {final_energy_mc / 1000.} kJ')

    cylinder.plot_director_field(ax)

    cylinder.plot_angle_histogram()

    cylinder.plot_energy_per_iteration(energies)

    cylinder.create_movie_from_checkpoints(output_filename='cholesteric_LC_simulation.mp4', frame_rate=5)


