import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CylinderGrid:
    def __init__(self, diameter_um, length_um, num_boundary_points_per_z, num_z_levels, num_inner_points, min_distance_um):
        self.diameter = diameter_um
        self.length = length_um
        self.radius = diameter_um / 2
        self.num_boundary_points_per_z = num_boundary_points_per_z
        self.num_z_levels = num_z_levels
        self.num_inner_points = num_inner_points
        self.min_distance = min_distance_um

    def generate_straight_cylinder(self, variable_diameter=None):
        if variable_diameter is None:
            variable_diameter = np.full(self.num_z_levels, self.diameter)
        
        z_boundary = np.linspace(0, self.length, self.num_z_levels)
        theta = np.linspace(0, 2 * np.pi, self.num_boundary_points_per_z)
        theta, z_boundary = np.meshgrid(theta, z_boundary)
        theta = theta.flatten()
        z_boundary = z_boundary.flatten()
        
        variable_radius = (variable_diameter / 2).repeat(self.num_boundary_points_per_z)

        X_boundary = variable_radius * np.cos(theta)
        Y_boundary = variable_radius * np.sin(theta)
        Z_boundary = z_boundary

        X_inner, Y_inner, Z_inner = self._generate_inner_points_grid()

        self.X = np.concatenate((X_boundary, X_inner))
        self.Y = np.concatenate((Y_boundary, Y_inner))
        self.Z = np.concatenate((Z_boundary, Z_inner))

    def generate_straight_cylinder_with_grid(self):
        # Boundary points
        z_boundary = np.linspace(0, self.length, self.num_z_levels)
        theta = np.linspace(0, 2 * np.pi, self.num_boundary_points_per_z)
        theta, z_boundary = np.meshgrid(theta, z_boundary)
        theta = theta.flatten()
        z_boundary = z_boundary.flatten()
        
        radius = np.full_like(theta, self.radius)

        X_boundary = radius * np.cos(theta)
        Y_boundary = radius * np.sin(theta)
        Z_boundary = z_boundary

        # Interior grid points
        X_inner, Y_inner, Z_inner = self._generate_interior_grid_points()

        self.X = np.concatenate((X_boundary, X_inner))
        self.Y = np.concatenate((Y_boundary, Y_inner))
        self.Z = np.concatenate((Z_boundary, Z_inner))

    def generate_bent_cylinder(self, bend_function):
        z_boundary = np.linspace(0, self.length, self.num_z_levels)
        theta = np.linspace(0, 2 * np.pi, self.num_boundary_points_per_z)
        theta, z_boundary = np.meshgrid(theta, z_boundary)
        theta = theta.flatten()
        z_boundary = z_boundary.flatten()
        
        X_boundary = self.radius * np.cos(theta)
        Y_boundary = self.radius * np.sin(theta)
        Z_boundary = z_boundary

        bend = bend_function(Z_boundary, self.length, self.radius)
        X_boundary += bend[0]
        Y_boundary += bend[1]

        X_inner, Y_inner, Z_inner = self._generate_inner_points_grid(lambda z: bend_function(z, self.length, self.radius))

        self.X = np.concatenate((X_boundary, X_inner))
        self.Y = np.concatenate((Y_boundary, Y_inner))
        self.Z = np.concatenate((Z_boundary, Z_inner))

    def _generate_inner_points_grid(self, bend_function=None):
        # Generate a grid of points within the radius and length of the cylinder
        grid_spacing = self.min_distance
        r_values = np.arange(grid_spacing, self.radius, grid_spacing)
        theta_values = np.arange(0, 2 * np.pi, grid_spacing / self.radius)
        z_values = np.arange(0, self.length, grid_spacing)

        R, Theta, Z = np.meshgrid(r_values, theta_values, z_values)
        R = R.flatten()
        Theta = Theta.flatten()
        Z = Z.flatten()

        X_inner = R * np.cos(Theta)
        Y_inner = R * np.sin(Theta)
        Z_inner = Z

        if bend_function is not None:
            bends = bend_function(Z_inner)
            X_inner += bends[0]
            Y_inner += bends[1]

        return X_inner, Y_inner, Z_inner

    def _generate_interior_grid_points(self):
        grid_spacing = self.min_distance
        x_values = np.arange(-self.radius, self.radius + grid_spacing, grid_spacing)
        y_values = np.arange(-self.radius, self.radius + grid_spacing, grid_spacing)
        z_values = np.arange(0, self.length + grid_spacing, grid_spacing)
        
        X, Y, Z = np.meshgrid(x_values, y_values, z_values)
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        # Filter points to be inside the cylinder
        distance_from_axis = np.sqrt(X**2 + Y**2)
        inside_cylinder = distance_from_axis <= self.radius

        return X[inside_cylinder], Y[inside_cylinder], Z[inside_cylinder]

    def save_grid_to_file(self, filename):
        data = np.vstack((self.X, self.Y, self.Z)).T
        np.savetxt(filename, data, fmt='%.6f', header='X Y Z')

    def plot_grid(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X, self.Y, self.Z, s=1)

        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_zlabel('Z (µm)')
        ax.set_title('Cylinder Grid (µm)')
        plt.savefig('cylinder_grid.jpg', dpi=1000)
        #plt.show()

def bend_function(z, length, radius):
    bend_angle = np.pi / 2  # 90 degrees in radians
    bend_y = np.sin((np.pi / length) * z) * (radius / 2)
    bend_x = np.zeros_like(z)
    return bend_y, bend_x


if __name__ == '__main__':

    cylinder = CylinderGrid(diameter_um=5, length_um=20, num_boundary_points_per_z=20, num_z_levels=20, num_inner_points=20, min_distance_um=1.0)
    cylinder.generate_straight_cylinder_with_grid()
    cylinder.plot_grid()
    cylinder.save_grid_to_file('straight_cylinder_grid_with_grid.txt')

    #cylinder.generate_bent_cylinder(bend_function)
    #cylinder.plot_grid()
    #cylinder.save_grid_to_file('bent_cylinder_grid.txt')

