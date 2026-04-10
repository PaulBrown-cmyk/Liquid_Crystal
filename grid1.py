from dataclasses import dataclass
import numpy as np
import os

# Keep Matplotlib usable in headless shells and CI.
os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "codex_mplconfig")
os.environ["XDG_CACHE_HOME"] = os.path.join("/tmp", "codex_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class StructuredCylinderSolveMesh:
    """Body-fitted cylindrical FEM mesh with explicit connectivity."""

    nodes_m: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray
    boundary_face_kind: np.ndarray
    units: str = "m"
    metadata_json: str = "{}"


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
        # Use endpoint=False so the seam at 2π does not duplicate the 0-angle point.
        theta = np.linspace(0, 2 * np.pi, self.num_boundary_points_per_z, endpoint=False)
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
        # Use endpoint=False so the seam at 2π does not duplicate the 0-angle point.
        theta = np.linspace(0, 2 * np.pi, self.num_boundary_points_per_z, endpoint=False)
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
        # Use endpoint=False so the seam at 2π does not duplicate the 0-angle point.
        theta = np.linspace(0, 2 * np.pi, self.num_boundary_points_per_z, endpoint=False)
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
        # Keep interior points away from the end caps so they do not duplicate
        # the explicit boundary layers.
        z_values = np.arange(grid_spacing, self.length, grid_spacing)

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
        # Sample only the interior, leaving the side wall and end caps to the
        # dedicated boundary grid.
        x_values = np.arange(-self.radius + grid_spacing, self.radius, grid_spacing)
        y_values = np.arange(-self.radius + grid_spacing, self.radius, grid_spacing)
        z_values = np.arange(grid_spacing, self.length, grid_spacing)
        
        X, Y, Z = np.meshgrid(x_values, y_values, z_values)
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        # Filter points to be strictly inside the cylinder.
        distance_from_axis = np.sqrt(X**2 + Y**2)
        inside_cylinder = distance_from_axis < self.radius

        return X[inside_cylinder], Y[inside_cylinder], Z[inside_cylinder]

    @staticmethod
    def _cluster_wall_positions(count: int, radius: float, bias: float) -> np.ndarray:
        """Return monotonically increasing radial nodes clustered near the wall."""
        if count < 2:
            raise ValueError("Need at least two radial layers to build a solve mesh.")
        s = np.linspace(0.0, 1.0, count)
        if bias is None or bias <= 1.0:
            return radius * s
        clustered = 1.0 - np.power(1.0 - s, bias)
        clustered[0] = 0.0
        clustered[-1] = 1.0
        return radius * clustered

    @staticmethod
    def _cluster_end_positions(count: int, length: float, bias: float) -> np.ndarray:
        """Return monotonically increasing axial nodes clustered near the caps."""
        if count < 2:
            raise ValueError("Need at least two axial layers to build a solve mesh.")
        s = np.linspace(0.0, 1.0, count)
        if bias is None or bias <= 1.0:
            return length * s
        exponent = np.power(s, bias)
        mirrored = np.power(1.0 - s, bias)
        denom = exponent + mirrored
        clustered = np.divide(exponent, denom, out=np.zeros_like(s), where=denom > 0.0)
        clustered[0] = 0.0
        clustered[-1] = 1.0
        return length * clustered

    @staticmethod
    def _tetra_volume(nodes: np.ndarray) -> float:
        """Return the absolute volume of a tetrahedron from its four vertices."""
        return abs(np.linalg.det(nodes[1:] - nodes[0])) / 6.0

    def generate_structured_solve_mesh(
        self,
        num_radial_layers: int = 5,
        num_theta_points: int = 24,
        num_axial_layers: int = 12,
        radial_cluster_power: float = 2.0,
        axial_cluster_power: float = 2.0,
    ) -> StructuredCylinderSolveMesh:
        """Build a body-fitted cylindrical tetrahedral mesh.

        The solve mesh is intentionally separate from the plotting point cloud:
        it keeps the cylinder boundary explicit, stores tetrahedral connectivity,
        and tags the wall and caps so the FEM solver can apply boundary physics
        without guessing from array order.
        """
        if num_theta_points < 6:
            raise ValueError("num_theta_points should be at least 6 for a sensible cylinder mesh.")

        # Build a cylindrical layer stack: a center node plus concentric rings
        # in the cross-section, then extrude those rings along z.
        radii = self._cluster_wall_positions(num_radial_layers, self.radius, radial_cluster_power)
        z_layers = self._cluster_end_positions(num_axial_layers, self.length, axial_cluster_power)
        theta = np.linspace(0.0, 2.0 * np.pi, num_theta_points, endpoint=False)

        xy_nodes: list[tuple[float, float]] = [(0.0, 0.0)]
        ring_node_indices: list[list[int]] = [[0]]
        for radius in radii[1:]:
            ring = []
            for angle in theta:
                ring.append(len(xy_nodes))
                xy_nodes.append((radius * np.cos(angle), radius * np.sin(angle)))
            ring_node_indices.append(ring)

        xy_nodes_arr = np.asarray(xy_nodes, dtype=float)

        base_triangles: list[tuple[int, int, int]] = []
        boundary_edges: list[tuple[int, int]] = []

        # Fan the center node to the first ring.
        first_ring = ring_node_indices[1]
        for j in range(num_theta_points):
            a = ring_node_indices[0][0]
            b = first_ring[j]
            c = first_ring[(j + 1) % num_theta_points]
            base_triangles.append((a, b, c))

        # Connect each annular strip with two triangles per angular sector.
        for ring_index in range(1, num_radial_layers - 1):
            inner_ring = ring_node_indices[ring_index]
            outer_ring = ring_node_indices[ring_index + 1]
            for j in range(num_theta_points):
                a = inner_ring[j]
                b = outer_ring[j]
                c = outer_ring[(j + 1) % num_theta_points]
                d = inner_ring[(j + 1) % num_theta_points]
                base_triangles.append((a, b, c))
                base_triangles.append((a, c, d))

        outer_ring = ring_node_indices[-1]
        for j in range(num_theta_points):
            boundary_edges.append((outer_ring[j], outer_ring[(j + 1) % num_theta_points]))

        layer_node_count = len(xy_nodes_arr)
        nodes: list[np.ndarray] = []
        for z in z_layers:
            layer = np.column_stack(
                (
                    xy_nodes_arr[:, 0],
                    xy_nodes_arr[:, 1],
                    np.full(layer_node_count, z, dtype=float),
                )
            )
            nodes.append(layer)
        nodes_m = np.vstack(nodes) * 1.0e-6

        tetrahedra: list[tuple[int, int, int, int]] = []
        boundary_faces: list[tuple[int, int, int]] = []
        boundary_kinds: list[str] = []
        for layer_index in range(num_axial_layers - 1):
            bottom_offset = layer_index * layer_node_count
            top_offset = (layer_index + 1) * layer_node_count

            for tri in base_triangles:
                a, b, c = tri
                a0, b0, c0 = bottom_offset + a, bottom_offset + b, bottom_offset + c
                a1, b1, c1 = top_offset + a, top_offset + b, top_offset + c

                # Split each triangular prism into three tetrahedra so the
                # finite-element solver gets a simple, explicit connectivity.
                tetrahedra.extend(
                    [
                        (a0, b0, c0, c1),
                        (a0, b0, b1, c1),
                        (a0, a1, b1, c1),
                    ]
                )

            # Sidewall quads are split into two triangles per axial slab.
            for u, v in boundary_edges:
                u0, v0 = bottom_offset + u, bottom_offset + v
                u1, v1 = top_offset + u, top_offset + v
                # Tag the cylindrical wall separately from the end caps so the
                # solver can apply different anchoring modes patch-by-patch.
                boundary_faces.append((u0, v0, v1))
                boundary_kinds.append("sidewall")
                boundary_faces.append((u0, v1, u1))
                boundary_kinds.append("sidewall")

        # Caps are tagged explicitly so the solver can distinguish top and
        # bottom surfaces from the cylindrical wall.
        for tri in base_triangles:
            a, b, c = tri
            # The base triangle orientation is preserved on both caps; the
            # solver flips normals as needed when it loads the mesh.
            boundary_faces.append((a, b, c))
            boundary_kinds.append("bottom_cap")
            a_top, b_top, c_top = (
                (num_axial_layers - 1) * layer_node_count + a,
                (num_axial_layers - 1) * layer_node_count + b,
                (num_axial_layers - 1) * layer_node_count + c,
            )
            boundary_faces.append((a_top, b_top, c_top))
            boundary_kinds.append("top_cap")

        tetrahedra_arr = np.asarray(tetrahedra, dtype=int)
        boundary_faces_arr = np.asarray(boundary_faces, dtype=int)
        boundary_kinds_arr = np.asarray(boundary_kinds, dtype="<U16")

        volumes = np.array([self._tetra_volume(nodes_m[list(tet)]) for tet in tetrahedra_arr], dtype=float)
        if np.any(volumes <= 0.0):
            raise ValueError("Structured solve mesh produced a non-positive tetrahedral volume.")

        return StructuredCylinderSolveMesh(
            nodes_m=nodes_m,
            tetrahedra=tetrahedra_arr,
            boundary_faces=boundary_faces_arr,
            boundary_face_kind=boundary_kinds_arr,
            metadata_json=(
                "{"
                f"\"diameter_um\": {self.diameter}, "
                f"\"length_um\": {self.length}, "
                f"\"num_radial_layers\": {num_radial_layers}, "
                f"\"num_theta_points\": {num_theta_points}, "
                f"\"num_axial_layers\": {num_axial_layers}, "
                f"\"radial_cluster_power\": {radial_cluster_power}, "
                f"\"axial_cluster_power\": {axial_cluster_power}"
                "}"
            ),
        )

    def save_structured_solve_mesh(
        self,
        filename: str,
        num_radial_layers: int = 5,
        num_theta_points: int = 24,
        num_axial_layers: int = 12,
        radial_cluster_power: float = 2.0,
        axial_cluster_power: float = 2.0,
    ) -> StructuredCylinderSolveMesh:
        """Generate and save the body-fitted solve mesh to an NPZ file."""
        mesh = self.generate_structured_solve_mesh(
            num_radial_layers=num_radial_layers,
            num_theta_points=num_theta_points,
            num_axial_layers=num_axial_layers,
            radial_cluster_power=radial_cluster_power,
            axial_cluster_power=axial_cluster_power,
        )
        np.savez_compressed(
            filename,
            nodes_m=mesh.nodes_m,
            tetrahedra=mesh.tetrahedra,
            boundary_faces=mesh.boundary_faces,
            boundary_face_kind=mesh.boundary_face_kind,
            units=mesh.units,
            metadata_json=mesh.metadata_json,
        )
        return mesh

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
