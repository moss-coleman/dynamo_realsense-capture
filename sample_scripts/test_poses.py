import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Define the vertices of a cube ---
# This cube is centered at the origin with side length 2
cube_vertices = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]
])

# --- Define the faces of the cube by referencing vertex indices ---
# Each list of 4 numbers represents the indices of the vertices forming a face.
cube_faces = [
    [cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3]], # Bottom face
    [cube_vertices[4], cube_vertices[5], cube_vertices[6], cube_vertices[7]], # Top face
    [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]], # Front face
    [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]], # Back face
    [cube_vertices[1], cube_vertices[2], cube_vertices[6], cube_vertices[5]], # Right face
    [cube_vertices[0], cube_vertices[3], cube_vertices[7], cube_vertices[4]]  # Left face
]

# --- Rotation Matrix Functions ---

def rotation_matrix_x(angle_degrees):
    """Returns the rotation matrix for a given angle (in degrees) around the X-axis."""
    angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    return np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])

def rotation_matrix_y(angle_degrees):
    """Returns the rotation matrix for a given angle (in degrees) around the Y-axis."""
    angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

def rotation_matrix_z(angle_degrees):
    """Returns the rotation matrix for a given angle (in degrees) around the Z-axis."""
    angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

# --- Function to apply transformation ---

def transform_vertices(vertices, matrix):
    """Applies a transformation matrix to a set of vertices."""
    # For each vertex, we multiply it by the rotation matrix.
    # np.dot is used for matrix multiplication.
    # We transpose the vertices to be (3, N) for batch multiplication with a (3, 3) matrix,
    # then transpose back to (N, 3).
    return np.dot(vertices, matrix.T) # Transpose matrix for correct multiplication order if vertices are (N,3)

# --- Visualization Function ---

def plot_cube(ax, vertices, faces, color='blue', alpha=0.5, label='Object'):
    """Plots a 3D cube given its vertices and faces."""
    poly3d = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='r', alpha=alpha)
    ax.add_collection3d(poly3d)
    # Set plot limits and labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    # Add a legend
    if label:
        # Create a dummy artist for the legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, edgecolor='r', label=label, alpha=alpha)]
        ax.legend(handles=legend_elements)

# --- Main Script ---

if __name__ == "__main__":
    fig = plt.figure(figsize=(18, 6))

    # 1. Original Cube
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Original Cube')
    plot_cube(ax1, cube_vertices, cube_faces, color='cyan', label='Original')
    ax1.view_init(elev=20., azim=-35) # Set initial view angle

    # 2. Cube rotated around X-axis
    angle_x = 45  # degrees
    R_x = rotation_matrix_x(angle_x)
    rotated_vertices_x = transform_vertices(cube_vertices, R_x)
    rotated_faces_x = [
        [rotated_vertices_x[0], rotated_vertices_x[1], rotated_vertices_x[2], rotated_vertices_x[3]],
        [rotated_vertices_x[4], rotated_vertices_x[5], rotated_vertices_x[6], rotated_vertices_x[7]],
        [rotated_vertices_x[0], rotated_vertices_x[1], rotated_vertices_x[5], rotated_vertices_x[4]],
        [rotated_vertices_x[2], rotated_vertices_x[3], rotated_vertices_x[7], rotated_vertices_x[6]],
        [rotated_vertices_x[1], rotated_vertices_x[2], rotated_vertices_x[6], rotated_vertices_x[5]],
        [rotated_vertices_x[0], rotated_vertices_x[3], rotated_vertices_x[7], rotated_vertices_x[4]]
    ]
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f'Rotated {angle_x}° around X-axis')
    plot_cube(ax2, rotated_vertices_x, rotated_faces_x, color='magenta', label=f'Rotated X ({angle_x}°)')
    ax2.view_init(elev=20., azim=-35)

    # 3. Cube rotated around Y-axis then Z-axis
    angle_y = 30  # degrees
    angle_z = 60  # degrees
    R_y = rotation_matrix_y(angle_y)
    R_z = rotation_matrix_z(angle_z)

    # Apply Y rotation first, then Z rotation
    # Note: Matrix multiplication order matters! R_z @ R_y means apply R_y then R_z.
    # If you want to apply transformations in a global coordinate system, you multiply R_global * R_local.
    # If you want to apply transformations in a local (object's own) coordinate system, you multiply R_local * R_global.
    # Here, we're applying R_y to the original vertices, then R_z to the result of that.
    # So, transformed_vertices = (R_z @ R_y) @ original_vertices.T  (if vertices are column vectors)
    # Or, transformed_vertices = original_vertices @ R_y.T @ R_z.T (if vertices are row vectors as in our case)

    # Step 1: Rotate around Y
    temp_vertices = transform_vertices(cube_vertices, R_y)
    # Step 2: Rotate the result around Z
    rotated_vertices_yz = transform_vertices(temp_vertices, R_z)

    # Alternatively, combine matrices first: R_combined = R_z @ R_y (for column vectors)
    # For row vectors (N,3) as we use: R_combined_row_vectors = R_y @ R_z
    # Then apply: rotated_vertices_yz = transform_vertices(cube_vertices, R_combined_row_vectors)
    # Let's use the combined matrix approach for row vectors
    R_combined = R_y @ R_z # This means R_y is applied, then R_z to that result in the object's new local frame
                            # Or, if thinking globally, it's a rotation by R_z then by R_y around fixed global axes.
                            # To be precise for sequential local rotations (rotate by Y, then by Z around *new* Z axis):
                            # For row vectors: V_new = V_old @ Ry @ Rz
    combined_rotation_matrix = R_y @ R_z
    rotated_vertices_yz_combined = transform_vertices(cube_vertices, combined_rotation_matrix)


    rotated_faces_yz = [
        [rotated_vertices_yz_combined[0], rotated_vertices_yz_combined[1], rotated_vertices_yz_combined[2], rotated_vertices_yz_combined[3]],
        [rotated_vertices_yz_combined[4], rotated_vertices_yz_combined[5], rotated_vertices_yz_combined[6], rotated_vertices_yz_combined[7]],
        [rotated_vertices_yz_combined[0], rotated_vertices_yz_combined[1], rotated_vertices_yz_combined[5], rotated_vertices_yz_combined[4]],
        [rotated_vertices_yz_combined[2], rotated_vertices_yz_combined[3], rotated_vertices_yz_combined[7], rotated_vertices_yz_combined[6]],
        [rotated_vertices_yz_combined[1], rotated_vertices_yz_combined[2], rotated_vertices_yz_combined[6], rotated_vertices_yz_combined[5]],
        [rotated_vertices_yz_combined[0], rotated_vertices_yz_combined[3], rotated_vertices_yz_combined[7], rotated_vertices_yz_combined[4]]
    ]
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(f'Rotated {angle_y}° Y then {angle_z}° Z')
    plot_cube(ax3, rotated_vertices_yz_combined, rotated_faces_yz, color='green', label=f'Rotated Y({angle_y}°), Z({angle_z}°)')
    ax3.view_init(elev=20., azim=-35)

    plt.tight_layout()
    plt.show()
