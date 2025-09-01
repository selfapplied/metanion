"""Generate a simple Metanion field diagram.

The diagram visualizes the Boolean cube for three transforms with
alpha as vertical axis, color indicating ``heartspace`` potential,
and quaternion orientation as arrows.
"""
from __future__ import annotations

import itertools
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

Quaternion = Tuple[float, float, float, float]


def qmul(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Quaternion multiplication (Hamilton product)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def qconj(q: Quaternion) -> Quaternion:
    """Quaternion conjugate."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def qnorm(q: Quaternion) -> Quaternion:
    """Normalize a quaternion to unit length."""
    w, x, y, z = q
    mag = (w * w + x * x + y * y + z * z) ** 0.5
    if mag == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / mag, x / mag, y / mag, z / mag)


def quat_to_vec(q: Quaternion) -> np.ndarray:
    """Map quaternion to a 3D vector for visualization.

    This rotates the x-axis (1,0,0) by the unit quaternion ``q``.
    """
    w, x, y, z = q
    return np.array([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ])


def field_diagram(weights: Sequence[float], quats: Sequence[Quaternion], *, filename: str | None = None) -> None:
    """Plot the field diagram for three transforms.

    Parameters
    ----------
    weights:
        Length-3 sequence of positive weights ``w_i``.
    quats:
        Length-3 sequence of unit quaternions ``U_i``.
    filename:
        Optional file path to save the figure instead of displaying it.
    """
    if len(weights) != 3 or len(quats) != 3:
        raise ValueError("This diagram is defined for exactly three transforms")

    total_w = sum(weights)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create energy-based shading (red=high energy, blue=low energy)
    # Calculate energy for each state based on bitmask
    energies = []
    for m in itertools.product([0, 1], repeat=3):
        # Energy increases with number of 1s (more active transforms = higher energy)
        energy = sum(m) / 3.0  # Normalize to [0,1]
        energies.append(energy)

    # Create cube edges first
    edges = [
        # Bottom face (z=0)
        [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)], [
            (1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],
        # Top face (z=1)
        [(0, 0, 1), (1, 0, 1)], [(1, 0, 1), (1, 1, 1)], [
            (1, 1, 1), (0, 1, 1)], [(0, 1, 1), (0, 0, 1)],
        # Vertical edges
        [(0, 0, 0), (0, 0, 1)], [(1, 0, 0), (1, 0, 1)], [
            (1, 1, 0), (1, 1, 1)], [(0, 1, 0), (0, 1, 1)]
    ]

    # Draw cube faces with energy-based shading
    faces = [
        # Bottom face (z=0)
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        # Top face (z=1)
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        # Side faces
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],  # left
        [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)],  # right
        [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],  # front
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]   # back
    ]

    # Calculate face colors based on average energy of vertices
    for face in faces:
        # Calculate average energy of the face vertices
        face_energies = []
        face_alphas = []
        for vertex in face:
            vertex_energy = sum(vertex) / 3.0
            vertex_alpha = sum((1 - mi) * wi for mi,
                               wi in zip(vertex, weights)) / total_w
            face_energies.append(vertex_energy)
            face_alphas.append(vertex_alpha)

        avg_energy = sum(face_energies) / len(face_energies)

        # Create face color based on average energy
        face_color = (avg_energy, 0.2, 1 - avg_energy)  # RGB: red to blue

        # Extract coordinates for the face
        x_coords = [v[0] for v in face]
        y_coords = [v[1] for v in face]
        z_coords = face_alphas

        # Draw the face with shading - all cube faces are rectangular
        # Create a 2x2 grid for each face
        x_grid = np.array(x_coords).reshape(2, 2)
        y_grid = np.array(y_coords).reshape(2, 2)
        z_grid = np.array(z_coords).reshape(2, 2)
        
        # Use plot_surface for rectangular faces
        ax.plot_surface(x_grid, y_grid, z_grid, 
                       color=face_color, alpha=0.4, edgecolor='none')

    # Draw cube edges
    for edge in edges:
        start, end = edge
        start_alpha = sum((1 - mi) * wi for mi,
                          wi in zip(start, weights)) / total_w
        end_alpha = sum((1 - mi) * wi for mi,
                        wi in zip(end, weights)) / total_w
        ax.plot([start[0], end[0]], [start[1], end[1]], [start_alpha, end_alpha],
                'k-', linewidth=2, alpha=0.8)

    # Draw vertices with energy-based shading
    for i, m in enumerate(itertools.product([0, 1], repeat=3)):
        alpha = sum((1 - mi) * wi for mi, wi in zip(m, weights)) / total_w
        energy = energies[i]

        # Compose quaternions according to bitmask; 0 means inverse
        Q = (1.0, 0.0, 0.0, 0.0)
        for mi, Ui in zip(m, quats):
            q = Ui if mi else qconj(Ui)
            Q = qnorm(qmul(Q, q))
        vec = quat_to_vec(Q)

        x, y, z = m

        # Energy-based color: red (high) to blue (low)
        color_intensity = energy
        color = (color_intensity, 0.2, 1 - color_intensity)  # RGB: red to blue

        # Draw vertex with energy shading
        ax.scatter(x, y, alpha, color=color, s=300, alpha=0.9,
                   edgecolors='black', linewidth=2)

        # Quaternion orientation arrows
        ax.quiver(x, y, alpha, vec[0], vec[1], vec[2],
                  length=0.3, color="darkgreen", linewidth=2, alpha=0.8)

        # Add bitmask labels
        bitmask_str = f"{m[0]}{m[1]}{m[2]}"
        ax.text(x, y, alpha - 0.15, bitmask_str, fontsize=10, ha='center',
                fontweight='bold', color='white')

        # Add energy annotations
        ax.text(x + 0.15, y + 0.15, alpha + 0.1, f'E={energy:.2f}',
                fontsize=8, ha='center', color='black')

    # Enhance the plot appearance
    ax.set_xlabel("m₁", fontsize=14, fontweight='bold')
    ax.set_ylabel("m₂", fontsize=14, fontweight='bold')
    ax.set_zlabel(r"$\alpha$ (inflation)", fontsize=14, fontweight='bold')
    ax.set_title("Metanion Field: Boolean Cube with Energy Shading & Quaternion Orientations",
                 fontsize=16, fontweight='bold', pad=20)

    # Set better viewing angle and limits
    ax.view_init(elev=25, azim=45)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_zlim(-0.2, 1.2)

    # Add grid and styling
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Add colorbar for energy scale
    from matplotlib.cm import RdBu_r
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(cmap=RdBu_r, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Energy Level', fontsize=12, fontweight='bold')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low (Blue)', 'Medium', 'High (Red)'])

    if filename:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    # Create a more visually interesting diagram with varied weights and quaternions
    # that create a more dynamic energy landscape and orientation field
    # Varied weights create asymmetric energy landscape
    weights = [2.0, 1.5, 0.8]
    quats = [
        (0.707, 0.707, 0.0, 0.0),  # 90-degree rotation about x-axis
        (0.0, 0.707, 0.707, 0.0),  # 90-degree rotation about y-axis
        (0.5, 0.5, 0.5, 0.5),      # Complex rotation combining all axes
    ]
    field_diagram(weights, quats, filename="metanion_field_diagram.pdf")
    field_diagram(weights, quats, filename="metanion_field_diagram.png")
