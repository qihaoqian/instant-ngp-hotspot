import matplotlib.pyplot as plt
import numpy as np
from hotspot.utils import extract_fields
import trimesh

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # range in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb

def plot_results_separately(results):
    """
    Visualize surface points and free space points of the dataset.get_item separately using trimesh.

    Parameters:
      results: dict with keys 'points_surf', 'sdfs_surf', 'points_free', 'sdfs_free'.
    """
    import trimesh

    # Extract data for surface points
    points_surf = results['points_surf']  # [N, 3]
    sdfs_surf = results['sdfs_surf']        # [N, 1]
    
    # Extract data for free space points
    points_free = results['points_free']    # [M, 3]
    sdfs_free = results['sdfs_free']          # [M, 1]

    # Map SDF values to colors (assuming map_color is defined elsewhere)
    colors_surf = map_color(sdfs_surf.squeeze(1))
    colors_free = map_color(sdfs_free.squeeze(1))

    # Construct trimesh point cloud objects
    pc_surf = trimesh.PointCloud(points_surf, colors_surf)
    pc_free = trimesh.PointCloud(points_free, colors_free)

    # Create separate scenes for each point cloud
    scene_surf = trimesh.Scene([pc_surf])
    scene_free = trimesh.Scene([pc_free])

    # Display the scenes independently
    scene_surf.show(title="Surface Points")
    scene_free.show(title="Free Space Points")
    
def visualize_cross_section(mesh, plane_normal=np.array([0, 1, 0]), plane_origin=None):
    """
    Extract and visualize a cross-section of the mesh, such as taking the xy-plane 
    (z-axis as the normal vector) at the middle of the z-axis.

    Parameters:
      mesh: trimesh.Trimesh object
      plane_normal: Normal vector of the plane, default is the y-axis direction [0,1,0], 
                    representing the xz-plane.
      plane_origin: A point on the plane. If None, the center of the mesh's bounding box 
                    (i.e., the middle value of the z-coordinate) is used by default.
    """
    # If the plane origin is not specified, use the center of the mesh's bounding box
    if plane_origin is None:
        plane_origin = mesh.bounds.mean(axis=0)
    
    # Compute the intersection line (cross-section) with the plane. 
    # The section method returns a 2D path.
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        print("No cross-section found on the specified plane.")
        return
    
    # Convert the cross-section path to a 3D curve in space
    section_3d = section.to_3D()
    
    # Create a scene and add the original mesh and the cross-section curve
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="Mesh")
    scene.add_geometry(section_3d, geom_name="Cross Section")
    
    # Display the scene
    # scene.show(title="Mesh Cross Section")
    image_data = scene.save_image(resolution=(1024, 768), visible=True)
    
    # Write the image data to a file
    image_path = "cross_section.png"
    with open(image_path, "wb") as f:
        f.write(image_data)
    print(f"Image saved to {image_path}")
    
def plot_sdf_points(points, sdfs, title='SDF Visualization', cmap='coolwarm'):
    """
    Visualize SDF values in 3D space using a scatter plot.

    Parameters:
      points: (N, 3) array of point coordinates.
      sdfs: (N,) or (N, 1) array of SDF values.
      title: Title of the plot.
      cmap: Colormap to use for visualizing SDF values.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the SDF values
    sdfs = sdfs.flatten()

    # Use scatter to plot 3D points with colors based on SDF values
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=sdfs, cmap=cmap, s=5)

    # Add a colorbar to show the SDF value scale
    plt.colorbar(sc, label='SDF Value')
    ax.set_title(title)
    plt.show()
    
