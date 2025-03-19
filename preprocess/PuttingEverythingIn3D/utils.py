import numpy as np

def select_anticipation_window(df, timestamp_col='timestamp', window_size=60., anticipation_time=5.):
    # Anticipation time is the tau_a in AVT, Rohit Girdhar et al.
    # Function to select a random 1-minute window that doesn't extend beyond the timestamp
    subset_dict = {}
    
    for index, row in df.iterrows():
        timestamp = row[timestamp_col]

        # We need minimum anticipation_time margin before we can start anticipation
        if timestamp < anticipation_time:
            continue
        
        # Define window start and end based on the random offset
        window_start = max(timestamp - anticipation_time, 0)
        window_end = window_start + window_size
        
        # Select rows where the timestamp lies within the 1-minute window
        window_df = df[(df[timestamp_col] >= window_start) & (df[timestamp_col] <= window_end)]

        # Expand the dataframe to have a new row for each object
        window_df_expanded = window_df.set_index(window_df.columns.difference(['object'], sort=False).tolist())['object'].str.split('@@@@@', expand=True).stack().reset_index(level=-1, drop=True).reset_index().rename(columns={0:'object'})

        # Keep only the first interaction
        window_df_sorted = window_df_expanded.sort_values(by=['object', 'timestamp'])
        window_df_unique = window_df_sorted.drop_duplicates(subset='object', keep='first')
        window_df_unique.reset_index(drop=True, inplace=True)
        
        # Append the resulting subset dataframe to the dictionary
        subset_dict[index] = window_df_unique
    
    return subset_dict

def is_point_in_obb(point, obb_vertices, tol=1e-6):
    """
    Determines if a 3D point is inside an oriented bounding box (OBB).

    Parameters:
    - point (np.ndarray): A numpy array of shape (3,) representing the (x, y, z) coordinates of the point.
    - obb_vertices (np.ndarray): A numpy array of shape (3, 8) representing the OBB vertices.
    - tol (float): Tolerance for numerical comparisons.

    Returns:
    - inside (bool): True if the point is inside the OBB, False otherwise.
    """
    # Transpose vertices to shape (8, 3) for easier indexing
    vertices = obb_vertices.T  # Now vertices is (8, 3)

    # Step 1: Compute the center of the OBB
    center = np.mean(vertices, axis=0)

    # Step 2: Compute the local axes of the OBB using edges from vertex 0
    u = vertices[1] - vertices[0]  # Edge from vertex 0 to vertex 1
    v = vertices[3] - vertices[0]  # Edge from vertex 0 to vertex 3
    w = vertices[4] - vertices[0]  # Edge from vertex 0 to vertex 4

    # Normalize the axes to get unit vectors
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    w_norm = w / np.linalg.norm(w)

    # Orthogonality Check
    dot_uv = np.dot(u_norm, v_norm)
    dot_uw = np.dot(u_norm, w_norm)
    dot_vw = np.dot(v_norm, w_norm)

    if (abs(dot_uv) > tol) or (abs(dot_uw) > tol) or (abs(dot_vw) > tol):
        print("Warning: Axes are not orthogonal. Check the vertex ordering.")
        # Optionally, you can attempt to orthogonalize the axes here or return False.
        # For now, we'll proceed, but the results may not be accurate.

    # Step 3: Compute the extents (half-sizes) along each axis
    extent_u = np.linalg.norm(u) / 2.0
    extent_v = np.linalg.norm(v) / 2.0
    extent_w = np.linalg.norm(w) / 2.0

    # Step 4: Compute the vector from the OBB center to the point
    d = point - center

    # Project the vector onto each local axis
    proj_u = np.dot(d, u_norm)
    proj_v = np.dot(d, v_norm)
    proj_w = np.dot(d, w_norm)

    # Step 5: Check if the projections are within the extents
    if (abs(proj_u) <= extent_u + tol) and (abs(proj_v) <= extent_v + tol) and (abs(proj_w) <= extent_w + tol):
        return True  # The point is inside the OBB
    else:
        return False  # The point is outside the OBB