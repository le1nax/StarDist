import numpy as np

def reslice_to_top_view(stack):
    """
    Reorients a 3D OCT stack to display the top view.

    Parameters:
    stack (numpy.ndarray): A 3D array representing the OCT stack, 
                           with dimensions (slices, height, width).

    Returns:
    numpy.ndarray: A 3D array representing the reoriented stack.
    """
    # Transpose the stack to swap axes, assuming input format (slices, height, width)
    top_view_stack = np.transpose(stack, (1, 0, 2))
    return top_view_stack

def reslice_to_90_degrees_view(stack, axis='x'):
    """
    Rotates a 3D OCT stack to view it from 90 degrees along a specified axis.

    Parameters:
    stack (numpy.ndarray): A 3D array representing the OCT stack, 
                           with dimensions (slices, height, width).
    axis (str): The axis to rotate the stack around. Options are:
                'x' - Rotate to view along the X-axis.
                'y' - Rotate to view along the Y-axis.
                'z' - Rotate to view along the Z-axis.

    Returns:
    numpy.ndarray: A 3D array representing the rotated stack.
    """
    if axis == 'x':
        # Rotate to view along the X-axis: swap height and slices
        rotated_stack = np.transpose(stack, (2, 1, 0))
    elif axis == 'y':
        # Rotate to view along the Y-axis: swap width and slices
        rotated_stack = np.transpose(stack, (1, 2, 0))
    elif axis == 'z':
        # Rotate to view along the Z-axis (default orientation): no change
        rotated_stack = stack
    else:
        raise ValueError("Invalid axis specified. Choose 'x', 'y', or 'z'.")
    
    return rotated_stack