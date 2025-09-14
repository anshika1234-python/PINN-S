# data_setup.py

# data_setup.py

import torch

def create_training_points(device, x_domain, t_domain, n_initial, n_boundary, n_physics):
    """
    Generates training points for the PINN.
    
    Args:
        device (torch.device): The device to store tensors on (CPU or GPU).
        x_domain (list): The spatial domain [x_min, x_max].
        t_domain (list): The temporal domain [t_min, t_max].
        n_initial (int): Number of points for the initial condition.
        n_boundary (int): Number of points for the boundary conditions.
        n_physics (int): Number of collocation points for enforcing physics.
        
    Returns:
        dict: A dictionary containing all training point tensors.
    """
    
    # --- 1. Initial Condition Points (t=0) ---
    
    # CORRECT ORDER: First, create the initial x and t points
    x_initial = torch.linspace(x_domain[0], x_domain[1], n_initial, device=device).view(-1, 1)
    t_initial = torch.zeros_like(x_initial, device=device)
    
    # Second, use x_initial to define the Gaussian pulse
    # Ey(x, 0) = exp(-20 * x^2)
    ey_initial_true = torch.exp(-20.0 * x_initial**2)
    bz_initial_true = torch.exp(-20.0 * x_initial**2) # For a simple pulse
    initial_data_true = torch.cat([ey_initial_true, bz_initial_true], dim=1)
    
    # NOTE: The old sine wave code has been removed.

    # --- 2. Boundary Condition Points (x=-1 and x=1) ---
    t_boundary = torch.linspace(t_domain[0], t_domain[1], n_boundary, device=device).view(-1, 1)
    x_boundary_left = torch.full_like(t_boundary, x_domain[0], device=device)
    x_boundary_right = torch.full_like(t_boundary, x_domain[1], device=device)

    # --- 3. Collocation Points (for enforcing physics) ---
    t_physics = torch.empty(n_physics, 1, device=device).uniform_(t_domain[0], t_domain[1])
    x_physics = torch.empty(n_physics, 1, device=device).uniform_(x_domain[0], x_domain[1])
    
    t_physics.requires_grad = True
    x_physics.requires_grad = True
    
    return {
        't_initial': t_initial,
        'x_initial': x_initial,
        'initial_data_true': initial_data_true,
        't_boundary': t_boundary,
        'x_boundary_left': x_boundary_left,
        'x_boundary_right': x_boundary_right,
        't_physics': t_physics,
        'x_physics': x_physics
    }