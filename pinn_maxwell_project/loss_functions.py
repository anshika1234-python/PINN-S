# loss_functions.py

# loss_functions.py

import torch

def calculate_loss(model, points, device, lambdas):
    """
    Calculates the total loss for the PINN with given weights.
    """
    # Unpack weights
    lambda_initial, lambda_boundary, lambda_physics = lambdas

    # Unpack points
    t_initial, x_initial, initial_data_true = points['t_initial'], points['x_initial'], points['initial_data_true']
    t_boundary, x_boundary_left, x_boundary_right = points['t_boundary'], points['x_boundary_left'], points['x_boundary_right']
    t_physics, x_physics = points['t_physics'], points['x_physics']
    
    # Initial & Boundary Loss
    initial_pred = model(t_initial, x_initial)
    loss_initial = torch.mean((initial_pred - initial_data_true)**2)
    boundary_pred_left = model(t_boundary, x_boundary_left)
    boundary_pred_right = model(t_boundary, x_boundary_right)
    loss_boundary = torch.mean((boundary_pred_left - boundary_pred_right)**2)

    # Physics Loss
    if lambda_physics > 0:
        physics_pred = model(t_physics, x_physics)
        ey_pred = physics_pred[:, 0].view(-1, 1)
        bz_pred = physics_pred[:, 1].view(-1, 1)
        dEy_dt = torch.autograd.grad(ey_pred, t_physics, grad_outputs=torch.ones_like(ey_pred), create_graph=True)[0]
        dEy_dx = torch.autograd.grad(ey_pred, x_physics, grad_outputs=torch.ones_like(ey_pred), create_graph=True)[0]
        dBz_dt = torch.autograd.grad(bz_pred, t_physics, grad_outputs=torch.ones_like(bz_pred), create_graph=True)[0]
        dBz_dx = torch.autograd.grad(bz_pred, x_physics, grad_outputs=torch.ones_like(bz_pred), create_graph=True)[0]
        residual_1 = dEy_dx + dBz_dt
        residual_2 = dBz_dx + dEy_dt
        loss_physics = torch.mean(residual_1**2) + torch.mean(residual_2**2)
    else:
        loss_physics = torch.tensor(0.0, device=device)

    # Combine losses
    total_loss = (lambda_initial * loss_initial + 
                  lambda_boundary * loss_boundary + 
                  lambda_physics * loss_physics)
    
    return total_loss, loss_initial, loss_boundary, loss_physics