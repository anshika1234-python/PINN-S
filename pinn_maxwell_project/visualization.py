# visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_results(model, x_domain, t_domain, device, filename="em_wave_animation.gif"):
    """
    Visualizes the trained PINN solution as an animation.
    
    Args:
        model (torch.nn.Module): The trained PINN model.
        x_domain (list): The spatial domain.
        t_domain (list): The temporal domain.
        device (torch.device): The device for computation.
        filename (str): The name of the file to save the animation.
    """
    model.eval() # Set the model to evaluation mode
    
    # Create a grid of points for plotting
    x = torch.linspace(x_domain[0], x_domain[1], 200).to(device)
    t_steps = torch.linspace(t_domain[0], t_domain[1], 100).to(device)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Field Amplitude")
    ax.set_title("PINN Solution of 1D Maxwell's Equations")
    ax.set_xlim(x_domain)
    ax.set_ylim([-1.5, 1.5])
    
    line_ey, = ax.plot([], [], lw=2, label="E_y (Electric Field)")
    line_bz, = ax.plot([], [], lw=2, label="B_z (Magnetic Field)")
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    def init():
        line_ey.set_data([], [])
        line_bz.set_data([], [])
        time_text.set_text('')
        return line_ey, line_bz, time_text

    def animate(i):
        t = t_steps[i]
        t_tensor = torch.full_like(x, t).view(-1, 1)
        x_tensor = x.view(-1, 1)
        
        with torch.no_grad():
            pred = model(t_tensor, x_tensor)
        
        ey_pred = pred[:, 0].cpu().numpy()
        bz_pred = pred[:, 1].cpu().numpy()
        
        line_ey.set_data(x.cpu().numpy(), ey_pred)
        line_bz.set_data(x.cpu().numpy(), bz_pred)
        time_text.set_text(f'Time = {t.item():.2f} s')
        return line_ey, line_bz, time_text

    ani = animation.FuncAnimation(fig, animate, frames=len(t_steps),
                                  init_func=init, blit=True, interval=50)
    
    print(f"Saving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=20)
    print("Animation saved.")
    plt.close(fig)


def calculate_l2_error(model, device):
    """Calculates the L2 relative error against the analytical solution."""
    model.eval()
    
    # Generate a fine grid of test points
    x_test = torch.linspace(-1.0, 1.0, 500, device=device).view(-1, 1)
    t_test = torch.linspace(0.0, 1.0, 500, device=device).view(-1, 1)
    
    # Create a meshgrid
    X, T = torch.meshgrid(x_test.squeeze(1), t_test.squeeze(1), indexing='xy')
    
    # Get model predictions
    with torch.no_grad():
        pred = model(T.flatten().view(-1, 1), X.flatten().view(-1, 1))
    ey_pred = pred[:, 0]
    
    # Get analytical solution
    ey_true = torch.sin(torch.pi * X.flatten()) * torch.cos(torch.pi * T.flatten())
    
    # Calculate L2 relative error
    l2_error = torch.linalg.norm(ey_pred - ey_true) / torch.linalg.norm(ey_true)
    
    print(f"L2 Relative Error: {l2_error.item():.6f}")
    return l2_error.item()