# main.py

import torch
from model import PINN
from data_setup import create_training_points
from training import train_model
from visualization import plot_results, calculate_l2_error

def main():
    # --- 1. Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    X_DOMAIN = [-1.0, 1.0]
    T_DOMAIN = [0.0, 1.0]
    N_INITIAL = 200
    N_BOUNDARY = 100
    N_PHYSICS = 10000

    # --- 2. Setup ---
    points = create_training_points(DEVICE, X_DOMAIN, T_DOMAIN, N_INITIAL, N_BOUNDARY, N_PHYSICS)
    pinn_model = PINN().to(DEVICE)
    
    # --- 3. Two-Phase Training ---

    # == PHASE 1: Fit the initial shape ==
    print("\n--- Starting Training Phase 1: Fitting Initial Condition ---")
    optimizer_p1 = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
    epochs_p1 = 20000
    # Weights for Phase 1: Focus ONLY on initial and boundary data.
    lambdas_p1 = [1.0, 1.0, 0.0] # (initial, boundary, physics)
    train_model(pinn_model, points, optimizer_p1, epochs_p1, DEVICE, lambdas_p1, start_epoch=1)

    # == PHASE 2: Learn the physics (annealing) ==
    print("\n--- Starting Training Phase 2: Learning Physics ---")
    optimizer_p2 = torch.optim.Adam(pinn_model.parameters(), lr=1e-4) # Lower learning rate for fine-tuning
    epochs_p2 = 30000 # Train for more epochs
    # Weights for Phase 2: Now, focus heavily on the physics.
    lambdas_p2 = [1.0, 1.0, 100.0] # (initial, boundary, physics)
    train_model(pinn_model, points, optimizer_p2, epochs_p2, DEVICE, lambdas_p2, start_epoch=epochs_p1 + 1)
    
    print("\nTraining finished.")

    # --- 4. Validation and Visualization ---
    # NOTE: L2 error is still for the old sin wave problem. Focus on the visual animation.
    print("Generating visualization...")
    plot_results(pinn_model, X_DOMAIN, T_DOMAIN, DEVICE, filename="em_wave_animation.gif")
    print("Project finished successfully!")

if __name__ == "__main__":
    main()