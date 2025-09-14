# training.py

from loss_functions import calculate_loss

def train_model(model, points, optimizer, epochs, device, lambdas, start_epoch=1):
    """
    The main training loop for the PINN, now accepting loss weights.
    """
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss, loss_i, loss_b, loss_p = calculate_loss(model, points, device, lambdas)
        
        loss.backward()
        optimizer.step()
        
        # We use start_epoch to keep the printout continuous
        current_epoch = start_epoch + epoch
        if (current_epoch) % 1000 == 0:
            print(f'Epoch [{current_epoch}], Total Loss: {loss.item():.6f} | '
                  f'Initial: {loss_i.item():.6f}, Boundary: {loss_b.item():.6f}, Physics: {loss_p.item():.6f}')