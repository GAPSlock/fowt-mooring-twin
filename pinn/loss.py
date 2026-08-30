import torch
import torch.nn.functional as F

def compute_data_loss(pred_tensions, true_tensions):
    """
    Standard supervised MSE loss against MoorDyn ground truth.
    """
    return F.mse_loss(pred_tensions, true_tensions)

def compute_physics_loss(pred_tensions, catenary_tensions, velocities, accelerations, velocity_threshold=0.01):
    """
    Custom Physics Loss for Mooring Lines.
    
    1. Tension Positivity: T >= 0 (Cables cannot bear compression)
    2. Quasi-Static Consistency: When v, a -> 0, T_dynamic -> T_catenary
    """
    # 1. Positivity Constraint
    # We penalize any predicted tension that is less than 0
    # ReLU(-T) is positive only when T is negative.
    neg_penalty = F.relu(-pred_tensions)
    loss_positivity = torch.mean(neg_penalty ** 2)
    
    # 2. Quasi-Static Consistency
    # We want to pull the network towards the catenary solution ONLY when 
    # the platform is moving very slowly (low dynamic effects).
    
    # Calculate magnitude of velocity for each timestep
    # velocities is shape (batch_size, 6)
    v_mag = torch.norm(velocities[:, :3], dim=1) # Just translational velocity magnitude
    
    # Mask: 1.0 where velocity is near zero, 0.0 otherwise
    static_mask = (v_mag < velocity_threshold).float().unsqueeze(1)
    
    if static_mask.sum() > 0:
        # Compute MSE against catenary only for quasi-static points
        static_mse = F.mse_loss(pred_tensions * static_mask, catenary_tensions * static_mask, reduction='sum')
        loss_static = static_mse / static_mask.sum()
    else:
        loss_static = torch.tensor(0.0, device=pred_tensions.device)
        
    return loss_positivity + 0.1 * loss_static
