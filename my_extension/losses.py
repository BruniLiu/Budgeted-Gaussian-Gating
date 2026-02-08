import torch


def gating_sparsity_loss(scores):
    """
    Implements the sparsity term  Σ w_i  in Eq.(4) of the paper.

    scores: (N, 1) tensor in [0,1]
    return: scalar sparsity loss
    """
    return scores.mean()


def gating_entropy_regularization(scores, eps=1e-6):
    """
    Implements a stability regularization term L_reg
    to encourage confident decisions (near 0 or 1).

    This corresponds to the γ L_reg term in Eq.(4).

    scores: (N, 1) tensor in [0,1]
    return: scalar entropy regularization loss
    """
    p = torch.clamp(scores, eps, 1 - eps)
    entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy.mean()


def gating_l2_regularization(scores):
    """
    Optional alternative regularization:
    Penalize overly large gate activations.

    Useful for smoother training.
    """
    return (scores ** 2).mean()


def total_gating_loss(scores, lambda_sparse=1.0, gamma_reg=0.1, reg_type="entropy"):
    """
    Improved gating loss with budget constraint instead of naive sparsity.

    New formulation:

        L_gate = λ * (mean(w) - rho)^2 + γ * L_reg

    This avoids collapse-to-zero behavior.
    """

  
    rho = 0.3   

    mean_w = scores.mean()
    loss_budget = (mean_w - rho) ** 2

    
    if reg_type == "entropy":
        loss_reg = gating_entropy_regularization(scores)
    elif reg_type == "l2":
        loss_reg = gating_l2_regularization(scores)
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")

    return lambda_sparse * loss_budget + gamma_reg * loss_reg

