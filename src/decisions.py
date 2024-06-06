"""
Functions for computing decision probabilities and making decisions.
"""

import numpy as np
from typing import Union


def _get_static_decision_prob(alpha: float) -> Union[float, np.ndarray]:
    """
    Decision probability under static (no) behavioral adaptation.
    """
    return 1 / (alpha + 1)


def _get_linear_decision_prob(
    alpha: float, z: np.ndarray, sum_fn: callable
) -> Union[float, np.ndarray]:
    """
    Decision probability under linear behavioral adaptation.

    Args:
        - alpha: rate parameter
        - z: treatment assignment vector
        - sum_fn: function to compute the assignment sum
    """
    return 1 / (sum_fn(z) * alpha + 1)


def _get_exp_decision_prob(
    alpha: float, z: np.ndarray, sum_fn: callable
) -> Union[float, np.ndarray]:
    """
    Decision probability under exponential behavioral adaptation.

    Args:
        - alpha: rate parameter
        - z: treatment assignment vector
        - sum_fn: function to compute the assignment sum
    """
    return 1 / (np.exp(alpha * sum_fn(z)) + 1)


def _get_step_decision_prob(
    alpha_pre: float, alpha_post: float, z: np.ndarray, l_t: float, sum_fn: callable
) -> Union[float, np.ndarray]:
    """
    Decision probability under step behavioral adaptation.

    Args:
        - alpha_pre: rate parameter before crossing threshold
        - alpha_post: rate parameter after crossing threshold
        - z: treatment assignment vector
        - l_t: threshold
        - sum_fn: function to compute the assignment sum
    """
    s = sum_fn(z)
    return (s < l_t) * (1 / (alpha_pre + 1)) + (s >= l_t) * (1 / (alpha_post + 1))


def get_decision_prob(
    z_vec: np.ndarray, adapt_type: str, **kwargs
) -> Union[float, np.ndarray]:
    """
    Get decision prob based on the adaptation type.

    Args:
        - z_vec: treatment assignment vector
        - adapt_type: adaptation type
        - kwargs: parameters for the adaptation type
    """

    if adapt_type == "static":
        phys_decision_prob = _get_static_decision_prob(kwargs["alpha"])
    elif adapt_type == "linear":
        phys_decision_prob = _get_linear_decision_prob(
            kwargs["alpha"], z_vec, kwargs["sum_fn"]
        )
    elif adapt_type == "exponential":
        phys_decision_prob = _get_exp_decision_prob(
            kwargs["alpha"], z_vec, kwargs["sum_fn"]
        )
    elif adapt_type == "step":
        phys_decision_prob = _get_step_decision_prob(
            kwargs["alpha_pre"],
            kwargs["alpha_post"],
            z_vec,
            kwargs["l_t"],
            kwargs["sum_fn"],
        )
    else:
        raise ValueError(f"Invalid adapt_type: {adapt_type}")

    return phys_decision_prob


def get_z0_decision_prob(
    z_vec: np.ndarray, adapt_type: str, **kwargs
) -> Union[float, np.ndarray]:
    """
    Get decision prob for individuals assigned to z=0 (control).

    Args:
        - z_vec: treatment assignment vector
        - adapt_type: adaptation type
        - kwargs: parameters for the adaptation type
    """
    if adapt_type == "step":
        return get_decision_prob(
            z_vec,
            adapt_type,
            alpha_pre=kwargs["alpha0_pre"],
            alpha_post=kwargs["alpha0_post"],
            **kwargs,
        )
    else:
        return get_decision_prob(z_vec, adapt_type, alpha=kwargs["alpha0"], **kwargs)


def get_z1_decision_prob(
    z_vec: np.ndarray, adapt_type: str, **kwargs
) -> Union[float, np.ndarray]:
    """
    Get decision prob for individuals assigned to z=1 (treatment).

    Args:
        - z_vec: treatment assignment vector
        - adapt_type: adaptation type
        - kwargs: parameters for the adaptation type
    """
    if adapt_type == "step":
        return get_decision_prob(
            z_vec,
            adapt_type,
            alpha_pre=kwargs["alpha1_pre"],
            alpha_post=kwargs["alpha1_post"],
            **kwargs,
        )
    else:
        return get_decision_prob(z_vec, adapt_type, alpha=kwargs["alpha1"], **kwargs)


def get_decisions(
    y0: np.ndarray, y1: np.ndarray, p_dict: dict, seed: float
) -> np.ndarray:
    """
    Get decisions based on individual effects and decision-making quality.

    Args:
        - y0: potential outcomes under decision 0
        - y1: potential outcomes under decision 1
        - p_dict: dictionary of decision-making quality
            p_dict[-1]: probability of making a decision resulting in effect=-1
            p_dict[0]: probability of making a decision resulting in effect=0
            p_dict[1]: probability of making a decision resulting in effect=1
        - seed: random seed
    """
    rng = np.random.default_rng(seed)
    indiv_effects = y1 - y0
    n = len(indiv_effects)

    mdl_d = np.zeros(n)
    for effect_val in [-1, 0, 1]:
        n_effect_val = np.sum(indiv_effects == effect_val)
        mdl_d_effect_val = rng.binomial(1, p_dict[effect_val], size=n_effect_val)
        mdl_d[indiv_effects == effect_val] = mdl_d_effect_val
    return mdl_d


def get_one_adaptive_decision(
    base_phys_decision: np.ndarray,
    mdl_decision: np.ndarray,
    z_vec: np.ndarray,
    adapt_type: str,
    seed: float,
    **kwargs,
) -> int:
    """
    Get one decision based on the treatment assignment vector and adaptation type.

    Args:
        - base_phys_decision: baseline physician decision (what they would do if no exposure to tool)
        - mdl_decision: decision recommended by tool/model
        - z_vec: treatment assignment vector
        - adapt_type: adaptation type (static, linear, exponential, step)
        - seed: random seed
    """
    rng = np.random.default_rng(seed)

    if z_vec[-1] == 0:
        w_phys = get_z0_decision_prob(z_vec, adapt_type, **kwargs)
    else:
        w_phys = get_z1_decision_prob(z_vec, adapt_type, **kwargs)

    if isinstance(w_phys, np.ndarray):
        w_phys = w_phys[-1]

    decisions = [mdl_decision, base_phys_decision]
    pick_d_phys = rng.binomial(1, w_phys)

    return decisions[pick_d_phys]


def get_mult_adaptive_decisions(
    base_phys_decisions: np.ndarray,
    mdl_decisions: np.ndarray,
    z_vec: np.ndarray,
    adapt_type: str,
    seed: float,
    **kwargs,
) -> np.ndarray:
    """
    Get sequence of decisions based on the treatment assignment vector and adaptation type.

    Args:
        - base_phys_decision: baseline physician decisions (what they would do if no exposure to tool)
        - mdl_decisions: decisions recommended by tool/model
        - z_vec: treatment assignment vector
        - adapt_type: adaptation type (static, linear, exponential, step)
        - seed: random seed
    """
    rng = np.random.default_rng(seed)
    n = len(z_vec)

    # Get probabilities of choosing physician's baseline decisions for z=0 and z=1
    w_phys_z0 = get_z0_decision_prob(z_vec, adapt_type, **kwargs)
    w_phys_z1 = get_z1_decision_prob(z_vec, adapt_type, **kwargs)

    d = np.zeros(n, dtype=int)

    # Sample observed decisions
    pick_d_phys_z0 = rng.binomial(1, w_phys_z0, size=n)
    pick_d_phys_z1 = rng.binomial(1, w_phys_z1, size=n)

    mask_z0_pick_d_phys = (z_vec == 0) & (pick_d_phys_z0 == 1)
    mask_z0_pick_d_mdl = (z_vec == 0) & (pick_d_phys_z0 == 0)
    mask_z1_pick_d_phys = (z_vec == 1) & (pick_d_phys_z1 == 1)
    mask_z1_pick_d_mdl = (z_vec == 1) & (pick_d_phys_z1 == 0)

    d[mask_z0_pick_d_phys] = base_phys_decisions[mask_z0_pick_d_phys]
    d[mask_z0_pick_d_mdl] = mdl_decisions[mask_z0_pick_d_mdl]
    d[mask_z1_pick_d_phys] = base_phys_decisions[mask_z1_pick_d_phys]
    d[mask_z1_pick_d_mdl] = mdl_decisions[mask_z1_pick_d_mdl]

    return d
