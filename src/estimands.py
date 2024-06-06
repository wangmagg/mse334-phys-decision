"""
Functions for calculating ground truth effect estimands and estimating effects
"""

import numpy as np
from src.decisions import get_one_adaptive_decision, get_mult_adaptive_decisions


def get_tale(phys_d: np.ndarray, mdl_d: np.ndarray, seed: float, **kwargs) -> float:
    """
    Get the ground truth total average learning effect.

    Args:
        - phys_d: array of baseline physician decisions
        - mdl_d: array of model decisions
        - seed: random seed
        - kwargs: additional arguments
            order_type: p for order-preserved, m for order-marginalized
    """
    n = len(phys_d)
    d_all0 = np.zeros(n, dtype=int)
    d_all1_0 = np.zeros(n, dtype=int)

    all1_0 = np.ones(n, dtype=int)
    all1_0[-1] = 0
    all_0 = np.zeros(n, dtype=int)
    for n_sub in range(2, n):
        if kwargs["order_type"] == "p":
            all1_0 = np.ones(n_sub, dtype=int)
            all1_0[-1] = 0
            all_0 = np.zeros(n_sub, dtype=int)
        d_all1_0[n_sub] = get_one_adaptive_decision(
            phys_d[n_sub], mdl_d[n_sub], all1_0, seed=seed, **kwargs
        )
        d_all0[n_sub] = get_one_adaptive_decision(
            phys_d[n_sub], mdl_d[n_sub], all_0, seed=seed, **kwargs
        )

    q_all1_0 = (d_all1_0 == mdl_d).astype(int)
    q_all0 = (d_all0 == mdl_d).astype(int)

    p_tale = np.mean(q_all1_0 - q_all0)

    return p_tale


def get_tahe(phys_d: np.ndarray, mdl_d: np.ndarray, seed: float, **kwargs) -> float:
    """
    Get the ground truth total average habituation effect.

    Args:
        - phys_d: array of baseline physician decisions
        - mdl_d: array of model decisions
        - seed: random seed
        - kwargs: additional arguments
            order_type: p for order-preserved, m for order-marginalized
    """
    n = len(phys_d)

    d_all1 = np.zeros(n, dtype=int)
    d_all0_1 = np.zeros(n, dtype=int)

    all0_1 = np.zeros(n, dtype=int)
    all0_1[-1] = 1
    all_1 = np.ones(n, dtype=int)

    for n_sub in range(2, n):
        if kwargs["order_type"] == "p":
            all0_1 = np.zeros(n_sub, dtype=int)
            all0_1[-1] = 1
            all_1 = np.ones(n_sub, dtype=int)

        d_all0_1[n_sub] = get_one_adaptive_decision(
            phys_d[n_sub], mdl_d[n_sub], all0_1, seed=seed, **kwargs
        )
        d_all1[n_sub] = get_one_adaptive_decision(
            phys_d[n_sub], mdl_d[n_sub], all_1, seed=seed, **kwargs
        )

    q_all1 = (d_all1 == mdl_d).astype(int)
    q_all0_1 = (d_all0_1 == mdl_d).astype(int)

    p_tahe = np.mean(q_all1 - q_all0_1)

    return p_tahe


def get_tate(
    y0: np.ndarray,
    y1: np.ndarray,
    phys_d: np.ndarray,
    mdl_d: np.ndarray,
    seed: float,
    **kwargs
) -> float:
    """
    Get the ground truth total average treatment effect.

    Args:
        - y0: array of potential outcomes under decision 0
        - y1: array of potential outcomes under decision 1
        - phys_d: array of baseline physician decisions
        - mdl_d: array of model decisions
        - seed: random seed
    """
    n = len(y0)

    all1 = np.ones(n, dtype=int)
    all0 = np.zeros(n, dtype=int)

    d_all0 = get_mult_adaptive_decisions(phys_d, mdl_d, all0, seed=seed, **kwargs)
    y_all0 = y1 * d_all0 + y0 * (1 - d_all0)

    d_all1 = get_mult_adaptive_decisions(phys_d, mdl_d, all1, seed=seed, **kwargs)
    y_all1 = y1 * d_all1 + y0 * (1 - d_all1)

    p_tate = np.mean(y_all1 - y_all0)

    return p_tate
