"""
Main script for running simulated experiments.
"""

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

from src.decisions import get_mult_adaptive_decisions, get_decisions
from src.estimands import get_tate, get_tale, get_tahe


def get_potential_outcomes(n: int, p_neg: float, p_pos:float, seed:Optional[float]=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample potential outcomes

    Args:
        - n: number of patients
        - p_neg: probability of negative effect
        - p_pos: probability of positive effect
        - seed: random seed
    """
    rng = np.random.default_rng(seed)
    y0 = rng.binomial(1, 0.5, size=n)

    n_y0eq0 = np.sum(y0 == 0)
    n_y0eq1 = n - n_y0eq0

    y0eq0_effects = rng.choice([1, 0], p=[p_pos, 1 - p_pos], size=n_y0eq0)
    y0eq1_effects = rng.choice([-1, 0], p=[p_neg, 1 - p_neg], size=n_y0eq1)

    y1 = y0.copy()
    y1[y0 == 0] += y0eq0_effects
    y1[y0 == 1] += y0eq1_effects

    return y0, y1


def run_exp_one_sample(
    it: int,
    y0: np.ndarray,
    y1: np.ndarray,
    phys_d: np.ndarray,
    mdl_d: np.ndarray,
    n_z_iter: int,
    seed: float,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one experiment for a single sample of patients.

    Args:
        - it: sample iteration number
        - y0: array of potential outcomes under decision 0
        - y1: array of potential outcomes under decision 1
        - phys_d: array of baseline physician decisions
        - mdl_d: array of model decisions
        - n_z_iter: number of iterations for treatment assignment
        - seed: random seed

    Returns:
        - tau_hat_dm: dataframe of difference-in-means estimates across repeated treatment assignments
        - p_ta: dataframe of order-preserved treatment effects
        - m_ta: dataframe of order-marginalized treatment effects
    """

    # Get order-preserved treatment effects
    p_tate = get_tate(y0, y1, phys_d, mdl_d, seed, sum_fn=np.cumsum, **kwargs)
    p_tale = get_tale(phys_d, mdl_d, seed, sum_fn=np.cumsum, order_type="p", **kwargs)
    p_tahe = get_tahe(phys_d, mdl_d, seed, sum_fn=np.cumsum, order_type="p", **kwargs)
    p_ta = {"it": it, "tate": p_tate, "tale": p_tale, "tahe": p_tahe}

    # Get order-marginalized treatment effects
    m_tate = get_tate(y0, y1, phys_d, mdl_d, seed, sum_fn=np.sum, **kwargs)
    m_tale = get_tale(phys_d, mdl_d, seed, sum_fn=np.sum, order_type="m", **kwargs)
    m_tahe = get_tahe(phys_d, mdl_d, seed, sum_fn=np.sum, order_type="m", **kwargs)
    m_ta = {"it": it, "tate": m_tate, "tale": m_tale, "tahe": m_tahe}

    rng = np.random.default_rng(seed)
    tau_hat_dm = np.zeros(n_z_iter)
    
    # Get observed decisions and difference-in-means estimate of treatment effect
    # under repeated draws of treatment assignment vector
    n = len(y0)
    for i in range(n_z_iter):
        z = np.zeros(n, dtype=int)
        z[: n // 2] = 1
        rng.shuffle(z)

        obs_decisions = get_mult_adaptive_decisions(
            phys_d, mdl_d, z, seed=seed, sum_fn=np.cumsum, **kwargs
        )
        y = y1 * obs_decisions + y0 * (1 - obs_decisions)

        tau_hat_dm[i] = np.mean(y[z == 1]) - np.mean(y[z == 0])

    tau_hat_dm = pd.DataFrame({"it": it, "tau_hat_dm": tau_hat_dm})

    return tau_hat_dm, p_ta, m_ta


def run_exp_mult_sample(
    n: int,
    p_neg: float,
    p_pos: float,
    p_phys_dict: dict,
    p_mdl_dict: dict,
    n_z_iter: int,
    n_sample_iter: int,
    seed:Optional[int]=42,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run experiments for repeated patient sample draws.

    Args:
        - n: number of patients
        - p_neg: probability of negative effect
        - p_pos: probability of positive effect
        - p_phys_dict: dictionary of physician decision-making quality
        - p_mdl_dict: dictionary of model decision-making quality
        - n_z_iter: number of iterations for treatment assignment
        - n_sample_iter: number of sample iterations
        - seed: random seed

    Returns:
        - tau_hat_dm_all: dataframe of difference-in-means estimates across repeated samples
        - p_ta_all: dataframe of order-preserved treatment effects across repeated samples
        - m_ta_all: dataframe of order-marginalized treatment effects across repeated samples
 
    """
    tau_hat_dm_all = []
    p_ta_all = []
    m_ta_all = []

    # Run experiments for repeated samples of patients
    for it in range(n_sample_iter):
        # Sample potential outcomes, get baseline physician decisions and model decisions
        y0, y1 = get_potential_outcomes(n, p_neg, p_pos, seed + it)
        phys_d = get_decisions(y0, y1, p_phys_dict, seed + it)
        mdl_d = get_decisions(y0, y1, p_mdl_dict, seed + it)

        # Run experiment for a single sample of patients
        tau_hat_dm, p_ta, m_ta = run_exp_one_sample(
            it, y0, y1, phys_d, mdl_d, n_z_iter, seed + it, **kwargs
        )
        tau_hat_dm_all.append(tau_hat_dm)
        p_ta_all.append(p_ta)
        m_ta_all.append(m_ta)

    tau_hat_dm_all = pd.concat(tau_hat_dm_all)
    p_ta_all = pd.DataFrame.from_records(p_ta_all)
    m_ta_all = pd.DataFrame.from_records(m_ta_all)

    tau_hat_dm_all["n"] = n
    p_ta_all["n"] = n
    m_ta_all["n"] = n

    return tau_hat_dm_all, p_ta_all, m_ta_all


def config():
    """
    Parse command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--p-pos", type=float, default=0.2)
    parser.add_argument("--p-neg", type=float, default=0.7)

    parser.add_argument("--p-phys-pos", type=float, default=0.1)
    parser.add_argument("--p-phys-null", type=float, default=0.2)
    parser.add_argument("--p-phys-neg", type=float, default=0.9)

    parser.add_argument("--p-mdl-pos", type=float, default=0.4)
    parser.add_argument("--p-mdl-null", type=float, default=0.2)
    parser.add_argument("--p-mdl-neg", type=float, default=0.6)

    parser.add_argument(
        "--adapt-type",
        type=str,
        default="linear",
        choices=["linear", "exponential", "step", "static"],
    )
    parser.add_argument("--alpha0", type=float, default=0.01)
    parser.add_argument("--alpha0-pre", type=float, default=0.01)
    parser.add_argument("--alpha0-post", type=float, default=0.01)
    parser.add_argument("--alpha1", type=float, default=0.1)
    parser.add_argument("--alpha1-pre", type=float, default=0.1)
    parser.add_argument("--alpha1-post", type=float, default=0.1)

    parser.add_argument("--l-t", type=int, default=None)

    parser.add_argument("--n-z-iter", type=int, default=10)
    parser.add_argument("--n-sample-iter", type=int, default=100)
    parser.add_argument("--n-range", type=list, default=[100, 1000])
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

def get_save_subdir(args):
    """
    Get subdirectory for saving results.
    """
    save_subdir = (
        f"phys-{args.p_phys_pos}-{args.p_phys_null}-{args.p_phys_neg}"
        f"_mdl-{args.p_mdl_pos}-{args.p_mdl_null}-{args.p_mdl_neg}"
        )
    return save_subdir

def get_save_suffix(args):
    """
    Get filename suffix for saving results.
    """
    if args.adapt_type == "step":
        save_suffix = (
            f"ns-{args.n_sample_iter}"
            f"_nz-{args.n_z_iter}"
            f"_a0pr-{args.alpha0_pre}"
            f"_a0po-{args.alpha0_post}"
            f"_a1pr-{args.alpha1_pre}"
            f"_a1po-{args.alpha1_post}"
            f"_lt-{args.l_t}"
        )
    else:
        save_suffix = (
            f"ns-{args.n_sample_iter}"
            f"_nz-{args.n_z_iter}"
            f"_a0-{args.alpha0}"
            f"_a1-{args.alpha1}"
        )
    return save_suffix


if __name__ == "__main__":
    args = config()

    # Create directory for saving results
    save_subdir = get_save_subdir(args)
    save_dir = Path("res") / save_subdir / args.adapt_type
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Paths for saving results
    save_suffix = get_save_suffix(args)
    tau_hat_dm_save_path = save_dir / f"tau_hat_dm_{save_suffix}.pkl"
    p_ta_save_path = save_dir / f"p_ta_{save_suffix}.pkl"
    m_ta_save_path = save_dir / f"m_ta_{save_suffix}.pkl"
    p_tale_hat_dm_save_path = save_dir / f"p_tale_hat_dm_{save_suffix}.pkl"
    p_tahe_hat_dm_save_path = save_dir / f"p_tahe_hat_dm_{save_suffix}.pkl"

    # Run experiments for multiple sample sizes
    n_arr = np.linspace(args.n_range[0], args.n_range[1], args.n_steps, dtype=int)
    p_phys_dict = {-1: args.p_phys_neg, 0: args.p_phys_null, 1: args.p_phys_pos}
    p_mdl_dict = {-1: args.p_mdl_neg, 0: args.p_mdl_null, 1: args.p_mdl_pos}

    tau_hat_dm_all = []
    p_ta_all = []
    m_ta_all = []

    for n_idx, n in tqdm(enumerate(n_arr), total=len(n_arr)):
        (tau_hat_dm, p_ta, m_ta) = run_exp_mult_sample(
            n=n,
            p_neg=args.p_neg,
            p_pos=args.p_pos,
            p_phys_dict=p_phys_dict,
            p_mdl_dict=p_mdl_dict,
            n_z_iter=args.n_z_iter,
            n_sample_iter=args.n_sample_iter,
            alpha0=args.alpha0,
            alpha1=args.alpha1,
            alpha0_pre=args.alpha0_pre,
            alpha0_post=args.alpha0_post,
            alpha1_pre=args.alpha1_pre,
            alpha1_post=args.alpha1_post,
            adapt_type=args.adapt_type,
            l_t=args.l_t,
            seed=args.seed,
        )

        tau_hat_dm_all.append(tau_hat_dm)
        p_ta_all.append(p_ta)
        m_ta_all.append(m_ta)

    tau_hat_dm_all = pd.concat(tau_hat_dm_all)
    p_ta_all = pd.concat(p_ta_all)
    m_ta_all = pd.concat(m_ta_all)

    with open(tau_hat_dm_save_path, "wb") as f:
        tau_hat_dm_all.to_pickle(f)
    with open(p_ta_save_path, "wb") as f:
        p_ta_all.to_pickle(f)
    with open(m_ta_save_path, "wb") as f:
        m_ta_all.to_pickle(f)
