import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path
from scipy import stats
from numpy.linalg import inv, LinAlgError
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve
from scipy.optimize import minimize
from joblib import Parallel, delayed
import re
import os
from threadpoolctl import threadpool_limits



output_path_pic = Path(r"C:\Users\fipli\OneDrive - UvA\TI Master content\2.3 Advanced Time Series Econometrics\Assignment 2\Pictures Assignment 2")
output_path_tables = Path(r"C:\Users\fipli\OneDrive - UvA\TI Master content\2.3 Advanced Time Series Econometrics\Assignment 2\latex_tables")
output_path_pic.mkdir(parents=True, exist_ok=True)
output_path_tables.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Q1
# ======================================================================
# # Q1

# ======================================================================
# Q1.1
# ======================================================================
# ## Q1.1

N = 50
T = 100
B = 1000
rho = 0.5
seed = 1827

rng = np.random.default_rng(seed)


#Build equicorrelated Sigma
#Sigma = (1-rho) I + rho 11'
I = np.eye(N)
one = np.ones((N, 1))
Sigma = (1 - rho) * I + rho * (one @ one.T)  # N x N

# Population eigenvalues for equicorrelation:
# (1-rho) with multiplicity N-1, and 1+(N-1)rho once
pop_eigs = np.array([1 - rho] * (N - 1) + [1 + (N - 1) * rho])
pop_eigs.sort()  # ascending


# Monte Carlo loop
eigs_store = np.empty((B, N))

for b in range(B):
    # Simulate Y: T draws of N-dim normal with covariance Sigma
    Y = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T)

    # Sample covariance S = (1/T) (Y-mean)'(Y-mean)
    mean = Y.mean(axis=0)
    Y = Y - mean
    S = (Y.T @ Y) / T

    # Eigenvalues 
    lam = np.linalg.eigvalsh(S)  # already real, sorted ascending, used A.I. to determine which function to use on python
    eigs_store[b, :] = lam


avg_eigs = eigs_store.mean(axis=0)
i = np.arange(1, N + 1)

#plot (I added the plot also in a non-log scale,a nd used A.I. to make the graph more readable)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fig.suptitle(f"Non-shrunk eigenvalues: N={N}, T={T}, rho={rho}, B={B}")

# left panel
axes[0].plot(i, avg_eigs, marker="o", linestyle="-", label="Average sample eigenvalues")
axes[0].plot(i, pop_eigs, marker="x", linestyle="--", label="Population eigenvalues")
axes[0].set_xlabel("Eigenvalue index i (sorted)")
axes[0].set_ylabel("Eigenvalue")

# right panel
axes[1].plot(i, avg_eigs, marker="o", linestyle="-", label="Average sample eigenvalues")
axes[1].plot(i, pop_eigs, marker="x", linestyle="--", label="Population eigenvalues")
axes[1].set_xlabel("Eigenvalue index i (sorted)")
axes[1].set_ylabel("Eigenvalue, log scale")
axes[1].set_yscale("log")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 0.93])

plt.show()
out_path = output_path_pic / "A2_Fig1.png" 
fig.savefig(out_path, dpi=300, bbox_inches="tight")

# ======================================================================
# Q1.2
# ======================================================================
# ## Q1.2

def cov1Para(Y,k = None): #the code to obtain this linear shrinkage function is 
                          #retrieved from the following github repository: https://github.com/pald22/covShrinkage/blob/main/cov1Para.py

    #default setting
    if k is None or math.isnan(k):
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)                               #demean
        k = 1

    #vars
    T,N= Y.shape
    n = T-k                                  
   
    
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    
    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar= sum(diag)/len(diag)
    target=meanvar*np.eye(N)
    
    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    
    pihat = sum(piMat.sum())
    

    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    

    # diagonal part of the parameter that we call rho 
    rho_diag=0;
    
    # off-diagonal part of the parameter that we call rho 
    rho_off=0;
    
    # compute shrinkage intensity
    rhohat=rho_diag+rho_off
    kappahat=(pihat-rhohat)/gammahat
    shrinkage=max(0,min(1,kappahat/n))
    
    # compute shrinkage estimator
    sigmahat=shrinkage*target+(1-shrinkage)*sample
    
    return sigmahat


eigs_linear_store = np.empty((B, N))

for b in range(B):
    # Simulate Y: T draws of N-dim normal with covariance Sigma
    Y = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T)

    # Substitute the sample covariance with the linearly shrunk covariance
    sigmahat = cov1Para(pd.DataFrame(Y), k=None)
    # Eigenvalues 
    lam = np.linalg.eigvalsh(sigmahat)  # already real, sorted ascending, used A.I. to determine which function to use on python
    eigs_linear_store[b, :] = lam   

avg_eigs_linear = eigs_linear_store.mean(axis=0)
i = np.arange(1, N + 1)


fig = plt.figure(figsize=(8, 6))
plt.plot(i, avg_eigs, marker="o", linestyle="--", label="Average non-shrunk eigenvalues", color= "skyblue")
plt.plot(i, pop_eigs, marker="x", linestyle="--", label="Population eigenvalues", color = "darkorange")
plt.plot(i, avg_eigs_linear, marker="+", linestyle="-", label="Average linearly shrunk eigenvalues", color = "green")

plt.xlabel("Eigenvalue index i (sorted)")
plt.ylabel("Eigenvalue, log scale")
plt.yscale("log")
plt.legend()
plt.title(f"Linearly shrunk vs non-shrunk eigenvalues: N={N}, T={T}, rho={rho}, B={B}")

plt.show()
out_path = output_path_pic / "A2_Fig2.png" 
fig.savefig(out_path, dpi=300, bbox_inches="tight")

# ======================================================================
# Q1.3
# ======================================================================
# ## Q1.3

oracle_eigs = np.zeros((B, N))
sigma_oracle = np.zeros((B, N, N))

for b in range(B):
    Y = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T)

    Y = Y - Y.mean(axis=0)
    S = (Y.T @ Y) / T

    # --- CHANGE: compute eigenvalues + eigenvectors ---
    lam, U = np.linalg.eigh(S)  # lam ascending, U columns are eigenvectors

    # oracle eigenvalues: replace first N-1 by their mean, keep largest
    oracle_eigs[b, :-1] = np.mean(lam[:-1])
    oracle_eigs[b, -1]  = lam[-1]

    sigma_oracle[b, :, :] = U @ np.diag(oracle_eigs[b, :]) @ U.T

avg_eigs_oracle = oracle_eigs.mean(axis=0)
est_sigma_oracle = sigma_oracle.mean(axis=0)
i = np.arange(1, N + 1)

print("Estimated covariance matrix from oracle eigenvalues:\n", est_sigma_oracle)

fig = plt.figure(figsize=(8, 6))
plt.plot(i, avg_eigs_oracle, marker="o", linestyle="--", label="Oracle eigenvalues", color="skyblue")
plt.plot(i, pop_eigs, marker="x", linestyle="--", label="Population eigenvalues", color="darkorange")
plt.plot(i, avg_eigs_linear, marker="+", linestyle="-", label="Average linearly shrunk eigenvalues", color="green")
plt.xlabel("Eigenvalue index i (sorted)")
plt.ylabel("Eigenvalue, log scale")
plt.yscale("log")
plt.legend()
plt.title(f"Linearly shrunk vs oracle eigenvalues: N={N}, T={T}, rho={rho}, B={B}")

out_path = output_path_pic / "A2_Fig3.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(avg_eigs_oracle)
print(pop_eigs)

# ======================================================================
# Q1.4
# ======================================================================
# ## Q1.4

def make_equicorr_sigma(N, rho):
    I = np.eye(N)
    one = np.ones((N, 1))
    return (1 - rho) * I + rho * (one @ one.T)

def simulate_bekk_vec(T, alpha, beta, Sigma_bar, rng):
    N = Sigma_bar.shape[0]
    y = np.zeros((T, N))

    Sigma_t = Sigma_bar.copy()   # Sigma_1
    y_prev = np.zeros(N)         # y_0 = 0 so Sigma_1 = Sigma_bar

    for t in range(T):
        # draw y_t | F_{t-1} ~ N(0, Sigma_t)
        y[t] = rng.multivariate_normal(np.zeros(N), Sigma_t)

        # update Sigma_{t+1}
        Sigma_t = (1 - alpha - beta) * Sigma_bar \
                  + alpha * np.outer(y[t], y[t]) \
                  + beta * Sigma_t

    return y


def sample_cov_1_over_T(y):
    y_d = y - y.mean(axis=0)
    return (y_d.T @ y_d) / y.shape[0]  # 1/T

def oracle_cov_equicorr_pooling(S):
    lam, U = np.linalg.eigh(S)  # ascending
    lam_or = lam.copy()
    lam_or[:-1] = lam[:-1].mean()
    return U @ np.diag(lam_or) @ U.T

def alpha_beta_from_ab(a, b):
    """
    Map unconstrained (a,b) to (alpha,beta) with alpha,beta>=0 and alpha+beta<1
    """
    ea, eb = np.exp(a), np.exp(b)
    den = 1.0 + ea + eb
    alpha = ea / den
    beta  = eb / den
    return alpha, beta

def neg_loglik_one(params_ab, y_b, Sigma_bar):
    """
    Gaussian QML objective (constants dropped):
      sum_t 0.5 * (log|Sigma_t| + y_t' Sigma_t^{-1} y_t)
    with recursion:
      Sigma_{t+1} = (1-a-b)Sigma_bar + a y_t y_t' + b Sigma_t
    """
    a_raw, b_raw = params_ab
    alpha, beta = alpha_beta_from_ab(a_raw, b_raw)

    T, N = y_b.shape
    Sigma_t = Sigma_bar.copy()
    nll = 0.0

    for t in range(T):
        y = y_b[t]
        try:
            L = cholesky(Sigma_t, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        z = solve_triangular(L, y, lower=True, check_finite=False)
        quad = float(np.dot(z, z))
        nll += 0.5 * (logdet + quad)

        Sigma_t = (1.0 - alpha - beta) * Sigma_bar + alpha * np.outer(y, y) + beta * Sigma_t

    return nll

def estimate_one_replication_three_strategies(b, y_simulated, Sigma_bar_all_b, x0_ab, method, options):
    """
    One replication b:
      estimate alpha/beta separately for 3 strategies (Sigma_bar fixed per strategy)
    """
    y_b = y_simulated[b]
    ab_hat = np.zeros((3, 2))
    alpha_hat = np.zeros(3)
    beta_hat  = np.zeros(3)
    nll_hat   = np.zeros(3)
    success   = np.zeros(3, dtype=bool)

    for s in range(3):
        Sigma_bar = Sigma_bar_all_b[s]
        obj = lambda p: neg_loglik_one(p, y_b, Sigma_bar)
        res = minimize(obj, x0=np.asarray(x0_ab, dtype=float), method=method, options=options)

        ab_hat[s] = res.x
        a_s, b_s = alpha_beta_from_ab(res.x[0], res.x[1])
        alpha_hat[s] = a_s
        beta_hat[s]  = b_s
        nll_hat[s]   = res.fun
        success[s]   = res.success

    return ab_hat, alpha_hat, beta_hat, nll_hat, success

def estimate_qml_parallel(y_simulated, Sigma_bar_all, alpha0, beta0, method="L-BFGS-B", options=None, n_jobs=-1):
    """
    y_simulated: (B,T,N)
    Sigma_bar_all: (B,3,N,N)
    """
    c0 = 1 - alpha0 - beta0
    x0_ab = (np.log(alpha0/c0), np.log(beta0/c0))
    
    if options is None:
        options = {"maxiter": 50}

    B = y_simulated.shape[0]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(estimate_one_replication_three_strategies)(
            b, y_simulated, Sigma_bar_all[b], x0_ab, method, options
        )
        for b in range(B)
    )

    ab_hat    = np.stack([r[0] for r in results], axis=0)  # (B,3,2)
    alpha_hat = np.stack([r[1] for r in results], axis=0)  # (B,3)
    beta_hat  = np.stack([r[2] for r in results], axis=0)  # (B,3)
    nll_hat   = np.stack([r[3] for r in results], axis=0)  # (B,3)
    success   = np.stack([r[4] for r in results], axis=0)  # (B,3)

    summary = {
        "success_rate": success.mean(axis=0),
        "alpha_mean": alpha_hat.mean(axis=0),
        "alpha_sd": alpha_hat.std(axis=0, ddof=1) if B > 1 else np.zeros(3),
        "beta_mean": beta_hat.mean(axis=0),
        "beta_sd": beta_hat.std(axis=0, ddof=1) if B > 1 else np.zeros(3),
        "nll_mean": nll_hat.mean(axis=0),
        "nll_sd": nll_hat.std(axis=0, ddof=1) if B > 1 else np.zeros(3),
    }
    return ab_hat, alpha_hat, beta_hat, nll_hat, success, summary

def min_var_weights(cov_matrix, jitter=1e-8, use_pin=False):
    """
    w = Sigma^{-1}1 / (1' Sigma^{-1} 1) via Cholesky solves for stability (no explicit inverse)
    If Sigma is not positive definite, either use pseudoinverse or add tiny jitter and retry
    """
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    try:
        L = cholesky(cov_matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        if use_pin:
            pinv = np.linalg.pinv(cov_matrix)
            x = pinv @ ones
            return x / (ones @ x)
        cov_matrix = cov_matrix + jitter * np.eye(n)
        L = cholesky(cov_matrix, lower=True, check_finite=False)

    y = solve_triangular(L, ones, lower=True, check_finite=False)
    x = solve_triangular(L.T, y, lower=False, check_finite=False)
    return x / (ones @ x)

def portfolio_stats_per_replication(y_simulated, Sigma_bar_all, alpha_hat, beta_hat):
    """
    For each replication b and method s:
      - filter Sigma_t
      - compute min-var portfolio returns r_t = w_t' y_t
      - compute mean_t(r) and var_t(r)
    Then return averages over b (means only; NO SEs)
    """
    B, T, N = y_simulated.shape
    port_mean_b = np.zeros((B, 3))
    port_var_b  = np.zeros((B, 3))

    for b in range(B):
        y_b = y_simulated[b]
        for s in range(3):
            Sigma_bar = Sigma_bar_all[b, s]
            a_hat = alpha_hat[b, s]
            b_hat = beta_hat[b, s]

            Sigma_t = Sigma_bar.copy()
            r = np.zeros(T)

            for t in range(T):
                w = min_var_weights(Sigma_t)
                r[t] = float(w @ y_b[t])
                Sigma_t = (1.0 - a_hat - b_hat) * Sigma_bar + a_hat * np.outer(y_b[t], y_b[t]) + b_hat * Sigma_t

            port_mean_b[b, s] = r.mean()
            port_var_b[b, s]  = r.var(ddof=1)

    return {
        "mean_mean": port_mean_b.mean(axis=0),
        "var_mean":  port_var_b.mean(axis=0),
    }

# =============================
# Main runner (ONE grid point)
# =============================
def run_q14_experiment(
    N, T, B,
    rho=0.5,
    alpha_true = 0.93, beta_true=0.05,
    seed=1827,
    qml_maxiter=200,
    n_jobs=-1,
):
    rng = np.random.default_rng(seed)
    Sigma_true = make_equicorr_sigma(N, rho)

    # 1) Simulate B replications
    y_sim = np.zeros((B, T, N))
    for b in range(B):
        y_sim[b] = simulate_bekk_vec(T, alpha_true, beta_true, Sigma_true, rng)

    # 2) Replication-specific unconditional Sigmas
    Sigma_bar_all = np.zeros((B, 3, N, N))
    for b in range(B):
        S_b = sample_cov_1_over_T(y_sim[b])
        Sigma_bar_all[b, 0] = S_b
        Sigma_bar_all[b, 1] = cov1Para(pd.DataFrame(y_sim[b]), k=None)  
        Sigma_bar_all[b, 2] = oracle_cov_equicorr_pooling(S_b)

    # 3) QML estimation (alpha/beta) per replication, per method
    ab_hat, alpha_hat, beta_hat, nll_hat, success, qml_summary = estimate_qml_parallel(
        y_simulated=y_sim,
        Sigma_bar_all=Sigma_bar_all,
        alpha0=alpha_true, beta0=beta_true,
        method="L-BFGS-B",
        options={"maxiter": qml_maxiter},
        n_jobs=n_jobs
    )

    # 4) Portfolio performance (means only)
    port_summary = portfolio_stats_per_replication(y_sim, Sigma_bar_all, alpha_hat, beta_hat)

    return {
        "N": N, "T": T, "B": B, "rho": rho, "alpha_true": alpha_true, "beta_true": beta_true, "seed": seed,
        "qml": qml_summary,
        "portfolio": port_summary,
        "success": success,
    }

# =============================
# Table writing (ONE table per grid point) — UPDATED EXPORT STYLE
# =============================
def cell_with_se(mean, se, fmt_mean="{:.4f}", fmt_se="{:.4f}"):
    return rf"\shortstack[l]{{{fmt_mean.format(mean)} \\ ({fmt_se.format(se)})}}"

def cell_plain(x, fmt="{:.6f}"):
    return fmt.format(x)

def save_one_grid_table(
    res,
    outdir: Path,
    tag_prefix="q14",
    tabcolsep_pt=3,          
    arraystretch=1.25,       
    add_midrules=True,
    center_caption_and_table=True
):
    """
    Saves ONE LaTeX table for that (N,T,B)
    Columns:
      Alpha (mean + SE), Beta (mean + SE), Mean return (mean only), Var return (mean only)

    Extras:
      - centered table + caption
      - tighter spacing with tabcolsep
      - midrules between methods
    """
    outdir.mkdir(parents=True, exist_ok=True)

    N, T, B = res["N"], res["T"], res["B"]
    q = res["qml"]
    p = res["portfolio"]

    strategies = ["No Shrinkage", "Linear Shrinkage", "Oracle"]

    rows = []
    for s_idx, s_name in enumerate(strategies):
        alpha_mean = q["alpha_mean"][s_idx]
        alpha_sd   = q["alpha_sd"][s_idx]  if B > 1 else 0.0

        beta_mean  = q["beta_mean"][s_idx]
        beta_sd    = q["beta_sd"][s_idx]  if B > 1 else 0.0

        mean_ret = p["mean_mean"][s_idx]
        var_ret  = p["var_mean"][s_idx]

        rows.append({
            "Method": s_name,
            "Alpha": cell_with_se(alpha_mean, alpha_sd, "{:.4f}", "{:.4f}"),
            "Beta":  cell_with_se(beta_mean,  beta_sd,  "{:.4f}", "{:.4f}"),
            "Mean return": cell_plain(mean_ret, "{:.6f}"),
            "Var return":  cell_plain(var_ret,  "{:.6f}"),
        })

    df = pd.DataFrame(rows)

    caption = f"Q1.4 results (N={N}, T={T}, B={B})"
    label = f"tab:{tag_prefix}_N{N}_T{T}_B{B}"

    tex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="lcccc"
    )

    # ---- Post-process: centering + spacing ----
    if center_caption_and_table:
        # center the tabular
        if "\\begin{table}\n\\centering" not in tex:
            tex = tex.replace("\\begin{table}", "\\begin{table}\n\\centering", 1)
        # force caption centering even under classes that left-align captions
        tex = re.sub(r"\\caption\{", r"\\caption{\\centering ", tex, count=1)

    # tighter columns + nicer row spacing
    tex = tex.replace(
        "\\begin{table}",
        "\\begin{table}\n"
        f"\\setlength\\tabcolsep{{{tabcolsep_pt}pt}}\n"
        f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}",
        1
    )

    # ---- Post-process: add midrules between method rows ----
    if add_midrules:
        m = re.search(r"(\\midrule\s*\n)(.*?)(\\bottomrule)", tex, flags=re.S)
        if m:
            header_mid = m.group(1)
            body = m.group(2)
            bottom = m.group(3)

            lines = body.splitlines()
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if line.strip().endswith(r"\\"):
                    new_lines.append(r"\midrule")
            body2 = "\n".join(new_lines) + "\n"

            tex = tex[:m.start()] + header_mid + body2 + bottom + tex[m.end():]
            # remove an extra midrule right before bottomrule, if inserted
            tex = tex.replace("\\midrule\n\\bottomrule", "\\bottomrule")

    outpath = outdir / f"{tag_prefix}_N{N}_T{T}_B{B}.tex"
    outpath.write_text(tex, encoding="utf-8")
    return outpath

# RUN THE GRID
grid = [
    {"N": 20, "T": 100, "B": 500},
    {"N": 50, "T": 100, "B": 500},
    {"N": 20, "T": 200, "B": 500},
    {"N": 50, "T": 200, "B": 500},
    {"N": 20, "T": 1000, "B": 500},
    {"N": 50, "T": 1000, "B": 500}
]

saved_paths = []

for g in grid:
    res = run_q14_experiment(
        N=g["N"], T=g["T"], B=g["B"],
        rho=0.5,
        alpha_true=0.05, beta_true=0.93,
        seed=1827,
        qml_maxiter=200,
        n_jobs=-1
    )

    print(f"Done N={g['N']} T={g['T']} B={g['B']}")
    print("  alpha means:", res["qml"]["alpha_mean"])
    print("  beta  means:", res["qml"]["beta_mean"])
    print("  success rates:", res["qml"]["success_rate"])

    outpath = save_one_grid_table(
        res,
        outdir=output_path_tables,
        tag_prefix="q14",
        tabcolsep_pt=3,       
        arraystretch=1.25,    
        add_midrules=True,
        center_caption_and_table=True
    )
    saved_paths.append(outpath)
    print("  saved table:", outpath)

print("All saved tables:")
for p in saved_paths:
    print(" ", p)

# ======================================================================
# Q2
# ======================================================================
# # Q2

#GSPC
gspc_path = "GSPC.csv"  
gspc = pd.read_csv(gspc_path, skiprows=3, header=None,
                   names=["Date","Close","High","Low","Open","Volume"])
gspc["Date"] = pd.to_datetime(gspc["Date"])
gspc = gspc.sort_values("Date")
gspc["r"] =  ((np.log(gspc["Close"]) - np.log(gspc["Close"].shift(1))) * 100)
gspc = gspc.dropna(subset=["r"])

# RV/vol
rv_path = "dachxiu.chicagobooth.csv"  
rv = pd.read_csv(rv_path)

rv["Date"] = pd.to_datetime(rv["Date"])
rv = rv.sort_values("Date")

rv = rv[(rv["Symbol"] == "SPY") & (rv["Type"] == "QMLE-Trade")].copy()

rv["RV"] = ((rv["Volatility"])**2)*100

rv["logRV"] = np.log(rv["RV"])
rv = rv.dropna(subset=["RV","logRV"])

#merge
start = pd.Timestamp("2014-02-11")
end   = pd.Timestamp("2026-02-11")

gspc_sub = gspc[(gspc["Date"] >= start) & (gspc["Date"] <= end)].copy()
rv_sub   = rv[(rv["Date"] >= start) & (rv["Date"] <= end)].copy()

df = pd.merge(gspc_sub[["Date","r"]],
              rv_sub[["Date","RV","logRV"]],
              on="Date", how="inner").sort_values("Date")

# ======================================================================
# Q2.1
# ======================================================================
# ## Q2.1 - Q2.2

machine_zero = 1e-12  # variance floor used only for numerical safety

def aug_garch_filter_safe(r, RV, mu, omega, alpha, beta, gamma): #augmented model
    T = len(r)
    eps = r - mu
    h = np.zeros(T)

    h0 = np.var(eps) #use the sample variance of eps as the initial variance (unconditional variance) and enforce positivity
    if not np.isfinite(h0) or h0 <= 0:
        return eps, None
    h[0] = h0

    for t in range(1, T):
        ht = omega + alpha * eps[t-1]**2 + beta * h[t-1] + gamma * RV[t-1]
        if (not np.isfinite(ht)) or (ht <= 0): #enforce positivity of h_t
            return eps, None
        h[t] = ht

    return eps, h


def garch_filter_safe(r, mu, omega, alpha, beta):
    T = len(r)
    eps = r - mu
    h = np.zeros(T)

    h0 = np.var(eps) #use the sample variance of eps as the initial variance (unconditional variance) and enforce positivity
    if not np.isfinite(h0) or h0 <= 0:
        return eps, None
    h[0] = h0

    for t in range(1, T):
        ht = omega + alpha * eps[t-1]**2 + beta * h[t-1]
        if (not np.isfinite(ht)) or (ht <= 0): #enforce positivity of h_t
            return eps, None
        h[t] = ht

    return eps, h


def negloglik_aug_garch(theta, r, RV):
    mu, omega, alpha, beta, gamma = theta
    eps, h = aug_garch_filter_safe(r, RV, mu, omega, alpha, beta, gamma)

    if h is None:
        return np.inf

    h_safe = np.maximum(h, machine_zero)
    return 0.5 * np.sum(np.log(h_safe) + (eps**2) / h_safe)


def negloglik_garch(theta, r):
    mu, omega, alpha, beta = theta
    eps, h = garch_filter_safe(r, mu, omega, alpha, beta)

    if h is None:
        return np.inf

    h_safe = np.maximum(h, machine_zero)
    return 0.5 * np.sum(np.log(h_safe) + (eps**2) / h_safe)


def fit_garch(r):
    # starting guess
    mu0 = r.mean()
    omega0 = 0.05 * np.var(r)
    alpha0 = 0.05
    beta0  = 0.90
    theta0 = np.array([mu0, omega0, alpha0, beta0])

    bounds = [
        (None, None),   # mu (unbounded)
        (1e-9, None),   # omega (>0, as it is a variance level)
        (0, 1.0),     # alpha (between 0 and 1)
        (0, 1.0)      # beta (between 0 and 1)
    ]

    # constraint: alpha + beta < 1
    constraints = [
        {"type": "ineq", "fun": lambda th: 0.999 - (th[2] + th[3])}
    ]

    #solve the (constrained) optimization problem
    res = minimize(
        negloglik_garch,
        theta0,
        args=(r,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return res


def fit_aug_garch(r, RV):
    # ---- starting guesses
    mu0 = r.mean()
    omega0 = 0.05 * np.var(r)
    alpha0 = 0.05
    beta0  = 0.10
    gamma0 = 0.80
    theta0 = np.array([mu0, omega0, alpha0, beta0, gamma0])

    #impose gamma >= 0 because RV is a variance measure and
    bounds = [
        (None, None),   # mu
        (1e-9, None),   # omega
        (0, 1.0),     # alpha
        (0, 1.0),     # beta
        (None, None)     # I expect it to be positive and bounded in a similar way to beta
    ]

    # constraint: alpha + beta < 1, guarantee stability
    constraints = [
        {"type": "ineq", "fun": lambda th: 0.999 - (th[2] + th[3])}
    ]

    res = minimize(
    negloglik_aug_garch,
    theta0,
    args=(r, RV),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": True, "maxiter": 5000, "ftol": 1e-12}
    )
    return res



def numerical_hessian(fun, x0, args=(), rel_eps=1e-5):
    """
    Central-difference Hessian with parameter-scaled steps.
    Raises ValueError if non-finite evaluations occur.
    """
    x0 = np.asarray(x0, float)
    n = x0.size
    H = np.zeros((n, n), float)

    f0 = fun(x0, *args)
    if not np.isfinite(f0):
        raise ValueError("fun(x0) is not finite at x0.")

    h = rel_eps * (1.0 + np.abs(x0))  # scale step per parameter

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        for j in range(i, n):
            ej = np.zeros(n); ej[j] = 1.0

            if i == j:
                f_plus  = fun(x0 + h[i]*ei, *args)
                f_minus = fun(x0 - h[i]*ei, *args)
                if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
                    raise ValueError(f"Non-finite Hessian eval at i={i}. Try smaller rel_eps.")
                H[i, i] = (f_plus - 2.0*f0 + f_minus) / (h[i]**2)
            else:
                f_pp = fun(x0 + h[i]*ei + h[j]*ej, *args)
                f_pm = fun(x0 + h[i]*ei - h[j]*ej, *args)
                f_mp = fun(x0 - h[i]*ei + h[j]*ej, *args)
                f_mm = fun(x0 - h[i]*ei - h[j]*ej, *args)
                if not all(np.isfinite([f_pp, f_pm, f_mp, f_mm])):
                    raise ValueError(f"Non-finite Hessian eval at (i,j)=({i},{j}). Try smaller rel_eps.")
                Hij = (f_pp - f_pm - f_mp + f_mm) / (4.0*h[i]*h[j])
                H[i, j] = Hij
                H[j, i] = Hij
    return H


def se_from_hessian(fun, theta_hat, args=(), rel_eps=1e-5, ridge=1e-10):
    """
    Naive inverse-Hessian covariance from sum objective.
    Returns (se, cov, H).
    """
    H = numerical_hessian(fun, theta_hat, args=args, rel_eps=rel_eps)

    # stabilize inversion
    Hs = H + ridge*np.eye(len(theta_hat))

    try:
        cov = np.linalg.inv(Hs)
    except LinAlgError:
        return None, None, H

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return se, cov, H


def run_models_and_test(r, RV):

    res0 = fit_garch(r)          # restricted
    res1 = fit_aug_garch(r, RV)  # unrestricted

    if (not res0.success) or (not res1.success):
        print("Optimization did not fully converge.")
        print("GARCH message :", res0.message)
        print("AUG_GARCH message:", res1.message)

    ll0 = -res0.fun
    ll1 = -res1.fun

    LR = 2 * (ll1 - ll0)
    pval = stats.chi2.sf(LR, df=1)

   
    se0, cov0, H0 = se_from_hessian(negloglik_garch, res0.x, args=(r,))
    se1, cov1, H1 = se_from_hessian(negloglik_aug_garch, res1.x, args=(r, RV))

    print("STANDARD GARCH(1,1) (restricted, gamma=0)")
    print("params [mu, omega, alpha, beta] =", res0.x)
    if se0 is not None:
        print("SEs    [mu, omega, alpha, beta] =", se0)
        print("alpha (se) =", res0.x[2], "(", se0[2], ")")
        print("beta  (se) =", res0.x[3], "(", se0[3], ")")
    print("log-likelihood =", ll0)
    print("alpha + beta   =", res0.x[2] + res0.x[3])

    print("\nAUG_GARCH(1,1) with RV_{t-1} (unrestricted)")
    print("params [mu, omega, alpha, beta, gamma] =", res1.x)
    if se1 is not None:
        print("SEs    [mu, omega, alpha, beta, gamma] =", se1)
        print("alpha (se) =", res1.x[2], "(", se1[2], ")")
        print("beta  (se) =", res1.x[3], "(", se1[3], ")")
        print("gamma (se) =", res1.x[4], "(", se1[4], ")")
    print("log-likelihood =", ll1)
    print("alpha + beta   =", res1.x[2] + res1.x[3])
    print("beta + gamma   =", res1.x[3] + res1.x[4])

    print("\nLikelihood Ratio test")
    print("LR statistic =", LR)
    print("p-value      =", pval)

    res0.se = se0
    res1.se = se1

    return res0, res1, LR, pval



def results_to_latex(res0, res1, T, caption, label):
    ll0 = -res0.fun
    ll1 = -res1.fun

    def fmt_est_se(est, se):
        if (est is None) or (pd.isna(est)):
            return ""
        if se is None or (pd.isna(se)):
            return f"{est:.6g}"
        return f"{est:.6g} ({se:.3g})"

    se0 = getattr(res0, "se", None)
    se1 = getattr(res1, "se", None)

    rows = [
        ("$\\mu$",    fmt_est_se(res0.x[0], se0[0] if se0 is not None else None),
                     fmt_est_se(res1.x[0], se1[0] if se1 is not None else None)),
        ("$\\omega$", fmt_est_se(res0.x[1], se0[1] if se0 is not None else None),
                     fmt_est_se(res1.x[1], se1[1] if se1 is not None else None)),
        ("$\\alpha$", fmt_est_se(res0.x[2], se0[2] if se0 is not None else None),
                     fmt_est_se(res1.x[2], se1[2] if se1 is not None else None)),
        ("$\\beta$",  fmt_est_se(res0.x[3], se0[3] if se0 is not None else None),
                     fmt_est_se(res1.x[3], se1[3] if se1 is not None else None)),
        ("$\\gamma$", "",
                     fmt_est_se(res1.x[4], se1[4] if se1 is not None else None)),
        ("$\\alpha+\\beta$", f"{(res0.x[2]+res0.x[3]):.6g}", f"{(res1.x[2]+res1.x[3]):.6g}"),
        ("Log-likelihood", f"{ll0:.6g}", f"{ll1:.6g}"),
        ("Converged", int(res0.success), int(res1.success)),
        ("Iterations", res0.nit, res1.nit),
    ]

    df_out = pd.DataFrame(rows, columns=["", "GARCH(1,1)", "AUG GARCH(1,1)"])

    latex = df_out.to_latex(
        index=False,
        escape=False,
        column_format="lcc",
        caption=caption,
        label=label
    )
    return latex






def save_latex_table(latex_code, output_path_tables, filename="qml_garch_results.tex"):
    output_path_tables = Path(output_path_tables)
    output_path_tables.mkdir(parents=True, exist_ok=True)
    out_file = output_path_tables / filename
    out_file.write_text(latex_code, encoding="utf-8")
    print("Saved LaTeX table to:", out_file)




r = df["r"].to_numpy()
RV = df["RV"].to_numpy()
res0, res1, LR, pval = run_models_and_test(r, RV)



latex_code = results_to_latex(
    res0, res1, T=len(r),
    caption="Gaussian QML estimates for GARCH(1,1) and AUG GARCH(1,1)",
    label="tab:qml_garch"
)

save_latex_table(latex_code, output_path_tables, filename="qml_garch_results.tex")
