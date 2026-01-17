import os
import zipfile
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from itertools import combinations
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================================================== CONFIG =============================================================================
MIN_REQUIRED_INSTRUMENTS = 2  # Minimum number of instruments required in a window to keep it

ZETA = 0.95  # Publication value = 0.95
TAU = 0.75   # Publication value = 0.75

# ================================================================ LOADING AND PREPROCESSING ====================================================

def calculate_default_window_params(num_dates, target_windows=50):
    """
    Calculate default window_size and step to achieve approximately target_windows windows.
    
    Parameters
    ----------
    num_dates : int
        Number of data points available
    target_windows : int
        Target number of windows (default 50)
    
    Returns
    -------
    tuple
        (window_size, step) calculated to achieve target windows with ~50% overlap
    """
    # Formula: (num_dates - window_size) / step + 1 = target_windows
    # With 50% overlap: step = window_size / 2
    # Solving: window_size = num_dates * 2 / (target_windows + 1)
    window_size = max(1, int(num_dates * 2 / (target_windows + 1)))
    step = max(1, window_size // 2)  # 50% overlap for smoother evolution
    
    # Verify how many windows we actually get
    actual_windows = (num_dates - window_size) // step + 1
    return window_size, step

def load_csv_to_df(csv_path, start_date=None, end_date=None):
    """
    Loads a CSV file into a DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing data with columns:
        Data, Otwarcie, Najwyzszy, Najnizszy, Zamkniecie, Wolumen
    start_date : str, optional
        Start date for filtering data (format: 'YYYY-MM-DD'). If None, no lower bound.
    end_date : str, optional
        End date for filtering data (format: 'YYYY-MM-DD'). If None, no upper bound.

    Returns
    -------
    DataFrame
        A DataFrame with columns ["Name", "Date", "Close"].
        - "Date" is converted to datetime.
        - Data is filtered by date range if specified.
        - Duplicates are removed.
        - Data is sorted by ["Date"].
    """
    df = pd.read_csv(csv_path, sep=",")
    
    df["Date"] = pd.to_datetime(df["Data"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Date", "Zamkniecie"])
    df["Name"] = Path(csv_path).stem  # Use filename (without extension) as the instrument name
    df = df[["Name", "Date", "Zamkniecie"]]
    df = df.rename(columns={"Zamkniecie": "Close"})
    
    # Filter by date range if specified
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df["Date"] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["Date"] <= end_date]
    
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    
    return df

def compute_log_returns(dfs):
    """
    Computes log returns for all instruments.

    Parameters
    ----------
    dfs : list of DataFrames
        Each DataFrame must contain ["Name", "Date", "Close"] columns.

    Returns
    -------
    DataFrame
        Pivoted DataFrame of log returns with:
        - Index = Date
        - Columns = instrument names
        - Values = log returns
    """
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort_values(by=["Name", "Date"], inplace=True)
    df_all["LogReturn"] = df_all.groupby("Name")["Close"].transform(lambda x: np.log(x).diff())
    df_all = df_all.dropna(subset=["LogReturn"])
    returns_df = df_all.pivot_table(index="Date", columns="Name", values="LogReturn", aggfunc="mean")
    return returns_df

# ============================================================== TAILCOR ===========================================================

def TailCoR(df, zeta=ZETA, tau=TAU, mode="tailcor"):
    """
    Computes TailCoR (or its components) correlation matrix for a given DataFrame of log returns.

    Parameters
    ----------
    df : DataFrame
        Index = dates, columns = instruments, values = log returns.
    zeta : float
        Parameter from publication.
    tau : float
        Parameter from publication.
    mode : {"tailcor", "linear", "nonlinear"}
        Which correlation type to compute.

    Returns
    -------
    DataFrame
        Symmetric correlation matrix of the chosen type.
    """
    cols = df.columns
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(n):
        for j in range(i, n):
            data = pd.concat([df.iloc[:, i], df.iloc[:, j]], axis=1).dropna()
            if data.shape[0] < 10:
                val = np.nan
            else:
                x_q = data.iloc[:, 0].quantile([tau, 1 - tau])
                y_q = data.iloc[:, 1].quantile([tau, 1 - tau])
                iqr_x = x_q[tau] - x_q[1 - tau]
                iqr_y = y_q[tau] - y_q[1 - tau]

                if iqr_x <= 1e-6 or iqr_y <= 1e-6:
                    val = np.nan
                else:
                    X = (data.iloc[:, 0] - data.iloc[:, 0].median()) / iqr_x
                    Y = (data.iloc[:, 1] - data.iloc[:, 1].median()) / iqr_y
                    rho = X.corr(Y)

                    if pd.isna(rho):
                        val = np.nan
                    else:
                        Z = (X + Y) / np.sqrt(2) if rho >= 0 else (X - Y) / np.sqrt(2)
                        q_upper = Z.quantile(zeta)
                        q_lower = Z.quantile(1 - zeta)
                        IQR_tail = q_upper - q_lower

                        if IQR_tail <= 1e-6:
                            val = np.nan
                        else:
                            sg = norm.ppf(tau) / norm.ppf(zeta)
                            tailcor = sg * IQR_tail
                            linear_component = np.sqrt(1 + abs(rho))
                            nonlinear_component = (
                                tailcor / linear_component
                                if linear_component > 1e-6 else np.nan
                            )

                            if mode == "tailcor":
                                val = tailcor
                            elif mode == "linear":
                                val = linear_component
                            elif mode == "nonlinear":
                                val = nonlinear_component
                            else:
                                raise ValueError("mode must be one of {'tailcor','linear','nonlinear'}")

            matrix.iat[i, j] = matrix.iat[j, i] = val

    return matrix

# ============== Pair TailCor ====================
def compute_TailCor_pairs(file_dfs, zeta=ZETA, tau=TAU):
    """
    Computes TailCoR, linear, and nonlinear matrices for each pair of instruments.

    Parameters
    ----------
    file_dfs : list of (str, DataFrame)
        List of tuples where:
        - First element = name (file or instrument identifier)
        - Second element = DataFrame with ["Name", "Date", "Close"].
    zeta : float
        Parameter from publication.
    tau : float
        Parameter from publication.

    Returns
    -------
    list of dict
        Each dict contains:
        - "zip1", "zip2" : names of the compared instruments
        - "tailcor_matrix" : TailCoR matrix
        - "linear_matrix" : linear component matrix
        - "nonlinear_matrix" : nonlinear component matrix
    """
    results = []
    all_pairs = list(combinations(file_dfs, 2))
    total_pairs = len(all_pairs)

    for idx, ((name1, df1), (name2, df2)) in enumerate(all_pairs, start=1):
        combined = pd.concat([df1, df2], ignore_index=True)
        returns_df = compute_log_returns([combined])

        tailcor_matrix = TailCoR(returns_df, zeta, tau, mode="tailcor")
        linear_matrix = TailCoR(returns_df, zeta, tau, mode="linear")
        nonlinear_matrix = TailCoR(returns_df, zeta, tau, mode="nonlinear")

        results.append({
            "zip1": name1,
            "zip2": name2,
            "tailcor_matrix": tailcor_matrix,
            "linear_matrix": linear_matrix,
            "nonlinear_matrix": nonlinear_matrix,
        })

    return results

# ================ Combine Matrix ===================
def build_combined_tailcor_matrices(pairwise_results):
    """
    Builds combined TailCoR, linear, and nonlinear matrices
    from pairwise results.

    Parameters
    ----------
    pairwise_results : list of dict
        Output of `compute_TailCor_pairs`.

    Returns
    -------
    tuple of DataFrames
        (combined_tailcor, combined_linear, combined_nonlinear)
        Each matrix has instruments as rows/columns.
    """
    all_names = set()
    for res in pairwise_results:
        all_names.update(res['tailcor_matrix'].columns)

    all_names = sorted(list(all_names))
    combined_tailcor = pd.DataFrame(np.nan, index=all_names, columns=all_names)
    combined_linear = pd.DataFrame(np.nan, index=all_names, columns=all_names)
    combined_nonlinear = pd.DataFrame(np.nan, index=all_names, columns=all_names)

    for res in pairwise_results:
        for matrix_name in ['tailcor_matrix', 'linear_matrix', 'nonlinear_matrix']:
            partial = res[matrix_name]
            target_matrix = {
                'tailcor_matrix': combined_tailcor,
                'linear_matrix': combined_linear,
                'nonlinear_matrix': combined_nonlinear,
            }[matrix_name]
            for i in partial.index:
                for j in partial.columns:
                    target_matrix.loc[i, j] = partial.loc[i, j]

    return combined_tailcor, combined_linear, combined_nonlinear

# ============== Average TailCor over Time =================
def avg_TailCor(df_returns, window_size=None, step=None, zeta=ZETA, tau=TAU):
    """
    Computes average TailCoR, linear, and nonlinear values
    over rolling windows.

    Parameters
    ----------
    df_returns : DataFrame
        Log returns DataFrame (Date index, instruments as columns).
    window_size : int, optional
        Number of days in each window. If None, calculated to get ~50 windows.
    step : int, optional
        Step size for rolling windows. If None, calculated as window_size // 2.
    zeta : float
        Parameter from publication.
    tau : float
        Parameter from publication.

    Returns
    -------
    DataFrame
        Time series of average TailCoR, linear, and nonlinear values.
    """
    # Auto-calculate window_size and step if not provided
    if window_size is None or step is None:
        window_size, step = calculate_default_window_params(len(df_returns))
        print(f"Auto-calculated: window_size={window_size}, step={step}")
    
    dates = df_returns.index
    avg_tailcors = []
    
    total_windows = (len(dates) - window_size) // step + 1
    print(f"Calculating TailCoR over {total_windows} windows:")
    print(f"  Data points: {len(dates)}, Window size: {window_size} days, Step: {step} days")

    for window_count, start in enumerate(range(0, len(dates) - window_size + 1, step), start=1):
        end = start + window_size
        window_returns = df_returns.iloc[start:end]
        window_returns = window_returns.loc[:, window_returns.notna().sum() > 0]
        window_returns = window_returns.dropna(thresh=int(0.8 * window_returns.shape[1]))

        if window_returns.shape[1] < MIN_REQUIRED_INSTRUMENTS:
            print(f"Skipped window {window_count}/{total_windows} – too few instruments ({window_returns.shape[1]})")
            continue

        tailcor_matrix = TailCoR(window_returns, zeta, tau, mode="tailcor")
        linear_matrix = TailCoR(window_returns, zeta, tau, mode="linear")
        nonlinear_matrix = TailCoR(window_returns, zeta, tau, mode="nonlinear")

        mask = np.triu(np.ones(tailcor_matrix.shape), k=1).astype(bool)

        tailcors = tailcor_matrix.where(mask).values.flatten()
        linear_vals = linear_matrix.where(mask).values.flatten()
        nonlinear_vals = nonlinear_matrix.where(mask).values.flatten()

        tailcors = tailcors[~np.isnan(tailcors)]
        linear_vals = linear_vals[~np.isnan(linear_vals)]
        nonlinear_vals = nonlinear_vals[~np.isnan(nonlinear_vals)]

        if len(tailcors) == 0:
            print(f"Skipped window {window_count}/{total_windows} – no TailCoR values")
            continue

        avg_tailcors.append({
            "date": dates[end - 1],
            "avg_tailcor": np.mean(tailcors),
            "avg_linear": np.mean(linear_vals) if len(linear_vals) > 0 else np.nan,
            "avg_nonlinear": np.mean(nonlinear_vals) if len(nonlinear_vals) > 0 else np.nan,
        })

    if len(avg_tailcors) == 0:
        print("Warning: No valid windows found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["avg_tailcor", "avg_linear", "avg_nonlinear"])

    df_avg = pd.DataFrame(avg_tailcors)
    df_avg.set_index("date", inplace=True)
    return df_avg

# ============================================================= PLOTTING FUNCTIONS =====================================================
def plot_clustermap(matrix, title, filename, annot=True):
    """
    Plots and saves a hierarchical clustered heatmap of the given matrix.

    Parameters
    ----------
    matrix : DataFrame
        Correlation matrix (TailCoR or its components).
    title : str
        Title for the plot.
    filename : str
        Path where the figure will be saved.
    annot : bool, default=True
        Whether to annotate each cell with its value.
    """
    clean_matrix = matrix.fillna(0)
    fig_w, fig_h = 10, 10
    cbar_pos = (0.92, 0.2, 0.02, 0.55)

    g = sns.clustermap(
        clean_matrix,
        cmap="vlag",
        linewidths=0.5,
        figsize=(fig_w, fig_h),
        cbar_kws={'label': title},
        metric="euclidean",
        method="ward",
        cbar_pos=cbar_pos
    ) 
    g.fig.suptitle(title, y=0.95, fontsize=14)
    g.fig.subplots_adjust(left=0.18, right=0.80, top=0.92, bottom=0.12)
    g.cax.set_position([0.92, 0.2, 0.02, 0.55])

    if annot:
        ax = g.ax_heatmap
        for i, row in enumerate(clean_matrix.index):
            for j, col in enumerate(clean_matrix.columns):
                val = clean_matrix.loc[row, col]
                ax.text(
                    j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=6
                )

    g.fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(g.fig)

def plot_avg_tailcor(df_avg, title="Average TailCoR Evolution", filename="avg_tailcor.png"):
    """
    Plots the time evolution of average TailCoR, linear, and nonlinear values.

    Parameters
    ----------
    df_avg : DataFrame
        DataFrame with average values computed by `avg_TailCor`.
    title : str
        Title for the plot.
    filename : str
        Path where the figure will be saved.
    """
    if df_avg.empty:
        print("Warning: DataFrame is empty, no plot created.")
        return
    
    plt.figure(figsize=(12, 5))
    plt.plot(df_avg.index, df_avg["avg_tailcor"], label="Average TailCoR", linestyle="-")
    plt.plot(df_avg.index, df_avg["avg_linear"], label="Linear Component", linestyle="--")
    plt.plot(df_avg.index, df_avg["avg_nonlinear"], label="Nonlinear Component", linestyle=":")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(filename))
    plt.close()

# ================================================================== EXECUTION =======================================================================

# NOTE: Window size and step should be adjusted based on your data size:
#   - If you have less data (e.g., 1-2 years), use smaller values (e.g., WINDOW_SIZE=30, STEP=10)
#   - For publication-level analysis with 3+ years of data, use: WINDOW_SIZE=780, STEP=252
#   - Step should typically be <= WINDOW_SIZE for overlapping windows and smoother evolution curves


# SET TO NONE FOR AUTO-CALCULATION BASED ON DATA SIZE
WINDOW_SIZE = None  # Window size in days (publication value = 780)
STEP = None # Step size (publication value = 252)



def main():
    # ============ Loading data ==========================
    folder_path = Path("/path/to/your/data_directory")  # Path to directory containing .csv files
    save_path = Path("/path/to/your/results_directory")  # Directory where results will be saved
    save_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    
    # Set date range for filtering data (set to None to use all data)
    start_date = "2022-01-01"  # Set to None to include all earlier dates
    end_date = None    # Set to None to include all later dates
    
    csv_files = [p for p in folder_path.glob("*.csv")]
    dfs = []

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"Loading data {idx}/{len(csv_files)}: {csv_path.name}")
        df = load_csv_to_df(csv_path, start_date=start_date, end_date=end_date)
        print(f"  Loaded {len(df)} rows, Date range: {df['Date'].min()} to {df['Date'].max()}")
        dfs.append(df)

    # ============= Calculations ============================
    returns = compute_log_returns(dfs)
    file_dfs = list(zip([p.name for p in csv_files], dfs))
    pair_results = compute_TailCor_pairs(file_dfs)
    combined_tailcor, combined_linear, combined_nonlinear = build_combined_tailcor_matrices(pair_results)

    # =============== Saving Data =======================
    combined_tailcor.to_csv(save_path / "tailcor_matrix.csv", encoding="utf-8")
    combined_linear.to_csv(save_path / "linear_component_matrix.csv", encoding="utf-8")
    combined_nonlinear.to_csv(save_path / "nonlinear_component_matrix.csv", encoding="utf-8")

    # ============== Plotting ================================
    plot_clustermap(combined_tailcor, "TailCor", save_path / "tailcor_clustermap.png")
    plot_clustermap(combined_linear, "Linear Component", save_path / "linear_clustermap.png")
    plot_clustermap(combined_nonlinear, "Nonlinear Component", save_path / "nonlinear_clustermap.png")

    df_avg_tailcor = avg_TailCor(returns, window_size=WINDOW_SIZE, step=STEP)
    plot_avg_tailcor(df_avg_tailcor, filename=save_path / "avg_tailcor.png")


if __name__ == "__main__":
    main()