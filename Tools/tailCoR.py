import numpy as np
import pandas as pd
from scipy.stats import norm

def tailCoR(df, zeta=0.75, tau=0.95, mode="tailcor"):
    """
    Twoja funkcja wyliczająca macierz TailCoR.
    """
    cols = df.columns
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(n):
        for j in range(i, n):
            # i == j to przekątna (zmienna sama ze sobą)
            if i == j:
                matrix.iat[i, j] = 1.0 # Korelacja własna (dla uproszczenia macierzy)
                continue

            data = pd.concat([df.iloc[:, i], df.iloc[:, j]], axis=1).dropna()

            # Zabezpieczenie przed zbyt małą liczbą danych
            if data.shape[0] < 20:
                val = np.nan
            else:
                try:
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
                            # Zabezpieczenie sqrt
                            term = (X + Y) / np.sqrt(2) if rho >= 0 else (X - Y) / np.sqrt(2)
                            Z = term

                            q_upper = Z.quantile(zeta)
                            q_lower = Z.quantile(1 - zeta)
                            IQR_tail = q_upper - q_lower

                            if IQR_tail <= 1e-6:
                                val = np.nan
                            else:
                                sg = norm.ppf(tau) / norm.ppf(zeta)
                                tailcor_val = sg * IQR_tail

                                linear_component = np.sqrt(1 + abs(rho))
                                nonlinear_component = (
                                    tailcor_val / linear_component
                                    if linear_component > 1e-6 else np.nan
                                )

                                if mode == "tailcor":
                                    val = tailcor_val
                                elif mode == "linear":
                                    val = linear_component
                                elif mode == "nonlinear":
                                    val = nonlinear_component
                                else:
                                    raise ValueError("mode error")
                except Exception as e:
                    # Catch-all dla błędów numerycznych w pętli
                    val = np.nan

            matrix.iat[i, j] = matrix.iat[j, i] = val

    return matrix