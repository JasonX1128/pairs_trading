import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

class CrudeOilArbitrageHMM:
    def __init__(self, n_states=2, bandwidth_alpha=0.20, batch_m=10, prob_i_window=20):
        self.N = n_states
        self.alpha_bw = bandwidth_alpha
        self.m = batch_m 
        self.n = prob_i_window
        
        # Model Parameters (Regime-switching AR(1))
        self.pi = None      # Transition matrix
        self.gamma = None   # Intercepts
        self.phi = None     # AR coefficients
        self.eta = None     # Volatilities
        
        # Cointegration Parameters
        self.lambda_vec = None
        self.lambda_0 = None
        
        # Trackers for Online EM (integrals for M-step updates)
        self.T_count = np.zeros(self.N)
        self.T_S = np.zeros(self.N)
        self.T_S_prev = np.zeros(self.N)
        self.T_S2 = np.zeros(self.N)
        self.T_S2_prev = np.zeros(self.N)
        self.T_SS_prev = np.zeros(self.N)
        self.J_tracker = np.zeros((self.N, self.N))

    def fit_cointegration(self, df):
        """Johansen test to find the long-term equilibrium spread."""
        res = coint_johansen(df, det_order=0, k_ar_diff=1)
        # First eigenvector normalized to Brent (first column)
        evec = res.evec[:, 0]
        self.lambda_vec = evec / evec[0]
        
        # Calculate lambda_0 (intercept) to mean-zero the spread
        raw_spread = df @ self.lambda_vec
        self.lambda_0 = -np.mean(raw_spread)
        return raw_spread + self.lambda_0

    def initialize_params(self, spread_init):
        """OLS initialization using the first 20 trading days."""
        y = spread_init.iloc[1:21].values
        x = spread_init.iloc[0:20].values
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        gamma_ols, phi_ols = beta[0], beta[1]
        eta_ols = np.std(y - (gamma_ols + phi_ols * x))
        
        # Seed two regimes with different volatility/intercept levels
        self.gamma = np.array([gamma_ols * 1.1, gamma_ols * 0.9])
        self.phi = np.array([phi_ols, phi_ols])
        self.eta = np.array([eta_ols * 1.2, eta_ols * 0.8])
        self.pi = np.array([[0.9, 0.1], [0.1, 0.9]])

    def _e_step(self, s_curr, s_prev, x_hat):
        """Recursive filtering Equation 2.6 to update state probabilities."""
        # Numerical guards: avoid division by zero / NaNs in densities
        eps = 1e-8
        eta_safe = np.where(self.eta <= 0, eps, self.eta)

        diff = s_curr - (self.gamma + self.phi * s_prev)
        # exponent can underflow; clip to reasonable numeric range
        exponent = -0.5 * (diff**2) / (eta_safe**2)
        exponent = np.clip(exponent, -700, 700)

        # Density components for each regime (Gaussian pdf)
        d = np.exp(exponent) / (eta_safe * np.sqrt(2 * np.pi))

        # Filtered state probability
        x_next_unnorm = self.pi.T @ (d * x_hat)
        denom = np.sum(x_next_unnorm)
        if not np.isfinite(denom) or denom <= 0:
            # fallback: keep previous belief or uniform if that is invalid
            x_next = x_hat.copy()
            if not np.all(np.isfinite(x_next)):
                x_next = np.ones(self.N) / self.N
        else:
            x_next = x_next_unnorm / denom

        # final safety: replace any NaN/inf and renormalize
        x_next = np.nan_to_num(x_next, nan=1.0/self.N, posinf=1.0/self.N, neginf=1.0/self.N)
        s = x_next.sum()
        if s <= 0 or not np.isfinite(s):
            x_next = np.ones(self.N) / self.N
        else:
            x_next = x_next / np.sum(x_next)
        
        # Accumulate trackers for the M-step (only finite values)
        x_next_for_accum = np.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0)
        self.T_count += x_next_for_accum
        self.T_S += x_next_for_accum * s_curr
        self.T_S_prev += x_next_for_accum * s_prev
        self.T_S2 += x_next_for_accum * (s_curr**2)
        self.T_S2_prev += x_next_for_accum * (s_prev**2)
        self.T_SS_prev += x_next_for_accum * (s_curr * s_prev)
        self.J_tracker += np.outer(x_next_for_accum, x_hat) # Jump estimation
        
        return x_next

    def _m_step(self):
        """Online EM Parameter Update Equation 2.7."""
        # Update Transition Matrix
        self.pi = (self.J_tracker / self.T_count[:, None]).T
        # Normalize rows to sum to 1
        self.pi = self.pi / self.pi.sum(axis=1)[:, None]
        
        # Update Regime Parameters (Weighted OLS)
        for i in range(self.N):
            denom = (self.T_count[i] * self.T_S2_prev[i] - self.T_S_prev[i]**2)
            # Safeguard against zero denominator or insufficient data for OLS
            if denom > 1e-8 and self.T_count[i] > 5:
                self.phi[i] = (self.T_count[i] * self.T_SS_prev[i] - self.T_S[i] * self.T_S_prev[i]) / denom
                self.gamma[i] = (self.T_S[i] - self.phi[i] * self.T_S_prev[i]) / self.T_count[i]
                
                resid_sq = (self.T_S2[i] + (self.gamma[i]**2) * self.T_count[i] + (self.phi[i]**2) * self.T_S2_prev[i] - 
                            2 * self.gamma[i] * self.T_S[i] - 2 * self.phi[i] * self.T_SS_prev[i] + 
                            2 * self.gamma[i] * self.phi[i] * self.T_S_prev[i])
                self.eta[i] = np.sqrt(np.abs(resid_sq / self.T_count[i]))

    def get_signal(self, strategy, t, spread, x_hat):
        s_curr, s_prev = spread.iloc[t], spread.iloc[t-1]
        q = abs(norm.ppf(self.alpha_bw / 2))
        
        # Use a rolling window for historical increments to avoid permanently inflated thresholds
        lookback = min(t, 100) # 100-day rolling window
        
        if strategy == 'PV':
            return -1 if s_curr > 0 else 1
            
        elif strategy == 'ProbI':
            if t < self.n: return 0
            window = spread.iloc[t-self.n:t]
            mu, std = np.mean(window), np.std(window)
            if s_curr > (mu + q * std): return -1
            if s_curr < (mu - q * std): return 1
            
        elif strategy == 'PredI':
            e_s = np.dot(x_hat, self.gamma + self.phi * s_prev)
            # Safeguard: Blend HMM volatility with empirical volatility to prevent infinite bands
            empirical_std = np.std(spread.iloc[t-lookback:t]) if t > 10 else 0
            std_s = min(np.sqrt(np.dot(x_hat, self.eta**2)), empirical_std * 2) 
            
            if s_curr > (e_s + q * std_s): return -1
            if s_curr < (e_s - q * std_s): return 1
            
        elif strategy == 'RI':
            if t < 2: return 0
            x_t = s_curr - s_prev
            # Rolling window for percentiles
            hist_inc = np.abs(spread.iloc[t-lookback+1:t].values - spread.iloc[t-lookback:t-1].values)
            q_val = np.percentile(hist_inc, 100 * (1 - self.alpha_bw)) if len(hist_inc) > 10 else 999
            if x_t > q_val: return -1
            if x_t < -q_val: return 1
            
        elif strategy == 'PI':
            if t < 1: return 0
            e_next = np.dot(x_hat, self.gamma + self.phi * s_curr)
            pred_inc = e_next - s_curr
            # Rolling window for percentiles
            hist_inc = np.abs(spread.iloc[t-lookback+1:t].values - spread.iloc[t-lookback:t-1].values)
            q_val = np.percentile(hist_inc, 100 * (1 - self.alpha_bw)) if len(hist_inc) > 10 else 999
            
            # FIXED LOGIC: Buy when predicting an increase, sell when predicting a decrease
            if pred_inc > q_val: return 1   
            if pred_inc < -q_val: return -1 
            
        return 0

    def run_backtest(self, strategy, df_prices, window_size=100, warmup=20):
        """The main loop: Cointegration -> Initialize -> Filter -> Trade -> M-Step."""
        # Fit initial cointegration on the first 'window_size' data points to avoid lookahead
        init_prices = df_prices.iloc[:max(window_size, warmup)]
        self.fit_cointegration(init_prices)
        
        # Initial spread for parameters
        spread_init = pd.Series(np.dot(init_prices, self.lambda_vec) + self.lambda_0, index=init_prices.index)
        self.initialize_params(spread_init)
        
        T = len(df_prices)
        signals = [0]
        position = 0
        x_hat = np.array([0.5, 0.5])
        
        # Trackers
        spread_series = pd.Series(index=df_prices.index, dtype=float)
        spread_series.iloc[0] = np.dot(df_prices.iloc[0], self.lambda_vec) + self.lambda_0
        lambda_history = [(self.lambda_vec.copy(), self.lambda_0)]
        
        for t in range(1, T):
            # Rolling update: re-fit cointegration on the lookback window
            if t >= window_size:
                self.fit_cointegration(df_prices.iloc[t-window_size:t])
            
            # Recalculate current spread values using CURRENT rolling lambda
            s_curr = np.dot(df_prices.iloc[t], self.lambda_vec) + self.lambda_0
            s_prev = np.dot(df_prices.iloc[t-1], self.lambda_vec) + self.lambda_0
            
            spread_series.iloc[t] = s_curr
            lambda_history.append((self.lambda_vec.copy(), self.lambda_0))

            # 1. Update Filter (E-Step) using rolling values
            x_hat = self._e_step(s_curr, s_prev, x_hat)
            
            # 2. Update Parameters every m steps (M-Step)
            if t % self.m == 0:
                self._m_step()
            
            # 3. Generate Trading Signal (respecting warmup)
            if t > warmup:
                if position == 0:
                    position = self.get_signal(strategy, t, spread_series, x_hat)
                elif (position == 1 and s_curr >= 0) or (position == -1 and s_curr <= 0):
                    position = 0
            else:
                position = 0
            
            signals.append(position)
            
        return signals, spread_series, lambda_history


def evaluate_performance(signals, prices, lambda_history):
    """Calculates returns per unit of gross notional exposure."""
    # Transaction costs (bps): Brent 5.8, Shanghai 53.71, WTI 20.24
    costs = np.array([5.80, 53.71, 20.24]) / 10000
    daily_rets = []
    
    for t in range(1, len(signals)):
        # Use lambda from t-1 to prevent lookahead in PnL calculation
        lambda_vec, lambda_0 = lambda_history[t-1]
        
        # Gross Exposure G_t-1
        g_t_prev = np.abs(lambda_0) + np.sum(np.abs(lambda_vec) * prices.iloc[t-1])
        
        # PnL from price changes
        p_delta = prices.iloc[t] - prices.iloc[t-1]
        pnl = signals[t-1] * np.sum(lambda_vec * p_delta)
        
        # Fees on position changes
        fee = np.sum(np.abs(signals[t] - signals[t-1]) * costs * prices.iloc[t] * np.abs(lambda_vec))
        
        daily_rets.append((pnl - fee) / g_t_prev)
        
    return pd.Series(daily_rets)