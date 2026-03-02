import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

class CrudeOilArbitrageHMM:
    def __init__(self, n_states=2, bandwidth_alpha=0.20, batch_m=10, prob_i_window=20, fixed_hmm=True):
        self.N = n_states
        self.alpha_bw = bandwidth_alpha
        self.m = batch_m 
        self.n = prob_i_window
        self.fixed_hmm = fixed_hmm # Flag to toggle mathematically strict HMM regimes
        
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

    def fit_cointegration(self, df, prev_lambda=None):
        """Johansen test with max-weight normalization and sign-flip prevention."""
        res = coint_johansen(df, det_order=0, k_ar_diff=1)
        evec = res.evec[:, 0]
        
        # Max absolute weight normalization (preserves relative signs)
        idx = np.argmax(np.abs(evec))
        new_lambda = evec / np.abs(evec[idx])
        
        # SIGN ALIGNMENT: Prevent structural breaks by ensuring the vector 
        # points in the same economic direction as yesterday
        if prev_lambda is not None:
            if np.dot(new_lambda, prev_lambda) < 0:
                new_lambda = -new_lambda
                
        self.lambda_vec = new_lambda
        raw_spread = df @ self.lambda_vec
        self.lambda_0 = -np.mean(raw_spread)
        return raw_spread + self.lambda_0

    def initialize_params(self, spread_init):
        y = spread_init.iloc[1:21].values
        x = spread_init.iloc[0:20].values
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        gamma_ols, phi_ols = beta[0], beta[1]
        eta_ols = np.std(y - (gamma_ols + phi_ols * x))
        
        if self.fixed_hmm:
            # Force a strong initial distinction between High-Vol and Low-Vol states
            self.gamma = np.array([gamma_ols + eta_ols, gamma_ols - eta_ols])
            self.phi = np.array([phi_ols, phi_ols])
            self.eta = np.array([eta_ols * 2.0, eta_ols * 0.5])
            self.pi = np.array([[0.95, 0.05], [0.05, 0.95]])
        else:
            # Seed two regimes with different volatility/intercept levels (Highly profitable baseline)
            self.gamma = np.array([gamma_ols * 1.1, gamma_ols * 0.9])
            self.phi = np.array([phi_ols, phi_ols])
            self.eta = np.array([eta_ols * 1.2, eta_ols * 0.8])
            self.pi = np.array([[0.9, 0.1], [0.1, 0.9]])

    def _e_step(self, s_curr, s_prev, x_hat):
        eps = 1e-8
        eta_safe = np.where(self.eta <= 0, eps, self.eta)

        diff = s_curr - (self.gamma + self.phi * s_prev)
        exponent = -0.5 * (diff**2) / (eta_safe**2)
        exponent = np.clip(exponent, -700, 700)

        d = np.exp(exponent) / (eta_safe * np.sqrt(2 * np.pi))

        if self.fixed_hmm:
            # EXACT MATH: Calculate the joint probability matrix xi[i, j] 
            xi = (x_hat * d)[:, None] * self.pi
            
            x_next_unnorm = xi.sum(axis=0)
            denom = np.sum(x_next_unnorm)
            
            if denom > 0 and np.isfinite(denom):
                x_next = x_next_unnorm / denom
                xi_norm = xi / denom
            else:
                x_next = np.ones(self.N) / self.N
                xi_norm = self.pi.copy() / self.N

            weight_i = xi_norm.sum(axis=1)
            
            # Exponential Moving Average (EMA) Trackers with a ~200 day half-life
            rho = 0.995 
            self.T_count = rho * self.T_count + weight_i
            self.T_S = rho * self.T_S + weight_i * s_curr
            self.T_S_prev = rho * self.T_S_prev + weight_i * s_prev
            self.T_S2 = rho * self.T_S2 + weight_i * (s_curr**2)
            self.T_S2_prev = rho * self.T_S2_prev + weight_i * (s_prev**2)
            self.T_SS_prev = rho * self.T_SS_prev + weight_i * (s_curr * s_prev)
            self.J_tracker = rho * self.J_tracker + xi_norm 
            
            return x_next
        else:
            # Original infinite accumulation (Profitable baseline)
            x_next_unnorm = self.pi.T @ (d * x_hat)
            denom = np.sum(x_next_unnorm)
            if not np.isfinite(denom) or denom <= 0:
                x_next = x_hat.copy()
                if not np.all(np.isfinite(x_next)):
                    x_next = np.ones(self.N) / self.N
            else:
                x_next = x_next_unnorm / denom

            x_next = np.nan_to_num(x_next, nan=1.0/self.N, posinf=1.0/self.N, neginf=1.0/self.N)
            s = x_next.sum()
            if s <= 0 or not np.isfinite(s):
                x_next = np.ones(self.N) / self.N
            else:
                x_next = x_next / np.sum(x_next)
            
            x_next_for_accum = np.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.T_count += x_next_for_accum
            self.T_S += x_next_for_accum * s_curr
            self.T_S_prev += x_next_for_accum * s_prev
            self.T_S2 += x_next_for_accum * (s_curr**2)
            self.T_S2_prev += x_next_for_accum * (s_prev**2)
            self.T_SS_prev += x_next_for_accum * (s_curr * s_prev)
            self.J_tracker += np.outer(x_next_for_accum, x_hat) 
            
            return x_next

    def _m_step(self):
        if self.fixed_hmm:
            self.pi = self.J_tracker / self.J_tracker.sum(axis=1, keepdims=True)
            self.pi = np.clip(self.pi, 1e-4, 1.0)
            self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)
        else:
            self.pi = (self.J_tracker / self.T_count[:, None]).T
            self.pi = self.pi / self.pi.sum(axis=1)[:, None]
        
        for i in range(self.N):
            denom = (self.T_count[i] * self.T_S2_prev[i] - self.T_S_prev[i]**2)
            if denom > 1e-8 and self.T_count[i] > 5:
                self.phi[i] = (self.T_count[i] * self.T_SS_prev[i] - self.T_S[i] * self.T_S_prev[i]) / denom
                self.gamma[i] = (self.T_S[i] - self.phi[i] * self.T_S_prev[i]) / self.T_count[i]
                
                resid_sq = (self.T_S2[i] + (self.gamma[i]**2) * self.T_count[i] + (self.phi[i]**2) * self.T_S2_prev[i] - 
                            2 * self.gamma[i] * self.T_S[i] - 2 * self.phi[i] * self.T_SS_prev[i] + 
                            2 * self.gamma[i] * self.phi[i] * self.T_S_prev[i])
                
                self.eta[i] = np.sqrt(np.abs(resid_sq / self.T_count[i]))
                self.eta[i] = np.clip(self.eta[i], 0.001, 10.0)

        if self.fixed_hmm:
            # LABEL SWAP: Ensure State 0 is consistently the High-Volatility regime for the plots
            if self.N == 2 and self.eta[0] < self.eta[1]:
                self.eta[[0, 1]] = self.eta[[1, 0]]
                self.gamma[[0, 1]] = self.gamma[[1, 0]]
                self.phi[[0, 1]] = self.phi[[1, 0]]
                self.pi[[0, 1]] = self.pi[[1, 0]]
                self.pi[:, [0, 1]] = self.pi[:, [1, 0]]
                self.T_count[[0, 1]] = self.T_count[[1, 0]]
                self.T_S[[0, 1]] = self.T_S[[1, 0]]
                self.T_S_prev[[0, 1]] = self.T_S_prev[[1, 0]]
                self.T_S2[[0, 1]] = self.T_S2[[1, 0]]
                self.T_S2_prev[[0, 1]] = self.T_S2_prev[[1, 0]]
                self.T_SS_prev[[0, 1]] = self.T_SS_prev[[1, 0]]
                self.J_tracker[[0, 1]] = self.J_tracker[[1, 0]]
                self.J_tracker[:, [0, 1]] = self.J_tracker[:, [1, 0]]

    def get_signal(self, strategy, t, spread, x_hat):
        s_curr, s_prev = spread.iloc[t], spread.iloc[t-1]
        
        # Base quantile multiplier
        q_base = abs(norm.ppf(self.alpha_bw / 2))
        lookback = min(t, 100) 
        signal = 0
        
        if strategy == 'PV':
            if s_curr > 0: signal = -1
            elif s_curr < 0: signal = 1
            
        elif strategy == 'ProbI':
            if t >= self.n:
                window = spread.iloc[t-self.n:t]
                mu, std = np.mean(window), np.std(window)
                if s_curr > (mu + q_base * std): signal = -1
                elif s_curr < (mu - q_base * std): signal = 1
            
        elif strategy == 'PredI':
            # OFFENSIVE HMM USAGE: Use the HMM strictly for the expected value prediction
            e_s = np.dot(x_hat, self.gamma + self.phi * s_prev)
            
            # Decouple the band width from the HMM's variance (eta). 
            # Use strict empirical market reality so bands don't artificially explode.
            empirical_std = np.std(spread.iloc[t-lookback:t]) if t > 10 else 0
            
            # REGIME-DEPENDENT AGGRESSION: If the HMM is highly confident we are in the 
            # High-Vol state (State 0), reduce the entry threshold by up to 30%.
            # This allows the strategy to actively attack volatile mean-reverting swings.
            volatility_confidence = x_hat[0] if self.fixed_hmm else 0.0
            q_dynamic = q_base * (1.0 - (0.30 * volatility_confidence))
            
            if s_curr > (e_s + q_dynamic * empirical_std): signal = -1
            elif s_curr < (e_s - q_dynamic * empirical_std): signal = 1
            
        elif strategy == 'RI':
            if t >= 2:
                x_t = s_curr - s_prev
                hist_inc = np.abs(spread.iloc[t-lookback+1:t].values - spread.iloc[t-lookback:t-1].values)
                q_val = np.percentile(hist_inc, 100 * (1 - self.alpha_bw)) if len(hist_inc) > 10 else 999
                if x_t > q_val: signal = -1
                elif x_t < -q_val: signal = 1
            
        elif strategy == 'PI':
            if t >= 1:
                e_next = np.dot(x_hat, self.gamma + self.phi * s_curr)
                pred_inc = e_next - s_curr
                
                if t > 10:
                    hist_s = spread.iloc[t-lookback:t].values
                    e_next_hist = np.dot(x_hat, self.gamma[:, None] + self.phi[:, None] * hist_s[:-1])
                    hist_pred_inc = np.abs(e_next_hist - hist_s[:-1])
                    
                    # Apply regime-dependent aggression to the PI strategy as well
                    volatility_confidence = x_hat[0] if self.fixed_hmm else 0.0
                    dynamic_alpha = self.alpha_bw * (1.0 + (0.50 * volatility_confidence)) # increase alpha (lower percentile)
                    q_val = np.percentile(hist_pred_inc, 100 * (1 - dynamic_alpha))
                else:
                    q_val = 999
                
                if pred_inc > q_val: signal = 1   
                elif pred_inc < -q_val: signal = -1 
        
        # STRICT WRONG-SIDE FILTER
        if signal == 1 and s_curr >= 0: return 0
        if signal == -1 and s_curr <= 0: return 0
        return signal

    def run_backtest(self, strategy, df_prices, window_size=100, warmup=20):
        init_prices = df_prices.iloc[:max(window_size, warmup)]
        self.fit_cointegration(init_prices)
        
        spread_init = pd.Series(np.dot(init_prices, self.lambda_vec) + self.lambda_0, index=init_prices.index)
        self.initialize_params(spread_init)
        
        T = len(df_prices)
        signals = [0]
        position = 0
        x_hat = np.array([0.5, 0.5])
        
        spread_series = pd.Series(index=df_prices.index, dtype=float)
        spread_series.iloc[0] = np.dot(df_prices.iloc[0], self.lambda_vec) + self.lambda_0
        lambda_history = [(self.lambda_vec.copy(), self.lambda_0)]
        x_hat_history = [x_hat.copy()]
        
        for t in range(1, T):
            if t >= window_size:
                self.fit_cointegration(df_prices.iloc[t-window_size:t])
            
            s_curr = np.dot(df_prices.iloc[t], self.lambda_vec) + self.lambda_0
            s_prev = np.dot(df_prices.iloc[t-1], self.lambda_vec) + self.lambda_0
            
            spread_series.iloc[t] = s_curr
            lambda_history.append((self.lambda_vec.copy(), self.lambda_0))

            x_hat = self._e_step(s_curr, s_prev, x_hat)
            x_hat_history.append(x_hat.copy())
            
            if t % self.m == 0:
                self._m_step()
            
            if t > warmup:
                if position == 0:
                    position = self.get_signal(strategy, t, spread_series, x_hat)
                elif (position == 1 and s_curr >= 0) or (position == -1 and s_curr <= 0):
                    position = 0
            else:
                position = 0
            
            signals.append(position)
            
        return signals, spread_series, lambda_history, x_hat_history


def evaluate_performance(signals, prices, lambda_history):
    costs = np.array([5.80, 53.71, 20.24]) / 10000
    daily_rets = []
    
    for t in range(1, len(signals)):
        lambda_vec, lambda_0 = lambda_history[t-1]
        g_t_prev = np.abs(lambda_0) + np.sum(np.abs(lambda_vec) * prices.iloc[t-1])
        p_delta = prices.iloc[t] - prices.iloc[t-1]
        pnl = signals[t-1] * np.sum(lambda_vec * p_delta)
        fee = np.sum(np.abs(signals[t] - signals[t-1]) * costs * prices.iloc[t] * np.abs(lambda_vec))
        daily_rets.append((pnl - fee) / g_t_prev)
        
    return pd.Series(daily_rets)