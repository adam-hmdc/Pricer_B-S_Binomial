import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# -----------------------------
# Fonctions de calcul de prix
# -----------------------------
def bs_european(S, K, T, r, sigma, option_type='call', q=0):
    T = max(T, 1e-8)       # sécurités num.
    sigma = max(sigma, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

def binomial_european(S, K, T, r, sigma, n=100, option_type='call', q=0):
    T = max(T, 1e-8)
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # Prix en maturité
    ST = np.array([S * u**j * d**(n-j) for j in range(n+1)])
    if option_type == 'call':
        option_values = np.maximum(ST - K, 0.0)
    else:
        option_values = np.maximum(K - ST, 0.0)
    # Backward induction
    disc = np.exp(-r*dt)
    for _ in range(n):
        option_values = disc * (p*option_values[:-1] + (1-p)*option_values[1:])
    return float(option_values[0])

# -----------------------------
# Black-Scholes: Greeks fermés
# -----------------------------
def _d1_d2(S,K,T,r,sigma,q):
    T = max(T, 1e-8); sigma = max(sigma, 1e-8)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return d1, d2

def bs_delta(S,K,T,r,sigma,option_type='call',q=0):
    d1,_ = _d1_d2(S,K,T,r,sigma,q)
    if option_type=='call':
        return np.exp(-q*T)*norm.cdf(d1)
    return np.exp(-q*T)*(norm.cdf(d1)-1)

def bs_gamma(S,K,T,r,sigma,q=0):
    d1,_ = _d1_d2(S,K,T,r,sigma,q)
    return np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))

def bs_vega(S,K,T,r,sigma,q=0):
    d1,_ = _d1_d2(S,K,T,r,sigma,q)
    return S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)  # par 1.00 de sigma

def bs_theta(S,K,T,r,sigma,option_type='call',q=0):
    d1,d2 = _d1_d2(S,K,T,r,sigma,q)
    term1 = -(S*np.exp(-q*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
    if option_type=='call':
        term2 = -r*K*np.exp(-r*T)*norm.cdf(d2)
        term3 = +q*S*np.exp(-q*T)*norm.cdf(d1)
        return term1 + term2 + term3   # par an
    else:
        term2 = +r*K*np.exp(-r*T)*norm.cdf(-d2)
        term3 = -q*S*np.exp(-q*T)*norm.cdf(-d1)
        return term1 + term2 + term3

def bs_rho(S,K,T,r,sigma,option_type='call',q=0):
    _,d2 = _d1_d2(S,K,T,r,sigma,q)
    if option_type=='call':
        return K*T*np.exp(-r*T)*norm.cdf(d2)
    else:
        return -K*T*np.exp(-r*T)*norm.cdf(-d2)

# -----------------------------
# Différences finies (fallback pour le binomial)
# -----------------------------
def greek_fd(price_func, S,K,T,r,sigma, kind='delta'):
    # pas adaptatifs
    hS = max(1e-4*S, 1e-3)
    hT = min(0.01, max(1e-3, 0.1*T))
    hV = 1e-3
    hR = 1e-4
    if kind=='delta':
        return (price_func(S+hS,T,r,sigma) - price_func(S-hS,T,r,sigma))/(2*hS)
    if kind=='gamma':
        return (price_func(S+hS,T,r,sigma) - 2*price_func(S,T,r,sigma) + price_func(S-hS,T,r,sigma))/ (hS**2)
    if kind=='vega':
        return (price_func(S,T,r,sigma+hV) - price_func(S,T,r,sigma-hV))/(2*hV)
    if kind=='theta':  # dV/dt, convention theta < 0 pour le passage du temps
        t1 = max(1e-8, T+hT); t2 = max(1e-8, T-hT)
        return (price_func(S,t2,r,sigma) - price_func(S,t1,r,sigma))/(2*hT)
    if kind=='rho':
        return (price_func(S,T,r+hR,sigma) - price_func(S,T,r-hR,sigma))/(2*hR)
    return np.nan

# -----------------------------
# Interface
# -----------------------------
st.set_page_config(page_title="Prix d'Options", layout="wide")
linkedin_url = "https://www.linkedin.com/in/adam-hamadache-802172294/"
github_url = "https://github.com/adam-hmdc"
st.markdown(
    f"""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 100;">
        <a href={linkedin_url} target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30">
        </a>
        <a href="{github_url}" target="_blank" style="margin-left: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("👨‍💼 Priceur d'Options vanille Européennes")

# Layout en colonnes (1/4 - 3/4)
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("⚙️ Calcul du prix de l'option")
    model = st.selectbox("Modèle", ["Black-Scholes", "Binomial"])
    S = st.number_input("Prix spot actuel (S)", value=100.0)
    K = st.number_input("Strike (K)", value=100.0)
    T = st.number_input("Maturité en années (T)", value=1.0, step=0.1)
    r = st.number_input("Taux sans risque r", value=0.05, step=0.01)
    sigma = st.number_input("Volatilité du sous-jacent σ", value=0.2, step=0.01, min_value=0.0001)
    q = st.number_input("Taux de dividende q", value=0.0, step=0.01)
    if model == "Binomial":
        n = st.number_input("Nombre de pas (n)", min_value=1, value=100, step=1)
    
    st.markdown("---")  # ligne de séparation

    st.subheader("⚙️ Paramètres de sensibilité ")
    sigma_min = st.number_input("Volatilité min", value=0.05, step=0.01, min_value=0.0001)
    sigma_max = st.number_input("Volatilité max", value=1.0, step=0.01, min_value=0.0002)
    S_min = st.number_input("Prix spot min", value=50.0)
    S_max = st.number_input("Prix spot max", value=150.0)

    submitted = st.button("Calculer")

with col2:
    if submitted:
        # Choisir la fonction en fonction du modèle
        if model == "Black-Scholes":
            func = bs_european
        else:
            func = lambda S_,K_,T_,r_,sigma_,option_type_,q_: binomial_european(S_,K_,T_,r_,sigma_,n,option_type_,q_)

        # ---------------- Prix actuels ----------------
        call_price = func(S, K, T, r, sigma, 'call', q)
        put_price  = func(S, K, T, r, sigma, 'put',  q)

        st.markdown(f"### 📈 Prix **Call** : <span style='color:green;font-size:20px'>{call_price:.4f}</span>", unsafe_allow_html=True)
        st.markdown(f"### 📉 Prix **Put** : <span style='color:red;font-size:20px'>{put_price:.4f}</span>", unsafe_allow_html=True)

        # ---------------- Tableau des Greeks (pour les paramètres ACTUELS) ----------------
        st.markdown("---")
        st.subheader("📊 Tableau des Greeks (paramètres actuels)")

        def price_func_factory(opt_type):
            return lambda S_,T_,r_,sigma_: func(S_, K, T_, r_, sigma_, opt_type, q)

        rows = []
        for opt in ['call', 'put']:
            if model == "Black-Scholes":
                price = bs_european(S,K,T,r,sigma,opt,q)
                delta = bs_delta(S,K,T,r,sigma,opt,q)
                gamma = bs_gamma(S,K,T,r,sigma,q)
                vega  = bs_vega(S,K,T,r,sigma,q)
                theta = bs_theta(S,K,T,r,sigma,opt,q)
                rho   = bs_rho(S,K,T,r,sigma,opt,q)
            else:
                pf = price_func_factory(opt)
                price = pf(S,T,r,sigma)
                delta = greek_fd(pf,S,K,T,r,sigma,'delta')
                gamma = greek_fd(pf,S,K,T,r,sigma,'gamma')
                vega  = greek_fd(pf,S,K,T,r,sigma,'vega')
                theta = greek_fd(pf,S,K,T,r,sigma,'theta')
                rho   = greek_fd(pf,S,K,T,r,sigma,'rho')

            rows.append({
                "Option": opt.capitalize(),
                "Prix": price,
                "Delta": delta,
                "Gamma": gamma,
                "Vega (par 1.00)": vega,
                "Theta (an)": theta,
                "Rho": rho
            })

        df_greeks = pd.DataFrame(rows).set_index("Option")
        st.dataframe(
            df_greeks.style.format({
                "Prix": "{:.4f}", "Delta": "{:.4f}", "Gamma": "{:.6f}",
                "Vega (par 1.00)": "{:.4f}", "Theta (an)": "{:.4f}", "Rho": "{:.4f}"
            }),
            use_container_width=True
        )

        # ---------------- Heatmaps (Prix) ----------------
        st.markdown("---")
        st.subheader("🟩 Matrices de sensibilité (Prix)")
        S_values = np.linspace(S_min, S_max, 10)
        sigma_values = np.linspace(sigma_min, sigma_max, 10)
        Z_call = np.array([[func(s, K, T, r, sig, 'call', q) for s in S_values] for sig in sigma_values])
        Z_put  = np.array([[func(s, K, T, r, sig, 'put',  q) for s in S_values] for sig in sigma_values])

        fig_call_heatmap = go.Figure(data=go.Heatmap(
            z=Z_call, x=S_values, y=sigma_values,
            colorscale='Viridis',
            text=np.round(Z_call, 2),
            texttemplate="%{text}", textfont={"size":10, "color":"white"}
        ))
        fig_call_heatmap.update_layout(title="Matrice de Sensibilité du Call", xaxis_title="Prix Spot", yaxis_title="Volatilité σ")

        fig_put_heatmap = go.Figure(data=go.Heatmap(
            z=Z_put, x=S_values, y=sigma_values,
            colorscale='Plasma',
            text=np.round(Z_put, 2),
            texttemplate="%{text}", textfont={"size":10, "color":"white"}
        ))
        fig_put_heatmap.update_layout(title="Matrice de Sensibilité du Put", xaxis_title="Prix Spot", yaxis_title="Volatilité σ")

        st.plotly_chart(fig_call_heatmap, use_container_width=True)
        st.plotly_chart(fig_put_heatmap, use_container_width=True)

        # ---------------- Surfaces 3D (automatiques, sans menus) ----------------
        st.markdown("---")
        st.subheader("🧭 Surfaces 3D — Prix & Greeks (Call en haut, Put en bas)")

        # Paramètres de grille
        DENSITY = 24  # augmente/diminue pour qualité/rapidité
        sigma_range = np.linspace(sigma_min, sigma_max, DENSITY)
        T_range = np.linspace(max(0.05, 0.05 if T <= 0.05 else 0.05), max(0.05, T), DENSITY)  # éviter T≈0

        # Calcul surfaces (cache pour inputs identiques)
        @st.cache_data(show_spinner=False)
        def compute_surface(model_name, metric_name, option_type, S0,K0,r0,q0, sigma_vals, T_vals, n_steps):
            # metric_name in {"Prix","Delta","Vega","Theta"}
            def local_func(Ss,Ks,Ts,rr,vv,opt,qq):
                if model_name == "Black-Scholes":
                    return bs_european(Ss,Ks,Ts,rr,vv,opt,qq)
                else:
                    return binomial_european(Ss,Ks,Ts,rr,vv,n_steps,opt,qq)

            Z = np.zeros((len(T_vals), len(sigma_vals)), dtype=np.float32)
            for i, t_ in enumerate(T_vals):
                for j, sig_ in enumerate(sigma_vals):
                    if model_name == "Black-Scholes":
                        if metric_name == "Prix":
                            Z[i, j] = bs_european(S0,K0,t_,r0,sig_,option_type,q0)
                        elif metric_name == "Delta":
                            Z[i, j] = bs_delta(S0,K0,t_,r0,sig_,option_type,q0)
                        elif metric_name == "Vega":
                            Z[i, j] = bs_vega(S0,K0,t_,r0,sig_,q0)
                        elif metric_name == "Theta":
                            Z[i, j] = bs_theta(S0,K0,t_,r0,sig_,option_type,q0)
                    else:
                        # binomial -> différences finies pour Greeks
                        pf = lambda Ss,Ts,rr,vv: local_func(Ss,K0,max(1e-6,Ts),rr,max(1e-6,vv),option_type,q0)
                        if metric_name == "Prix":
                            Z[i, j] = pf(S0,t_,r0,sig_)
                        elif metric_name == "Delta":
                            Z[i, j] = greek_fd(pf,S0,K0,t_,r0,sig_,'delta')
                        elif metric_name == "Vega":
                            Z[i, j] = greek_fd(pf,S0,K0,t_,r0,sig_,'vega')
                        elif metric_name == "Theta":
                            Z[i, j] = greek_fd(pf,S0,K0,t_,r0,sig_,'theta')
            return Z

        metrics = ["Prix","Delta","Vega","Theta"]
        option_types = ["call","put"]

        # Grille 2x4: Call (row=1), Put (row=2), colonnes = metrics
        specs = [[{'type': 'surface'}]*4, [{'type':'surface'}]*4]
        fig_grid = make_subplots(rows=2, cols=4, specs=specs,
                                 subplot_titles=[f"{m} — Call" for m in metrics] + [f"{m} — Put" for m in metrics])

     for row, opt in enumerate(option_types, start=1):
    for col, metric in enumerate(metrics, start=1):
        Z = compute_surface(model, metric, opt, S, K, r, q, sigma_range, T_range, n if model=="Binomial" else 0)
        colorscale = "Viridis" if opt == "call" else "Plasma"
        fig_grid.add_trace(
            go.Surface(z=Z, x=sigma_range, y=T_range, colorscale=colorscale, showscale=False),
            row=row, col=col
        )

        # axes/titres/caméra
        fig_grid.update_layout(
            title="Surfaces 3D (σ en X, T en Y)",
            margin=dict(l=0, r=0, b=0, t=50),
        )
        # Configure chaque scène
        # Plotly nomme scene, scene2, ...
        for idx in range(1, 9):
            scene_key = f"scene{'' if idx==1 else idx}"
            fig_grid.update_layout(**{
                scene_key: dict(
                    xaxis_title="Volatilité σ",
                    yaxis_title="Temps T (années)",
                    zaxis_title=metrics[(idx-1)%4],
                    camera=dict(eye=dict(x=1.4, y=1.2, z=0.9))
                )
            })

        st.plotly_chart(fig_grid, use_container_width=True)

# ---------------- Fin du script ----------------
