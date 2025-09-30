import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Fonctions de calcul
# -----------------------------
def bs_european(S, K, T, r, sigma, option_type='call', q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

def binomial_european(S, K, T, r, sigma, n=100, option_type='call', q=0):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # Prix en maturité
    ST = np.array([S * u**j * d**(n-j) for j in range(n+1)])
    if option_type == 'call':
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)
    # Backward induction
    for _ in range(n):
        option_values = np.exp(-r*dt) * (p*option_values[:-1] + (1-p)*option_values[1:])
    return option_values[0]

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
    sigma = st.number_input("Volatilité du sous-jacent σ", value=0.01, step=0.01)
    q = st.number_input("Taux de dividende q", value=0.0, step=0.01)
    if model == "Binomial":
        n = st.number_input("Nombre de pas (n)", min_value=1, value=100, step=1)
    
    st.markdown("---")  # ligne de séparation

    st.subheader("⚙️ Paramètres de sensibilité ")
    sigma_min = st.number_input("Volatilité min", value=0.01, step=0.01)
    sigma_max = st.number_input("Volatilité max", value=1.0, step=0.01)
    S_min = st.number_input("Prix spot min", value=50.0)
    S_max = st.number_input("Prix spot max", value=150.0)
   
   
    

    submitted = st.button("Calculer")

with col2:
    if submitted:
        # Choisir la fonction en fonction du modèle
        func = bs_european if model == "Black-Scholes" else lambda S,K,T,r,sigma,option_type,q: binomial_european(S,K,T,r,sigma,n,option_type,q)

        # Calcul Call et Put pour les paramètres actuels
        sigma_avg = (sigma_min + sigma_max)/2
        call_price = func(S, K, T, r, sigma, 'call', q)
        put_price = func(S, K, T, r, sigma, 'put', q)

        st.markdown(f"### 📈 Prix **Call** : <span style='color:green;font-size:20px'>{call_price:.4f}</span>", unsafe_allow_html=True)
        st.markdown(f"### 📉 Prix **Put** : <span style='color:red;font-size:20px'>{put_price:.4f}</span>", unsafe_allow_html=True)

        # ---------------- Heatmap 10x10 ----------------
        S_values = np.linspace(S_min, S_max, 10)
        sigma_values = np.linspace(sigma_min, sigma_max, 10)
        Z_call = np.array([[func(s, K, T, r, sig, 'call', q) for s in S_values] for sig in sigma_values])
        Z_put = np.array([[func(s, K, T, r, sig, 'put', q)  for s in S_values] for sig in sigma_values])

        # Heatmap Call
        fig_call_heatmap = go.Figure(data=go.Heatmap(
            z=Z_call, x=S_values, y=sigma_values,
            colorscale='Viridis',
            text=np.round(Z_call, 2),
            texttemplate="%{text}", textfont={"size":10, "color":"white"}
        ))
        fig_call_heatmap.update_layout(title="Matrice de Sensibilité du Call", xaxis_title="Prix Spot", yaxis_title="Volatilité σ")

        # Heatmap Put
        fig_put_heatmap = go.Figure(data=go.Heatmap(
            z=Z_put, x=S_values, y=sigma_values,
            colorscale='Plasma',
            text=np.round(Z_put, 2),
            texttemplate="%{text}", textfont={"size":10, "color":"white"}
        ))
        fig_put_heatmap.update_layout(title="Matrice de Senisbilité du Put", xaxis_title="Prix Spot", yaxis_title="Volatilité σ")

        st.plotly_chart(fig_call_heatmap, use_container_width=True)
        st.plotly_chart(fig_put_heatmap, use_container_width=True)

        # ---------------- Graphiques 3D : Prix en fonction de Volatilité et Temps ----------------
        sigma_range = np.linspace(sigma_min, sigma_max, 25)
        T_range = np.linspace(0.05, T, 25)  # éviter T=0

        # Call : surface prix(σ, T)
        Z_call_sigma_T = np.array([[func(S, K, t, r, sig, 'call', q) for sig in sigma_range] for t in T_range])
        # Put : surface prix(σ, T)
        Z_put_sigma_T  = np.array([[func(S, K, t, r, sig, 'put',  q) for sig in sigma_range] for t in T_range])

        # Subplots côte à côte
        fig_surfaces = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=("Surface 3D - Call ", "Surface 3D - Put ")
        )

        # Call
        fig_surfaces.add_trace(
            go.Surface(z=Z_call_sigma_T, x=sigma_range, y=T_range, colorscale="Viridis"),
            row=1, col=1
        )
        # Put
        fig_surfaces.add_trace(
            go.Surface(z=Z_put_sigma_T, x=sigma_range, y=T_range, colorscale="Plasma"),
            row=1, col=2
        )

        fig_surfaces.update_layout(
            title="Surfaces 3D interactives - Prix en fonction de Volatilité et Temps",
            scene=dict(xaxis_title="Volatilité σ", yaxis_title="Temps restant T", zaxis_title="Prix"),
            scene2=dict(xaxis_title="Volatilité σ", yaxis_title="Temps restant T", zaxis_title="Prix")
        )

        st.plotly_chart(fig_surfaces, use_container_width=True)
        # ---------------- Graphiques 3D améliorés ----------------
st.markdown("---")
st.subheader("🧭 Surfaces 3D avancées (Prix & Greeks)")

# Choix métrique + résolution
metric = st.selectbox("Métrique à afficher", ["Prix", "Delta", "Vega", "Theta"])
density = st.slider("Résolution de la grille", 10, 60, 25, step=5)

# Recalcule des gammes selon la résolution choisie
sigma_range = np.linspace(sigma_min, sigma_max, density)
T_range = np.linspace(max(0.02, 0.02 if T <= 0 else 0.02), max(0.05, T), density)  # éviter T≈0

# Helpers Black-Scholes (fermés)
def _d1_d2(S,K,T,r,sigma,q):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return d1, d2

def bs_delta(S,K,T,r,sigma,option_type='call',q=0):
    d1, _ = _d1_d2(S,K,T,r,sigma,q)
    if option_type == 'call':
        return np.exp(-q*T) * norm.cdf(d1)
    else:
        return np.exp(-q*T) * (norm.cdf(d1) - 1)

def bs_vega(S,K,T,r,sigma,q=0):
    d1, _ = _d1_d2(S,K,T,r,sigma,q)
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)  # par point de sigma (1.00 = 100%)

def bs_theta(S,K,T,r,sigma,option_type='call',q=0):
    d1, d2 = _d1_d2(S,K,T,r,sigma,q)
    term1 = - (S*np.exp(-q*T)*norm.pdf(d1)*sigma) / (2*np.sqrt(T))
    if option_type == 'call':
        term2 = - r*K*np.exp(-r*T)*norm.cdf(d2)
        term3 = + q*S*np.exp(-q*T)*norm.cdf(d1)
        return term1 + term2 + term3  # par an
    else:
        term2 = + r*K*np.exp(-r*T)*norm.cdf(-d2)
        term3 = - q*S*np.exp(-q*T)*norm.cdf(-d1)
        return term1 + term2 + term3

# Différences finies génériques (pour le modèle binomial)
def finite_diff(metric_name, S,K,t,r,sig,opt,q, hS=None, hT=None, hV=None):
    # pas adaptatifs
    hS = hS or max(1e-4*S, 1e-3)
    hT = hT or min(0.01, max(1e-3, 0.1*t))
    hV = hV or 1e-3

    price = lambda SS, TT, VV: func(SS, K, max(1e-6, TT), r, max(1e-6, VV), opt, q)

    if metric_name == "Prix":
        return price(S, t, sig)
    if metric_name == "Delta":
        return (price(S+hS, t, sig) - price(S-hS, t, sig)) / (2*hS)
    if metric_name == "Vega":
        return (price(S, t, sig+hV) - price(S, t, sig-hV)) / (2*hV)
    if metric_name == "Theta":
        t1 = max(1e-6, t+hT)
        t2 = max(1e-6, t-hT)
        return (price(S, t2, sig) - price(S, t1, sig)) / (2*hT)  # convention theta<0
    return np.nan

# Wrapper pour remplir une surface selon le modèle + métrique
@st.cache_data(show_spinner=False)
def compute_surface(model_name, metric_name, option_type, S, K, r, q, sigma_vals, T_vals):
    Z = np.zeros((len(T_vals), len(sigma_vals)), dtype=np.float32)
    for i, t in enumerate(T_vals):
        for j, sig in enumerate(sigma_vals):
            if model_name == "Black-Scholes":
                if metric_name == "Prix":
                    Z[i, j] = bs_european(S,K,t,r,sig,option_type,q)
                elif metric_name == "Delta":
                    Z[i, j] = bs_delta(S,K,t,r,sig,option_type,q)
                elif metric_name == "Vega":
                    Z[i, j] = bs_vega(S,K,t,r,sig,q)
                elif metric_name == "Theta":
                    Z[i, j] = bs_theta(S,K,t,r,sig,option_type,q)
            else:
                Z[i, j] = finite_diff(metric_name, S,K,t,r,sig,option_type,q)
    return Z

opt_type = st.radio("Type d’option", ["call","put"], horizontal=True)

with st.spinner("Calcul de la surface…"):
    Z = compute_surface(model, metric, opt_type, S, K, r, q, sigma_range, T_range)

fig_adv = go.Figure(data=go.Surface(
    z=Z, x=sigma_range, y=T_range,
    contours = {
        "z": {"show": True, "usecolormap": True, "project_z": True}
    },
    colorbar={"title": metric}
))
fig_adv.update_layout(
    title=f"Surface 3D — {metric} ({opt_type.capitalize()}) — Modèle: {model}",
    scene=dict(
        xaxis_title="Volatilité σ",
        yaxis_title="Temps restant T (années)",
        zaxis_title=metric,
        camera=dict(eye=dict(x=1.6, y=1.2, z=0.8))
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
hover_tpl = "<b>σ</b>=%{x:.3f}<br><b>T</b>=%{y:.3f}<br><b>"+metric+"</b>=%{z:.4f}"
fig_adv.update_traces(hovertemplate=hover_tpl)

st.plotly_chart(fig_adv, use_container_width=True)



        # ---------------- Graphiques 3D améliorés ----------------
