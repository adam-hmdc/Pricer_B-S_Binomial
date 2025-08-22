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
