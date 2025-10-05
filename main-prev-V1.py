import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ardl import ARDL
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from io import BytesIO
import re
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION DE LA PAGE (DOIT ÊTRE EN PREMIER) ===
st.set_page_config(
    page_title="Prévisions",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# === CSS AMÉLIORÉ POUR MASQUER LES ÉLÉMENTS PROBLÉMATIQUES ===
st.markdown("""
<style>
/* PRIORITÉ ABSOLUE : Masquer tous les éléments de contrôle de sidebar */
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

button[kind="header"] {
    display: none !important;
}

.css-1dp5vir {
    display: none !important;
}


/* Personnalisation de la barre d'outils Plotly */
.modebar {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border-radius: 8px;
    padding: 4px;
    backdrop-filter: blur(10px);
}

.modebar-btn {
    transition: all 0.2s ease;
}

.modebar-btn:hover {
    background-color: rgba(44, 44, 44, 0.1) !important;
}

/* Charger Garamond web */
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&display=swap');

/* police UI */
:root {
  --ui-font: "EB Garamond", "Garamond", "Times New Roman", serif;
}

/* Mettre Garamond sur (quasi) tout le texte, y compris titres */
.stApp * {
  font-family: var(--ui-font) !important;
}

/* EXCEPTIONS : laisser la police Material aux icônes
      (sinon Streamlit affiche le nom de l’icône en clair) */
.material-icons,
.material-symbols-outlined,
[data-testid="collapsedControl"] span {
  font-family: 'Material Symbols Outlined','Material Icons';
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}

/* Thème général */
.stApp {
    background-color: #FFFFFF;
}

/* Sidebar responsive */
[data-testid="stSidebar"] {
    background-color: rgba(248, 248, 248, 0.95) !important;
    border-right: 1px solid rgba(229, 229, 229, 0.5);
    backdrop-filter: blur(10px);
    min-width: 250px !important;
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        min-width: 200px !important;
    }
}

/* Titres */
h1, h2, h3 {
    color: #2C2C2C;
    font-weight: 600;
    letter-spacing: -0.3px;
}

@media (max-width: 768px) {
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.3rem !important; }
}

/* Boutons iOS style */
.stButton>button {
    background-color: transparent !important;
    color: #2C2C2C !important;
    border: 1.5px solid rgba(44, 44, 44, 0.3) !important;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 500;
    font-size: 15px;
    transition: all 0.3s ease;
    backdrop-filter: blur(5px);
    width: 100%;
}

@media (max-width: 768px) {
    .stButton>button {
        padding: 10px 16px;
        font-size: 14px;
    }
}

.stButton>button:hover {
    background-color: rgba(44, 44, 44, 0.1) !important;
    border-color: rgba(44, 44, 44, 0.7) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.stButton>button[kind="primary"] {
    background-color: rgba(44, 44, 44, 0.9) !important;
    color: #FFFFFF !important;
    border: 1.5px solid #2C2C2C !important;
}

.stButton>button[kind="primary"]:hover {
    background-color: rgba(44, 44, 44, 0.7) !important;
}

/* Inputs responsive */
.stTextInput>div>div>input,
.stSelectbox>div>div>select,
.stNumberInput>div>div>input {
    border: 1px solid rgba(229, 229, 229, 0.7);
    border-radius: 10px;
    padding: 10px;
    background-color: rgba(248, 248, 248, 0.8);
    transition: all 0.3s ease;
    font-size: 15px;
}

@media (max-width: 768px) {
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        font-size: 14px;
        padding: 8px;
    }
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(248, 248, 248, 0.8);
    border-radius: 12px;
    padding: 4px;
    backdrop-filter: blur(5px);
    flex-wrap: wrap;
}

@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 13px;
        padding: 8px 12px;
    }
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8E8E93;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    color: #2C2C2C !important;
}

/* Metrics responsive */
[data-testid="stMetricValue"] {
    color: #2C2C2C;
    font-size: 28px;
    font-weight: 600;
}

@media (max-width: 768px) {
    [data-testid="stMetricValue"] {
        font-size: 22px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 12px;
    }
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(229, 229, 229, 0.7);
    border-radius: 12px;
    padding: 20px;
    background-color: rgba(250, 250, 250, 0.8);
    backdrop-filter: blur(5px);
}

@media (max-width: 768px) {
    [data-testid="stFileUploader"] {
        padding: 15px;
    }
}

/* DataFrames responsive */
.stDataFrame {
    border-radius: 12px;
    overflow-x: auto;
    border: 1px solid rgba(229, 229, 229, 0.5);
}

@media (max-width: 768px) {
    .stDataFrame {
        font-size: 12px;
    }
}

/* Metric containers */
[data-testid="metric-container"] {
    border: 1px solid rgba(229, 229, 229, 0.5);
    border-radius: 12px;
    padding: 1rem;
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(5px);
}

@media (max-width: 768px) {
    [data-testid="metric-container"] {
        padding: 0.75rem;
    }
}

/* Colonnes responsive */
.row-widget.stHorizontal {
    flex-wrap: wrap;
}

@media (max-width: 768px) {
    .row-widget.stHorizontal > div {
        min-width: 100% !important;
        margin-bottom: 10px;
    }
}

/* Plotly charts responsive - Afficher la barre d'outils */
.js-plotly-plot {
    width: 100% !important;
}

.js-plotly-plot .plotly .modebar {
    display: flex !important;
}

@media (max-width: 768px) {
    .js-plotly-plot .plotly {
        font-size: 11px !important;
    }
    
    .modebar {
        right: 10px !important;
        top: 10px !important;
    }
    
    .modebar-btn {
        width: 32px !important;
        height: 32px !important;
    }
}

/* Slider responsive */
.stSlider {
    padding: 0 10px;
}

@media (max-width: 768px) {
    .stSlider {
        padding: 0 5px;
    }
}

/* Espacement du contenu principal */
.main .block-container {
    padding-top: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}

@media (max-width: 768px) {
    .main .block-container {
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
}

/* Navigation sidebar */
[data-testid="stSidebar"] .stButton>button {
    margin: 5px 0;
}

/* Alerts responsive */
.stAlert {
    border-radius: 12px;
    border: none;
    backdrop-filter: blur(5px);
}

@media (max-width: 768px) {
    .stAlert {
        font-size: 13px;
        padding: 10px;
    }
}

/* Expander responsive */
.streamlit-expanderHeader {
    font-size: 16px;
}

@media (max-width: 768px) {
    .streamlit-expanderHeader {
        font-size: 14px;
    }
}
</style>
""", unsafe_allow_html=True)

# === FORMATAGE DES NOMBRES ===
def format_number(value):
    """Formate les nombres en milliers, millions, milliards avec le format français"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    abs_value = abs(value)
    
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f} Md MGA".replace(',', ' ').replace('.', ',')
    elif abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.1f} M MGA".replace(',', ' ').replace('.', ',')
    elif abs_value >= 1_000:
        return f"{value / 1_000:,.1f} k MGA".replace(',', ' ').replace('.', ',')
    else:
        return f"{value:,.1f} MGA".replace(',', ' ').replace('.', ',')

# === ANALYSE DES SÉRIES TEMPORELLES ===
def analyze_time_series(series):
    """Analyse une série temporelle pour déterminer ses caractéristiques"""
    analysis = {
        'tendance': 'Non détectée',
        'saisonnalite': 'Non détectée',
        'stationnarite': 'Non déterminée',
        'recommandations': []
    }
    
    try:
        if len(series) > 12:
            mean_first = series[:6].mean()
            mean_last = series[-6:].mean()
            variation = abs(mean_last - mean_first) / (abs(mean_first) + 1e-10)
            
            if variation > 0.1:
                analysis['tendance'] = 'Détectée'
                analysis['recommandations'].append('Présence de tendance - Modèles avec différenciation recommandés')
            else:
                analysis['tendance'] = 'Faible'
        
        if len(series) >= 24:
            try:
                decomposition = seasonal_decompose(series, period=12, model='additive', extrapolate_trend='freq')
                seasonal_strength = np.std(decomposition.seasonal) / (np.std(decomposition.resid) + 1e-10)
                
                if seasonal_strength > 0.5:
                    analysis['saisonnalite'] = 'Forte'
                    analysis['recommandations'].append('Saisonnalité détectée - Modèles saisonniers recommandés')
                elif seasonal_strength > 0.2:
                    analysis['saisonnalite'] = 'Modérée'
                    analysis['recommandations'].append('Saisonnalité modérée - Modèles avec composante saisonnière recommandés')
            except:
                pass
        
        if analysis['tendance'] == 'Détectée' and analysis['saisonnalite'] in ['Forte', 'Modérée']:
            analysis['recommandations'].append('ARIMA saisonnier, Prophet, Exponential Smoothing recommandés')
        elif analysis['tendance'] == 'Détectée':
            analysis['recommandations'].append('ARIMA, Regression Linéaire, Random Forest recommandés')
        elif analysis['saisonnalite'] in ['Forte', 'Modérée']:
            analysis['recommandations'].append('SARIMA, Prophet, Seasonal Naive recommandés')
        else:
            analysis['recommandations'].append('AR, VAR, Random Forest recommandés')
            
    except Exception as e:
        analysis['erreur'] = f"Erreur d'analyse: {str(e)}"
    
    return analysis

# === FORECASTING FUNCTIONS ===
def forecast_ssae(series, periods):
    forecasts = []
    current_series = series.copy()
    for _ in range(periods):
        mean = current_series.mean()
        forecasts.append(mean)
        current_series = pd.concat([current_series[1:], pd.Series([mean])])
    return np.array(forecasts)

def forecast_ar(p, series, periods):
    try:
        model = ARIMA(series, order=(p, 0, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        st.error(f"Erreur AR({p}): {str(e)}")
        return np.zeros(periods)

def forecast_arima(order, series, periods):
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        st.error(f"Erreur ARIMA{order}: {str(e)}")
        return np.zeros(periods)

def forecast_var(lag_order, series_dict, target_var, periods):
    try:
        data_var = pd.DataFrame(series_dict)
        if len(data_var.columns) < 2:
            st.warning("VAR nécessite au moins 2 variables. Retour à un modèle univarié.")
            return forecast_ar(lag_order, series_dict[target_var], periods)
        model = VAR(data_var)
        model_fitted = model.fit(lag_order)
        forecast = model_fitted.forecast(data_var.values[-lag_order:], steps=periods)
        return forecast[:, data_var.columns.get_loc(target_var)]
    except Exception as e:
        st.error(f"Erreur VAR: {str(e)}")
        return np.zeros(periods)

def forecast_ardl(lags, series, exog=None, periods=1):
    try:
        if exog is not None:
            model = ARDL(series, lags=lags, exog=exog, order=0)
        else:
            model = ARDL(series, lags=lags, order=0)
        
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        st.error(f"Erreur ARDL: {str(e)}")
        return np.zeros(periods)

def forecast_prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, periods=12, df=None, col=None):
    try:
        prophet_df = df[["Date", col]].rename(columns={"Date": "ds", col: "y"})
        prophet_df = prophet_df.dropna()
        m = Prophet(
            changepoint_prior_scale=changepoint_prior_scale, 
            seasonality_prior_scale=seasonality_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=periods, freq='M')
        forecast = m.predict(future)
        return forecast["yhat"].tail(periods).values
    except Exception as e:
        st.error(f"Erreur Prophet: {str(e)}")
        return np.zeros(periods)

def forecast_linear_regression(series, periods):
    try:
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        return forecast
    except Exception as e:
        st.error(f"Erreur Régression Linéaire: {str(e)}")
        return np.zeros(periods)

def forecast_random_forest(series, periods, n_estimators=100, max_depth=10):
    try:
        lags = min(12, len(series) // 2)
        if lags < 1 or len(series) < lags + 10:
            st.error(f"Données insuffisantes pour Random Forest (besoin de {lags + 10} points minimum)")
            return np.zeros(periods)
        X, y = [], []
        for i in range(lags, len(series)):
            X.append(series.iloc[i-lags:i].values)
            y.append(series.iloc[i])
        X = np.array(X)
        y = np.array(y)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        forecasts = []
        last_lags = series.iloc[-lags:].values
        for _ in range(periods):
            pred = model.predict(last_lags.reshape(1, -1))[0]
            forecasts.append(pred)
            last_lags = np.roll(last_lags, -1)
            last_lags[-1] = pred
        return np.array(forecasts)
    except Exception as e:
        st.error(f"Erreur Random Forest: {str(e)}")
        return np.zeros(periods)

def forecast_mlp(series, periods, hidden_layer_sizes=(100,), max_iter=200):
    try:
        lags = min(12, len(series) // 2)
        if lags < 1 or len(series) < lags + 10:
            st.error(f"Données insuffisantes pour MLP (besoin de {lags + 10} points minimum)")
            return np.zeros(periods)
        X, y = [], []
        for i in range(lags, len(series)):
            X.append(series.iloc[i-lags:i].values)
            y.append(series.iloc[i])
        X = np.array(X)
        y = np.array(y)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
        model.fit(X, y)
        forecasts = []
        last_lags = series.iloc[-lags:].values
        for _ in range(periods):
            pred = model.predict(last_lags.reshape(1, -1))[0]
            forecasts.append(pred)
            last_lags = np.roll(last_lags, -1)
            last_lags[-1] = pred
        return np.array(forecasts)
    except Exception as e:
        st.error(f"Erreur MLP: {str(e)}")
        return np.zeros(periods)

def forecast_exponential_smoothing(series, periods, trend='add', seasonal='add', seasonal_periods=12):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        if len(series) < seasonal_periods * 2:
            st.error(f"Données insuffisantes pour Exponential Smoothing (besoin de {seasonal_periods * 2} points minimum)")
            return np.zeros(periods)
        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    except Exception as e:
        st.error(f"Erreur Exponential Smoothing: {str(e)}")
        return np.zeros(periods)

def forecast_variable(df, col, periods, model_type, params):
    if "Date" not in df.columns:
        st.error("La colonne 'Date' est manquante dans les données")
        return np.zeros(periods)
    
    series = df.set_index("Date")[col].dropna()
    if len(series) < 2:
        st.error(f"Données insuffisantes pour {col} (minimum 2 points)")
        return np.zeros(periods)
    
    if model_type == "NAIVE":
        return forecast_ssae(series, periods)
    elif model_type == "AR(p)":
        p = params.get('p', 1)
        if len(series) < p + 1:
            st.error(f"Données insuffisantes pour AR({p}) (besoin de {p + 1} points minimum)")
            return np.zeros(periods)
        return forecast_ar(p, series, periods)
    elif model_type == "ARIMA":
        order = params.get('order', (5, 1, 0))
        if len(series) < max(order) + 1:
            st.error(f"Données insuffisantes pour ARIMA{order} (besoin de {max(order) + 1} points minimum)")
            return np.zeros(periods)
        return forecast_arima(order, series, periods)
    elif model_type == "VAR":
        vars_list = [col]
        other_vars = [v for v in df.columns.drop("Date") if v != col][:1]
        vars_list.extend(other_vars)
        series_dict = {v: df.set_index("Date")[v].dropna() for v in vars_list}
        lag_order = params.get('lag_order', 1)
        if len(series) < lag_order + 1:
            st.error(f"Données insuffisantes pour VAR (besoin de {lag_order + 1} points minimum)")
            return np.zeros(periods)
        return forecast_var(lag_order, series_dict, col, periods)
    elif model_type == "ARDL":
        lags = params.get('lags', 1)
        if len(series) < lags + 1:
            st.error(f"Données insuffisantes pour ARDL (besoin de {lags + 1} points minimum)")
            return np.zeros(periods)
        return forecast_ardl(lags, series, periods=periods)
    elif model_type == "Prophet":
        changepoint = params.get('changepoint_prior_scale', 0.05)
        seasonality = params.get('seasonality_prior_scale', 10.0)
        return forecast_prophet(changepoint, seasonality, periods, df, col)
    elif model_type == "Régression Linéaire":
        return forecast_linear_regression(series, periods)
    elif model_type == "Random Forest":
        n_est = params.get('n_estimators', 100)
        max_d = params.get('max_depth', 10)
        return forecast_random_forest(series, periods, n_est, max_d)
    elif model_type == "MLP":
        hidden_layers = params.get('hidden_layer_sizes', (100,))
        max_iter = params.get('max_iter', 200)
        return forecast_mlp(series, periods, hidden_layers, max_iter)
    elif model_type == "Exponential Smoothing":
        trend = params.get('trend', 'add')
        seasonal = params.get('seasonal', 'add')
        sp = params.get('seasonal_periods', 12)
        return forecast_exponential_smoothing(series, periods, trend, seasonal, sp)
    else:
        st.error(f"Modèle {model_type} non supporté")
        return np.zeros(periods)

def generate_full_forecast_excel(df, periods, model_type, params):
    if "Date" not in df.columns:
        st.error("La colonne 'Date' est manquante dans les données")
        return pd.DataFrame()

    variables = df.columns.drop("Date")
    historical_dates = df["Date"].dt.strftime('%Y-%m')
    future_dates = pd.date_range(start=df["Date"].max() + pd.DateOffset(months=1), periods=periods, freq='M').strftime('%Y-%m')

    all_dates = list(historical_dates) + list(future_dates)
    forecast_df = pd.DataFrame(index=variables, columns=all_dates)

    for var in variables:
        historical = df.set_index("Date")[var]
        forecast = forecast_variable(df, var, periods, model_type, params)
        full_series = pd.concat([historical, pd.Series(forecast, index=pd.to_datetime(future_dates))])
        full_series.index = historical.index.strftime('%Y-%m').tolist() + list(future_dates)
        forecast_df.loc[var] = full_series.values

    forecast_df = forecast_df.T
    forecast_df.insert(0, "Variable", forecast_df.index)
    forecast_df = forecast_df.reset_index(drop=True)
    return forecast_df

# === DATA VISUALIZATION MODULE ===
def data_visualization_module():
    st.header("Tableaux de Bord & Prévisions")

    if "data_uploaded" not in st.session_state or not st.session_state.data_uploaded:
        st.warning("Veuillez d'abord importer des données dans le Module de Collecte des Données")
        return

    df = st.session_state.source_data.copy()
    if "Date" not in df.columns:
        st.error("La colonne 'Date' est manquante dans les données")
        return
    df["Date"] = pd.to_datetime(df["Date"])

    st.subheader("Indicateurs Clés")
    key_vars = df.columns.drop("Date")[:4]
    cols = st.columns(len(key_vars))
    
    for i, var in enumerate(key_vars):
        if var in df.columns:
            series = df.set_index("Date")[var].dropna()
            if len(series) < 2:
                latest = series.iloc[-1] if len(series) > 0 else 0
                base_value = latest
                delta_pct = 0.0
            else:
                latest = series.iloc[-1]
                base_value = series.iloc[0]
                delta_pct = ((latest - base_value) / base_value * 100) if base_value != 0 else 0.0
            
            formatted_value = format_number(latest)
            base_year = df["Date"].iloc[0].year if len(df) > 0 else "N/A"
            
            cols[i].metric(
                var, 
                formatted_value, 
                f"{delta_pct:+.1f}% (base {base_year})"
            )

    tab1, tab2, tab3, tab4 = st.tabs(["Évolution", "Visualisation", "Analyse", "Prévisions"])

    with tab1:
        selected_vars = st.multiselect("Variables à visualiser", df.columns.drop("Date"))
        if selected_vars:
            fig = px.line(
                df,
                x="Date",
                y=selected_vars,
                title="Évolution des Variables",
                labels={"value": "Valeur", "variable": "Variable"},
                height=500
            )
            fig.update_layout(
                font_family="Garamond",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'evolution_{"-".join(selected_vars)}',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            })
        else:
            st.info("Sélectionnez au moins une variable à visualiser")

    with tab2:
        st.subheader("Visualisation des Données")
        
        chart_type = st.selectbox(
            "Type de graphique",
            ["Barres Verticales", "Barres Horizontales", "Ligne", "Aire", "Histogramme", "Box Plot"]
        )
        
        selected_vars = st.multiselect(
            "Variables à visualiser", 
            df.columns.drop("Date"),
            default=[df.columns.drop("Date")[0]] if len(df.columns.drop("Date")) > 0 else None,
            key="viz_vars"
        )
        
        if selected_vars:
            if chart_type == "Barres Verticales":
                df_bar = df.copy()
                df_bar['Année'] = df_bar['Date'].dt.year
                df_annual = df_bar.groupby('Année')[selected_vars].mean().reset_index()
                
                fig = px.bar(
                    df_annual,
                    x="Année",
                    y=selected_vars,
                    title=f"Évolution Annuelle - {', '.join(selected_vars)}",
                    labels={"value": "Valeur", "variable": "Variable"},
                    barmode='group',
                    height=500
                )
                
            elif chart_type == "Barres Horizontales":
                last_values = df[selected_vars].iloc[-1].sort_values()
                fig = px.bar(
                    x=last_values.values,
                    y=last_values.index,
                    orientation='h',
                    title=f"Comparaison des Variables (Dernière Période)",
                    labels={"x": "Valeur", "y": "Variable"},
                    height=500
                )
                
            elif chart_type == "Ligne":
                fig = px.line(
                    df,
                    x="Date",
                    y=selected_vars,
                    title="Évolution des Variables",
                    labels={"value": "Valeur", "variable": "Variable"},
                    height=500
                )
                
            elif chart_type == "Aire":
                fig = px.area(
                    df,
                    x="Date",
                    y=selected_vars,
                    title="Évolution Cumulée des Variables",
                    labels={"value": "Valeur", "variable": "Variable"},
                    height=500
                )
                
            elif chart_type == "Histogramme":
                fig = px.histogram(
                    df,
                    x=selected_vars[0],
                    title=f"Distribution de {selected_vars[0]}",
                    labels={"value": "Valeur"},
                    height=500
                )
                if len(selected_vars) > 1:
                    st.warning("L'histogramme affiche seulement la première variable sélectionnée")
                    
            elif chart_type == "Box Plot":
                fig = px.box(
                    df,
                    y=selected_vars,
                    title=f"Distribution Statistique des Variables",
                    labels={"value": "Valeur", "variable": "Variable"},
                    height=500
                )
            
            fig.update_layout(
                font_family="Garamond",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'visualisation_{chart_type}',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            })
            
            with st.expander("📊 Statistiques Descriptives"):
                st.dataframe(
                    df[selected_vars].describe().round(2),
                    use_container_width=True
                )
                
        else:
            st.info("Sélectionnez au moins une variable à visualiser")

    with tab3:
        st.subheader("Analyse des Séries Temporelles")
        
        analysis_var = st.selectbox(
            "Variable à analyser",
            df.columns.drop("Date"),
            key="analysis_var"
        )
        
        if analysis_var:
            series = df.set_index("Date")[analysis_var].dropna()
            
            if len(series) < 12:
                st.warning("Données insuffisantes pour une analyse complète (minimum 12 points requis)")
            else:
                analysis = analyze_time_series(series)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tendance", analysis['tendance'])
                with col2:
                    st.metric("Saisonnalité", analysis['saisonnalite'])
                with col3:
                    st.metric("Points de données", len(series))
                
                st.subheader("🎯 Recommandations de Modèles")
                for recommendation in analysis['recommandations']:
                    st.write(f"• {recommendation}")
                
                st.subheader("📈 Décomposition de la Série")
                try:
                    decomposition = seasonal_decompose(series, period=12, model='additive', extrapolate_trend='freq')
                    
                    fig_decomp = go.Figure()
                    
                    fig_decomp.add_trace(go.Scatter(
                        x=series.index, y=decomposition.observed,
                        mode='lines', name='Série Originale',
                        line=dict(color='blue')
                    ))
                    
                    fig_decomp.add_trace(go.Scatter(
                        x=series.index, y=decomposition.trend,
                        mode='lines', name='Tendance',
                        line=dict(color='red')
                    ))
                    
                    fig_decomp.add_trace(go.Scatter(
                        x=series.index, y=decomposition.seasonal,
                        mode='lines', name='Saisonnalité',
                        line=dict(color='green')
                    ))
                    
                    fig_decomp.add_trace(go.Scatter(
                        x=series.index, y=decomposition.resid,
                        mode='lines', name='Résidu',
                        line=dict(color='orange')
                    ))
                    
                    fig_decomp.update_layout(
                        title=f"Décomposition de {analysis_var}",
                        height=600,
                        showlegend=True,
                        font_family="Garamond",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_decomp, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'decomposition_{analysis_var}',
                            'height': 1000,
                            'width': 1400,
                            'scale': 2
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Erreur lors de la décomposition: {str(e)}")

    with tab4:
        st.subheader("Modèle de Prévisions")
        col1, col2 = st.columns([1, 3])

        with col1:
            if "analysis_var" in st.session_state:
                analysis_var = st.session_state.analysis_var
                series = df.set_index("Date")[analysis_var].dropna()
                if len(series) >= 12:
                    analysis = analyze_time_series(series)
                    st.info("💡 Recommandations basées sur l'analyse:")
                    for rec in analysis['recommandations'][:2]:
                        st.write(f"• {rec}")

            model_type = st.selectbox("Modèle", [
                "NAIVE", "AR(p)", "ARIMA", "VAR", "ARDL", "Prophet", 
                "Régression Linéaire", "Random Forest", "MLP", 
                "Exponential Smoothing"
            ])
            indicator = st.selectbox("Indicateur à prévoir", df.columns.drop("Date"))
            periods = st.slider("Période de prévision (mois)", 3, 60, 12)

            params = {}
            if model_type == "AR(p)":
                params['p'] = st.slider("Lag p", 1, 12, 1)
            elif model_type == "ARIMA":
                p = st.slider("p (AR)", 0, 5, 1)
                d = st.slider("d (Diff)", 0, 2, 1)
                q = st.slider("q (MA)", 0, 5, 0)
                params['order'] = (p, d, q)
            elif model_type == "VAR":
                params['lag_order'] = st.slider("Lag order", 1, 4, 1)
                st.info("VAR utilise les 2 premières variables disponibles")
            elif model_type == "ARDL":
                params['lags'] = st.slider("Lags", 1, 12, 1)
            elif model_type == "Prophet":
                params['changepoint_prior_scale'] = st.slider("Échelle prior changepoint", 0.001, 0.5, 0.05)
                params['seasonality_prior_scale'] = st.slider("Échelle prior saisonnalité", 0.01, 100.0, 10.0)
            elif model_type == "Random Forest":
                params['n_estimators'] = st.slider("Nombre d'estimateurs", 10, 200, 100)
                params['max_depth'] = st.slider("Profondeur max", 3, 20, 10)
            elif model_type == "MLP":
                hidden_options = st.multiselect("Tailles des couches cachées", [50, 100, 200], [100])
                params['hidden_layer_sizes'] = tuple(hidden_options)
                params['max_iter'] = st.slider("Iterations max", 50, 1000, 200)
            elif model_type == "Exponential Smoothing":
                params['trend'] = st.selectbox("Tendance", ['add', 'mul', None])
                params['seasonal'] = st.selectbox("Saisonnalité", ['add', 'mul', None])
                params['seasonal_periods'] = st.slider("Périodes saisonnières", 4, 24, 12)

            if st.button("Lancer la prévision", type="primary"):
                with st.spinner("Prévision en cours..."):
                    series = df.set_index("Date")[indicator].dropna()
                    min_required = max(2, params.get('p', 1) + 1, 
                                     params.get('lag_order', 1) + 1, params.get('lags', 1) + 1)
                    if len(series) < min_required:
                        st.error(f"Données insuffisantes pour {model_type} (besoin de {min_required} points minimum)")
                    else:
                        forecast = forecast_variable(df, indicator, periods, model_type, params)
                        future_dates = pd.date_range(start=df["Date"].max() + pd.offsets.DateOffset(months=1), periods=periods, freq='M')
                        
                        historical_df = df[["Date", indicator]].copy()
                        historical_df["Type"] = "Historique"
                        
                        forecast_df = pd.DataFrame({
                            "Date": future_dates, 
                            indicator: forecast, 
                            "Type": "Prévision"
                        })
                        
                        full_df = pd.concat([historical_df, forecast_df], ignore_index=True)
                        
                        if len(series) >= 12:
                            train_size = len(series) - 12
                            train, test = series[:train_size], series[train_size:]
                            train_df = df[["Date", indicator]].iloc[:train_size].copy()
                            train_forecast = forecast_variable(train_df, indicator, 12, model_type, params)
                            mape = mean_absolute_percentage_error(test, train_forecast) if len(train_forecast) == len(test) else 0.1
                        else:
                            mape = 0.1
                            
                        st.session_state.forecast_data = full_df
                        st.session_state.mape = mape
                        st.session_state.forecast_variable = indicator
                        st.session_state.forecast_model = model_type
                        st.session_state.forecast_periods = periods
                        st.session_state.forecast_params = params
                        st.toast("Prévision terminée!")

            if st.button("Générer Excel avec toutes les prévisions"):
                with st.spinner("Génération des prévisions pour toutes les variables..."):
                    excel_df = generate_full_forecast_excel(df, periods, model_type, params)
                    if not excel_df.empty:
                        with BytesIO() as buffer:
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                excel_df.to_excel(writer, index=False, sheet_name="Prévisions")
                            st.download_button(
                                label="Télécharger Excel (Toutes Prévisions)",
                                data=buffer.getvalue(),
                                file_name=f"all_forecasts_{model_type}.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                        st.success("Fichier Excel généré avec succès!")

        with col2:
            if "forecast_data" in st.session_state:
                full_df = st.session_state.forecast_data
                indicator = st.session_state.forecast_variable
                
                historical_data = full_df[full_df["Type"] == "Historique"]
                forecast_data = full_df[full_df["Type"] == "Prévision"]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=historical_data["Date"],
                    y=historical_data[indicator],
                    mode='lines',
                    name='Historique',
                    line=dict(color='green', width=1.5)
                ))
                
                if len(historical_data) > 0 and len(forecast_data) > 0:
                    last_historical = historical_data.iloc[-1]
                    continuous_forecast = pd.concat([
                        pd.DataFrame([{
                            "Date": last_historical["Date"],
                            indicator: last_historical[indicator],
                            "Type": "Prévision"
                        }]),
                        forecast_data
                    ])
                    
                    fig.add_trace(go.Scatter(
                        x=continuous_forecast["Date"],
                        y=continuous_forecast[indicator],
                        mode='lines',
                        name='Prévision',
                        line=dict(color='brown', width=1.5)
                    ))
                
                fig.update_layout(
                    title=f"Prévision de {indicator} ({st.session_state.forecast_model})",
                    xaxis_title="Date",
                    yaxis_title="Valeur",
                    height=500,
                    showlegend=True,
                    font_family="Garamond",
                    hovermode='x unified'
                )
                
                if len(historical_data) > 0:
                    last_historical_date = historical_data["Date"].max()
                    
                    fig.add_shape(
                        type="line",
                        x0=last_historical_date,
                        x1=last_historical_date,
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(color="gray", width=2, dash="dot")
                    )
                    
                    fig.add_annotation(
                        x=last_historical_date,
                        y=1,
                        yref="paper",
                        text="Début prévision",
                        showarrow=False,
                        yshift=10,
                        xshift=10,
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1,
                        borderpad=4
                    )
                
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'prevision_{indicator}_{st.session_state.forecast_model}',
                        'height': 800,
                        'width': 1400,
                        'scale': 2
                    }
                })
                st.caption(f"Précision du modèle (MAPE): {st.session_state.mape:.2%}")
                
                if st.button("Exporter les prévisions (unique)"):
                    with BytesIO() as buffer:
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            full_df.to_excel(writer, index=False)
                        st.download_button(
                            label="Télécharger Prévisions (Excel)",
                            data=buffer.getvalue(),
                            file_name=f"forecast_{indicator}_{st.session_state.forecast_model}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
            else:
                st.info("Configurez et lancez une prévision")

# === DATA COLLECTION MODULE ===
def detect_data_orientation(df):
    first_row = df.iloc[0, 1:].astype(str)
    first_col = df.iloc[1:, 0].astype(str)
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(janv|fév|mars|avril|mai|juin|juil|août|sept|oct|nov|déc)\s*\d{4}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}',
        r'\d{4}[mM]\d{1,2}',
        r'\d{4}[-_]\d{2}',
    ]
    dates_in_row = sum(1 for cell in first_row if any(re.search(pattern, str(cell), re.IGNORECASE) for pattern in date_patterns))
    dates_in_col = sum(1 for cell in first_col if any(re.search(pattern, str(cell), re.IGNORECASE) for pattern in date_patterns))
    return "dates_in_columns" if dates_in_row > dates_in_col else "dates_in_rows"

def standardize_dataframe(df, orientation=None):
    if orientation is None:
        orientation = detect_data_orientation(df)

    if orientation == "dates_in_columns":
        first_col_name = df.columns[0]
        df = df.set_index(first_col_name).T.reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
    else:
        date_regex = r'^(date|year|annee|année|period|période)$'
        candidates = [c for c in df.columns if re.search(date_regex, str(c), flags=re.IGNORECASE)]
        if candidates:
            date_col = candidates[0]
            cols = list(df.columns)
            cols.insert(0, cols.pop(cols.index(date_col)))
            df = df[cols]
        if df.columns[0] != 'Date':
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    return df

def clean_and_convert_dates(df):
    if 'Date' not in df.columns:
        st.error("Colonne 'Date' non trouvée après standardisation")
        return df
    df['Date'] = df['Date'].astype(str).str.strip()
    month_replacements = {
        'janv': 'jan', 'fév': 'feb', 'mars': 'mar', 'avril': 'apr',
        'mai': 'may', 'juin': 'jun', 'juil': 'jul', 'août': 'aug',
        'sept': 'sep', 'oct': 'oct', 'nov': 'nov', 'déc': 'dec'
    }
    for fr_month, en_month in month_replacements.items():
        df['Date'] = df['Date'].str.replace(fr_month, en_month, case=False, regex=False)
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%b %Y', '%B %Y', '%Y%m', '%YM%m'
    ]
    converted_dates = []
    for date_str in df['Date']:
        converted = False
        for fmt in date_formats:
            try:
                converted_date = pd.to_datetime(date_str, format=fmt)
                converted_dates.append(converted_date)
                converted = True
                break
            except:
                continue
        if not converted:
            try:
                converted_date = pd.to_datetime(date_str, errors='coerce')
                converted_dates.append(converted_date)
            except:
                converted_dates.append(pd.NaT)
    df['Date'] = converted_dates
    return df

def validate_numeric_columns(df):
    numeric_columns = df.columns[1:]
    for col in numeric_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def display_data_summary(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", len(df))
    with col2:
        st.metric("Variables", len(df.columns) - 1)
    with col3:
        valid_dates = df['Date'].notna().sum()
        st.metric("Dates valides", valid_dates)
    if valid_dates > 0:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        st.info(f"Période: {min_date.strftime('%B %Y')} à {max_date.strftime('%B %Y')}")

def data_collection_module():
    st.header("Collecte des Données")

    uploaded_file = st.file_uploader(
        "Importer le fichier de données",
        type=["xlsx", "xls", "csv"],
        help="Formats supportés: Excel (.xlsx, .xls) ou CSV"
    )

    if uploaded_file:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("Fichier lu avec succès!")
            
            with st.expander("Aperçu des données brutes", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Options de traitement")
            col_process1, col_process2 = st.columns(2)

            with col_process1:
                orientation = st.radio(
                    "Orientation des données",
                    ["Auto-détection", "Variables en lignes, dates en colonnes", "Variables en colonnes, dates en lignes"],
                    help="Laissez en auto-détection pour un traitement automatique"
                )

            with col_process2:
                if orientation == "Variables en colonnes, dates en lignes":
                    max_skipc = max(0, min(10, df.shape[1] - 1))
                    skip_cols = st.number_input("Ignorer les premières colonnes", min_value=0, max_value=max_skipc, value=0)
                    if skip_cols > 0:
                        df = df.iloc[:, skip_cols:]
                else:
                    max_skipr = max(0, min(10, df.shape[0] - 1))
                    skip_rows = st.number_input("Ignorer les premières lignes", min_value=0, max_value=max_skipr, value=0)
                    if skip_rows > 0:
                        df = df.iloc[skip_rows:].reset_index(drop=True)

            orientation_code = None if orientation == "Auto-détection" else \
                             "dates_in_columns" if orientation == "Variables en lignes, dates en colonnes" else \
                             "dates_in_rows"

            df_processed = standardize_dataframe(df.copy(), orientation_code)
            df_processed = clean_and_convert_dates(df_processed)
            df_processed = validate_numeric_columns(df_processed)

            display_data_summary(df_processed)
            
            st.subheader("Aperçu des données traitées")
            st.dataframe(df_processed.head(10), use_container_width=True)
            
            col_val1, col_val2 = st.columns(2)
            
            with col_val1:
                if st.button("Valider et sauvegarder", type="primary", use_container_width=True):
                    st.session_state.source_data = df_processed
                    st.session_state.data_uploaded = True
                    st.session_state.upload_timestamp = datetime.datetime.now()
                    st.success("Données sauvegardées avec succès!")
                    st.balloons()
            
            with col_val2:
                if st.button("Retraiter", use_container_width=True):
                    st.rerun()

        except Exception as e:
            st.error(f"Erreur de traitement du fichier: {str(e)}")

# === LOGO ET CONFIGURATION ===
LOGO_DATA_URI = "https://img.icons8.com/?size=1024&id=Hrn58QQNnrR5&format=png&color=000000"
ICON_DATA_URI = "https://img.icons8.com/?size=1024&id=Hrn58QQNnrR5&format=png&color=000000"
st.logo(
    image=LOGO_DATA_URI,          
    link="https://ramanambonona.github.io/",
    icon_image=ICON_DATA_URI,
    size="large"
)

# === NAVIGATION ===
with st.sidebar:
    st.title("🌊 Navigation")
    st.divider()
    
    if "navigation_module" not in st.session_state:
        st.session_state.navigation_module = "Data"
    
    col_nav1, col_nav2 = st.columns([1, 1])
    
    with col_nav1:
        if st.button("📥 Data", 
                    key="nav_data", 
                    use_container_width=True,
                    type="primary" if st.session_state.navigation_module == "Data" else "secondary"):
            st.session_state.navigation_module = "Data"
            st.rerun()
    
    with col_nav2:
        if st.button("📈 Prév.", 
                    key="nav_forecast", 
                    use_container_width=True,
                    type="primary" if st.session_state.navigation_module == "Prévision" else "secondary"):
            st.session_state.navigation_module = "Prévision"
            st.rerun()
    
    st.divider()
    
    if "data_uploaded" in st.session_state and st.session_state.data_uploaded:
        st.success("✅ Données chargées")
        if "upload_timestamp" in st.session_state:
            st.caption(f"Dernière mise à jour: {st.session_state.upload_timestamp.strftime('%H:%M - %d/%m/%Y')}")

# === CONTENU PRINCIPAL ===
if st.session_state.navigation_module == "Data":
    data_collection_module()
elif st.session_state.navigation_module == "Prévision":
    data_visualization_module()




