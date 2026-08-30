import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import torch
import numpy as np
import sys
import os

# Add pinn dir to path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pinn')))
from model import MooringPINN

app = dash.Dash(__name__, title="FOWT Mooring Digital Twin")

# 1. Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(BASE_DIR, "..", "data", "training", "batch_01.h5")
try:
    df = pd.read_hdf(H5_PATH, 'simulations')
    cases = df['Meta_CaseID'].unique()
except Exception as e:
    df = pd.DataFrame()
    cases = []
    print(f"Error loading data: {e}")

# 2. Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MooringPINN(in_features=18, hidden_dim=256, out_features=3, fourier_features=64).to(device)

MODEL_PATH = os.path.join(BASE_DIR, "..", "pinn", "checkpoints", "pinn_adam_final.pt")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Loaded Trained PINN Model.")
else:
    print("Warning: Model checkpoint not found. Using untrained weights.")

# Standardize inputs using the SAME logic as train.py
if not df.empty:
    X_all = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
                'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
    X_mean, X_std = X_all.mean(axis=0), X_all.std(axis=0)

app.layout = html.Div([
    html.H1("FOWT Mooring Digital Twin - PINN vs OpenFAST", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    
    html.Div([
        html.Label("Select Sea State (Validation Case):"),
        dcc.Dropdown(
            id='case-dropdown',
            options=[{'label': f"Case {c}", 'value': c} for c in cases],
            value=cases[0] if len(cases) > 0 else None,
            clearable=False
        )
    ], style={'width': '30%', 'margin': 'auto'}),
    
    html.Div(id='metrics-output', style={'textAlign': 'center', 'marginTop': '20px', 'fontFamily': 'monospace'}),
    
    dcc.Graph(id='tension-plot', style={'height': '70vh'})
])

@app.callback(
    [Output('tension-plot', 'figure'),
     Output('metrics-output', 'children')],
    [Input('case-dropdown', 'value')]
)
def update_plot(selected_case):
    if df.empty or selected_case is None:
        return go.Figure(), "No data available."
        
    case_df = df[df['Meta_CaseID'] == selected_case].copy()
    time_arr = case_df['Time_[s]'].values
    
    # Ground Truth
    Y_true = case_df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0 # kN
    
    # PINN Prediction
    X_case = case_df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
                      'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
                      
    dt = 0.0125
    V_case = np.gradient(X_case, dt, axis=0)
    A_case = np.gradient(V_case, dt, axis=0)
    
    X_full_case = np.concatenate([X_case, V_case, A_case], axis=1)
    
    # Load mean/std from training
    mean_path = os.path.join(BASE_DIR, "..", "pinn", "checkpoints", "x_mean.npy")
    std_path = os.path.join(BASE_DIR, "..", "pinn", "checkpoints", "x_std.npy")
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        X_mean = np.load(mean_path)
        X_std = np.load(std_path)
    else:
        # Fallback to local
        X_mean, X_std = X_full_case.mean(axis=0), X_full_case.std(axis=0)
    
    # Normalize
    X_norm = (X_full_case - X_mean) / (X_std + 1e-8)
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
        Y_pred = model(X_tensor).cpu().numpy()
        
    fig = go.Figure()
    colors = ['#EF553B', '#AB63FA', '#FFA15A'] # Plotly colors
    
    mses = []
    
    for i in range(3):
        # Ground Truth (Solid line)
        fig.add_trace(go.Scatter(
            x=time_arr, y=Y_true[:, i], 
            mode='lines', name=f'Line {i+1} (OpenFAST)',
            line=dict(color=colors[i], width=2)
        ))
        
        # PINN Pred (Dashed line)
        fig.add_trace(go.Scatter(
            x=time_arr, y=Y_pred[:, i], 
            mode='lines', name=f'Line {i+1} (PINN)',
            line=dict(color=colors[i], width=2, dash='dash')
        ))
        
        mse = np.mean((Y_true[:, i] - Y_pred[:, i])**2)
        mses.append(mse)
        
    fig.update_layout(
        title=f"Mooring Tensions for {selected_case}",
        xaxis_title="Time (s)",
        yaxis_title="Fairlead Tension (kN)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    metrics_text = f"Mean Squared Error (kN^2) - Line 1: {mses[0]:.1f} | Line 2: {mses[1]:.1f} | Line 3: {mses[2]:.1f}"
    
    return fig, metrics_text

if __name__ == '__main__':
    # Run the Dash server locally
    app.run(debug=True, port=8050)
