import os
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import scipy
from scipy import stats
from fpdf import FPDF
import sklearn
import xgboost as xgb
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import requests  # For downloading models
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()  # This will load variables from .env into os.environ

connection_string = os.environ.get("NEON_DB_CONNECTION")
if not connection_string:
    raise ValueError("NEON_DB_CONNECTION is not set!")

# Ensure the "Graphs" folder exists
if not os.path.exists("Graphs"):
    os.makedirs("Graphs")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################
# GLOBALS & MODEL LOADING
##############################

# Load CSVs for college averages and extra pitch data (these remain unchanged)
filtered_pitchers = pd.read_csv('./DBs/filtered_pitchers.csv')  # College averages
pitches_with_stuff = pd.read_csv('./DBs/college_data_with_stuff.csv')  # Extra college pitch data

# Colors for pitchers
pitch_colors = {
    'Fastball': '#A30303', 'Sinker': '#F99A4D', 'Curveball': '#45A8FD',
    'Splitter': '#00889B', 'ChangeUp': '#7DAC00', 'Slider': '#E5BF00',
    'Sweeper': '#C29A00', 'Cutter': '#543000', 'Knuckleball': '#9B63E2'
}

metrics = [
    'RelSpeed', 'SpinRate', 'InducedVertBreak', 'Horizontal Break (in)',
    'Extension', 'Vertical Approach Angle (°)', 'RelHeight', 'Stuff+'
]

# Create a global dictionary to store precomputed values
COLLEGE_VALUES = {}

# Loop through each unique pitch type in the CSV and precompute the arrays
for pitch in filtered_pitchers['Pitch Type'].unique():
    COLLEGE_VALUES[pitch] = {}
    for metric in metrics:
        # We drop NA and take the absolute value as done in the original code.
        values = filtered_pitchers[filtered_pitchers['Pitch Type'] == pitch][metric].dropna().abs().values
        COLLEGE_VALUES[pitch][metric] = values

# Download models from remote storage
base_url = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/"
try:
    print("Downloading models from remote storage...")
    # Random Forest models
    rf_modelfb = pickle.load(BytesIO(requests.get(base_url + "rfc_modelfb-pHfyNtcUetdII0zTjEgPpbm8Mouf9e.sav").content))
    rf_modelsl = pickle.load(BytesIO(requests.get(base_url + "rfc_modelsl-1H7J4VrTChpEXd0iSnUcmRolixx9yu.sav").content))
    rf_modelcb = pickle.load(BytesIO(requests.get(base_url + "rfc_modelcb-qdhcUf9F7p8FRlGX14VQalFSMp2R9d.sav").content))
    rf_modelch = pickle.load(BytesIO(requests.get(base_url + "rfc_modelch-H1OdfQtAHsX0sjKsFOd1pxIELU1b0G.sav").content))
    rf_models = {
        "Fastball": rf_modelfb,
        "Sinker": rf_modelfb,
        "Curveball": rf_modelcb,
        "Slider": rf_modelsl,
        "Cutter": rf_modelsl,
        "ChangeUp": rf_modelch,
    }

    print("Downloading XGB models from remote storage...")
    xgb_modelfb = pickle.load(
        BytesIO(requests.get(base_url + "xgb_modelfb-rr3krkj30ylMQ9PxwU4sGIvsGOOKIN.sav").content))
    xgb_modelsl = pickle.load(
        BytesIO(requests.get(base_url + "xgb_modelsl-fgqqdxQ3P1DzZiqmxv48LUB1C53IFt.sav").content))
    xgb_modelcb = pickle.load(
        BytesIO(requests.get(base_url + "xgb_modelcb-DvGWFX5texrWl2ImDqKShKpSOb8YEh.sav").content))
    xgb_modelch = pickle.load(
        BytesIO(requests.get(base_url + "xgb_modelch-nQCeZmk7WtNKUsvlnFJ7cFQlrp9MSm.sav").content))
    xgb_models = {
        "Fastball": xgb_modelfb,
        "Sinker": xgb_modelfb,
        "Curveball": xgb_modelcb,
        "Slider": xgb_modelsl,
        "Cutter": xgb_modelsl,
        "ChangeUp": xgb_modelch,
    }
    print("Models loaded from remote storage.")
except Exception as e:
    print(f"Error loading models from remote storage: {e}")


##############################
# PLOTTING FUNCTIONS
##############################

def calculate_stuff_plus(row):
    pitch_type = row['Pitch Type']
    if pitch_type in rf_models:
        rf_model = rf_models[pitch_type]
        xgb_model = xgb_models[pitch_type]
        if pitch_type in ['Fastball', 'Sinker']:
            features = ['RelSpeed', 'SpinRate', 'differential_break', 'RelHeight', 'ABS_RelSide', 'Extension']
        else:
            features = ['RelSpeed', 'SpinRate', 'ABS_Horizontal', 'RelHeight', 'ABS_RelSide', 'Extension',
                        'InducedVertBreak']
        row_features = row[features].values.reshape(1, -1)
        xWhiff_rf = rf_model.predict_proba(row_features)[0][1]
        xWhiff_xg = xgb_model.predict_proba(row_features)[0][1]
        xWhiff = (xWhiff_rf + xWhiff_xg) / 2
        if pitch_type in ['Fastball', 'Sinker']:
            return (xWhiff / 0.18206374469443068) * 100
        elif pitch_type in ['Curveball', 'KnuckleCurve']:
            return (xWhiff / 0.30139757759674063) * 100
        elif pitch_type in ['Slider', 'Cutter']:
            return (xWhiff / 0.32823183402173944) * 100
        elif pitch_type in ['ChangeUp']:
            return (xWhiff / 0.32612872148563093) * 100


# Note: We update stuffappend so that it retains the columns needed for plotting (e.g., "Pitch Type" and "RelSide").
def stuffappend(bullpen):
    bullpen['Stuff+'] = bullpen.apply(calculate_stuff_plus, axis=1)
    return bullpen



def stuffhex(bullpen):
    pitch_types = bullpen['Pitch Type'].unique()
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes = axes.flatten()
    fig.patch.set_facecolor('white')
    if bullpen['RelSide'].iloc[0] > 0:
        base_pitch_data = pitches_with_stuff[pitches_with_stuff['PitcherThrows'] == 'Right']
    else:
        base_pitch_data = pitches_with_stuff[pitches_with_stuff['PitcherThrows'] == 'Left']
    for idx, pitch in enumerate(pitch_types):
        pitch_data = base_pitch_data[base_pitch_data['Pitch Type'] == pitch]
        bullpen_pitches = bullpen[bullpen['Pitch Type'] == pitch]
        bullpen_speed = bullpen_pitches['RelSpeed'].mean()
        avg_horizontal = bullpen_pitches['Horizontal Break (in)'].mean()
        avg_vertical = bullpen_pitches['InducedVertBreak'].mean()
        pitch_data = pitch_data.dropna(subset=['Horizontal Break (in)', 'InducedVertBreak', 'Stuff+'])
        pitch_data = pitch_data[~pitch_data.isin([np.inf, -np.inf]).any(axis=1)]
        Q1 = pitch_data['Stuff+'].quantile(0.25)
        Q3 = pitch_data['Stuff+'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        pitch_data = pitch_data[(pitch_data['Stuff+'] >= lower_bound) & (pitch_data['Stuff+'] <= upper_bound)]
        pitch_data = pitch_data[
            (pitch_data['RelSpeed'] > bullpen_speed - 1) &
            (pitch_data['RelSpeed'] < bullpen_speed + 1)
            ]
        pitch_data['Horizontal Break (in)'] = pitch_data['Horizontal Break (in)'].clip(upper=25)
        pitch_data['InducedVertBreak'] = pitch_data['InducedVertBreak'].clip(upper=25)
        pitch_data = pitch_data[pitch_data['Horizontal Break (in)'].abs() + pitch_data['InducedVertBreak'].abs() < 30]
        if len(pitch_data) > 0:
            max_break = 25
            min_break = -25
            hb = axes[idx].hexbin(
                x=pitch_data['Horizontal Break (in)'],
                y=pitch_data['InducedVertBreak'],
                C=pitch_data['Stuff+'],
                gridsize=40,
                cmap='coolwarm',
                mincnt=1
            )
            axes[idx].axhline(y=avg_vertical, color='gray', linestyle='--', alpha=0.5)
            axes[idx].axvline(x=avg_horizontal, color='gray', linestyle='--', alpha=0.5)
            axes[idx].plot(avg_horizontal, avg_vertical, 'kX', markersize=15, markeredgewidth=3)
            axes[idx].set_aspect('equal')
            axes[idx].set_xlim(min_break, max_break)
            axes[idx].set_ylim(min_break, max_break)
            axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[idx].set_title(pitch)
            axes[idx].set_xlabel('Horizontal Break')
            axes[idx].set_ylabel('Vertical Break')
            for spine in ['top', 'right', 'bottom', 'left']:
                axes[idx].spines[spine].set_visible(False)
            axes[idx].tick_params(axis='both', length=0)
            cbar = fig.colorbar(hb, ax=axes[idx], label='Stuff+', fraction=0.04, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0)
    for idx in range(len(pitch_types), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.01)
    plt.savefig('Graphs/stuffhexmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def stuffplusgraph(bullpen):
    sns.set_context("paper", font_scale=1.7,
                    rc={"font.family": "serif", "font.weight": "normal", "axes.labelweight": "normal"})
    pitch_types = bullpen['Pitch Type'].unique()
    fig, axes = plt.subplots(len(pitch_types), 1, figsize=(9, 2), sharex=True)
    # Ensure axes is an array, even if one subplot is returned.
    if len(pitch_types) == 1:
        axes = [axes]
    for i, pitch in enumerate(pitch_types):
        diffpitches = bullpen[bullpen['Pitch Type'] == pitch]
        mean_stuff = diffpitches['Stuff+'].mean()
        sns.kdeplot(diffpitches['Stuff+'], ax=axes[i], fill=True, color=pitch_colors[pitch])
        axes[i].set_facecolor('white')
        axes[i].axvline(x=mean_stuff, color=pitch_colors[pitch], linestyle=':')
        axes[i].set_ylabel(pitch, rotation=0, va='center', ha='left')
        axes[i].set_yticks([])
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].axvline(x=100, color='red', linestyle='--')
    axes[-1].set_xticks(np.arange(35, 250, 25))
    axes[-1].tick_params(axis='x', length=0, labelsize=10)
    axes[-1].set_xlabel('')
    plt.subplots_adjust(hspace=0.01)
    plt.xlim(min(bullpen['Stuff+'] - 30), max(bullpen['Stuff+']) + 20)
    fig.savefig('Graphs/stuff+_plot.png', bbox_inches='tight', dpi=300, format='png')
    plt.close(fig)


def velograph(bullpen):
    sns.set_context("paper", font_scale=1.7,
                    rc={"font.family": "serif", "font.weight": "normal", "axes.labelweight": "normal"})
    pitch_types = bullpen['Pitch Type'].unique()
    fig, axes = plt.subplots(len(pitch_types), 1, figsize=(9, 2), sharex=True)
    if len(pitch_types) == 1:
        axes = [axes]
    for i, pitch in enumerate(pitch_types):
        diffpitches = bullpen[bullpen['Pitch Type'] == pitch]
        mean_velo = diffpitches['RelSpeed'].mean()
        sns.kdeplot(diffpitches['RelSpeed'], ax=axes[i], fill=True, color=pitch_colors[pitch])
        axes[i].set_facecolor('white')
        axes[i].axvline(x=mean_velo, color=pitch_colors[pitch], linestyle=':')
        axes[i].set_ylabel(pitch, rotation=0, va='center', ha='left')
        axes[i].set_yticks([])
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
    axes[-1].set_xticks(np.arange(65, 105, 5))
    axes[-1].tick_params(axis='x', length=0, labelsize=10)
    axes[-1].set_xlabel('')
    plt.subplots_adjust(hspace=0.01)
    plt.xlim(min(bullpen['RelSpeed'] - 5), max(bullpen['RelSpeed']) + 1.5)
    fig.savefig('Graphs/velocity_plot.png', bbox_inches='tight', dpi=300, format='png')
    plt.close(fig)


def spinrategraph(bullpen):
    sns.set_context("paper", font_scale=1.7,
                    rc={"font.family": "serif", "font.weight": "normal", "axes.labelweight": "normal"})
    pitch_types = bullpen['Pitch Type'].unique()
    fig, axes = plt.subplots(len(pitch_types), 1, figsize=(9, 2), sharex=True)
    if len(pitch_types) == 1:
        axes = [axes]
    for i, pitch in enumerate(pitch_types):
        diffpitches = bullpen[bullpen['Pitch Type'] == pitch]
        mean_spin = diffpitches['SpinRate'].mean()
        sns.kdeplot(diffpitches['SpinRate'], ax=axes[i], fill=True, color=pitch_colors[pitch])
        axes[i].set_facecolor('white')
        axes[i].axvline(x=mean_spin, color=pitch_colors[pitch], linestyle=':')
        axes[i].set_ylabel(pitch, rotation=0, va='center', ha='left')
        axes[i].set_yticks([])
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
    axes[-1].set_xticks(np.arange(1500, 2500, 250))
    axes[-1].tick_params(axis='x', length=0, labelsize=10)
    axes[-1].set_xlabel('')
    plt.subplots_adjust(hspace=0.01)
    plt.xlim(min(bullpen['SpinRate'] - 30), max(bullpen['SpinRate']) + 20)
    fig.savefig('Graphs/spinrate_plot.png', bbox_inches='tight', dpi=300, format='png')
    plt.close(fig)


def locationheatmap(bullpen):
    pitch_types = bullpen['Pitch Type'].unique()
    num_pitches = len(pitch_types)
    sz_left, sz_right = -1.0083333, 1.083333
    sz_top, sz_bottom = 4, 1
    fig, axes = plt.subplots(1, num_pitches, figsize=(5*num_pitches, 5))
    axes = np.atleast_1d(axes)  # Fix: ensure axes is array-like
    for i, pitch in enumerate(pitch_types):
        pitch_data = bullpen[bullpen['Pitch Type'] == pitch]
        sns.kdeplot(
            data=pitch_data,
            x='Location Side (ft)',
            y='Location Height (ft)',
            fill=True, cmap='coolwarm',
            alpha=0.99, levels=100, ax=axes[i]
        )
        strikezone = patches.Rectangle(
            (sz_left, sz_bottom), sz_right*2, sz_top-sz_bottom,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        axes[i].add_patch(strikezone)
        axes[i].set_facecolor('white')
        axes[i].set_title(f'{pitch}', fontsize=32)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-1, 5.25)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig('Graphs/locationmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



def movementplot(bullpen):
    sns.scatterplot(data=bullpen, x="Horizontal Break (in)", y="InducedVertBreak", hue="Pitch Type",
                    palette=pitch_colors)
    fastball_index = bullpen[bullpen['Pitch Type'] == 'Fastball'].index[0]
    if bullpen.loc[fastball_index, 'RelSide'] > 0:
        compare = filtered_pitchers[filtered_pitchers['PitcherThrows'] == 'Right']
    else:
        compare = filtered_pitchers[filtered_pitchers['PitcherThrows'] == 'Left']
    for pitch_type in bullpen['Pitch Type'].unique():
        subset = compare[compare["Pitch Type"] == pitch_type]
        avghor = subset["Horizontal Break (in)"].mean()
        avgvert = subset["InducedVertBreak"].mean()
        plt.scatter(avghor, avgvert, color=pitch_colors[pitch_type], s=250, alpha=0.5)
    plt.scatter(10000, 10000, color='gray', s=250, alpha=0.5, label='Collegiate Average')
    x_max = max(abs(bullpen["Horizontal Break (in)"]))
    y_max = max(abs(bullpen["InducedVertBreak"]))
    limit = max(x_max + 1, y_max + 1)
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.gca().set_facecolor('white')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linestyle='-')
    plt.axvline(0, color='black', linestyle='-')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig('Graphs/movementplot.png', dpi=300, bbox_inches='tight')
    plt.close()


# def tableheatmap(bullpen):
#     metrics_mapping = {
#         'RelSpeed': 'Velocity',
#         'SpinRate': 'Spin Rate',
#         'InducedVertBreak': 'IVB',
#         'Horizontal Break (in)': 'HB',
#         'Extension': 'Extension',
#         'Vertical Approach Angle (°)': 'VAA',
#         'RelHeight': 'Release Height',
#         'Stuff+': 'Stuff+'
#     }
#     invert_metrics = ['Vertical Approach Angle (°)', 'RelHeight']
#     metrics = list(metrics_mapping.keys())
#     display_names = list(metrics_mapping.values())
#     actual_values = pd.DataFrame(index=bullpen['Pitch Type'].unique(), columns=metrics, dtype=float)
#     percentile_data = pd.DataFrame(index=bullpen['Pitch Type'].unique(), columns=metrics, dtype=float)
#     for pitch_type in bullpen['Pitch Type'].unique():
#         for metric in metrics:
#             actual_value = bullpen[bullpen['Pitch Type'] == pitch_type][metric].mean()
#             actual_values.loc[pitch_type, metric] = float(actual_value)
#             college_values = filtered_pitchers[filtered_pitchers['Pitch Type'] == pitch_type][metric].dropna()
#             if len(college_values) > 0:
#                 if metric in invert_metrics:
#                     percentile = 100 - stats.percentileofscore(college_values.abs(), abs(actual_value))
#                 else:
#                     percentile = stats.percentileofscore(college_values.abs(), abs(actual_value))
#                 percentile_data.loc[pitch_type, metric] = float(percentile)
#             else:
#                 percentile_data.loc[pitch_type, metric] = 50.0
#     actual_values.columns = display_names
#     percentile_data.columns = display_names
#     plt.figure(figsize=(14, 3))
#     heatmap = sns.heatmap(data=percentile_data, annot=actual_values.round(1), fmt='.1f',
#                           cmap='coolwarm', center=50, vmin=0, vmax=100, cbar_kws={'label': 'Percentile'})
#     heatmap.xaxis.set_label_position('top')
#     heatmap.xaxis.set_ticks_position('top')
#     plt.tick_params(axis='both', which='both', length=0)
#     plt.xticks(rotation=0)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig('Graphs/percentiletable.png', dpi=300, bbox_inches='tight')
#     plt.close()
def tableheatmap(bullpen):
    metrics_mapping = {
        'RelSpeed': 'Velocity',
        'SpinRate': 'Spin Rate',
        'InducedVertBreak': 'IVB',
        'Horizontal Break (in)': 'HB',
        'Extension': 'Extension',
        'Vertical Approach Angle (°)': 'VAA',
        'RelHeight': 'Release Height',
        'Stuff+': 'Stuff+'
    }
    invert_metrics = ['Vertical Approach Angle (°)', 'RelHeight']
    metrics = list(metrics_mapping.keys())
    display_names = list(metrics_mapping.values())
    actual_values = pd.DataFrame(index=bullpen['Pitch Type'].unique(), columns=metrics, dtype=float)
    percentile_data = pd.DataFrame(index=bullpen['Pitch Type'].unique(), columns=metrics, dtype=float)

    for pitch_type in bullpen['Pitch Type'].unique():
        for metric in metrics:
            actual_value = bullpen[bullpen['Pitch Type'] == pitch_type][metric].mean()
            actual_values.loc[pitch_type, metric] = float(actual_value)
            # Get precomputed college values from the dictionary
            college_vals = COLLEGE_VALUES.get(pitch_type, {}).get(metric, np.array([]))
            if len(college_vals) > 0:
                if metric in invert_metrics:
                    percentile = 100 - stats.percentileofscore(college_vals, abs(actual_value))
                else:
                    percentile = stats.percentileofscore(college_vals, abs(actual_value))
                percentile_data.loc[pitch_type, metric] = float(percentile)
            else:
                percentile_data.loc[pitch_type, metric] = 50.0

    actual_values.columns = display_names
    percentile_data.columns = display_names
    plt.figure(figsize=(14, 3))
    heatmap = sns.heatmap(data=percentile_data, annot=actual_values.round(1), fmt='.1f',
                          cmap='coolwarm', center=50, vmin=0, vmax=100, cbar_kws={'label': 'Percentile'})
    heatmap.xaxis.set_label_position('top')
    heatmap.xaxis.set_ticks_position('top')
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Graphs/percentiletable.png', dpi=300, bbox_inches='tight')
    plt.close()



def make_plots(bullpen):
    movementplot(bullpen)
    stuffplusgraph(bullpen)
    velograph(bullpen)
    spinrategraph(bullpen)
    locationheatmap(bullpen)
    tableheatmap(bullpen)
    stuffhex(bullpen)


##############################
# ENDPOINTS
##############################

# Endpoint for calculating Stuff+ from input pitch data
class PitchData(BaseModel):
    Pitch_Type: str
    RelSpeed: float
    SpinRate: float
    RelHeight: float
    ABS_RelSide: float
    Extension: float
    ABS_Horizontal: float = None
    InducedVertBreak: float = None
    differential_break: float = None


def calculate_stuff_plus_endpoint(row: pd.Series):
    pitch_type = row['Pitch_Type']
    if pitch_type in rf_models:
        rf_model = rf_models[pitch_type]
        xgb_model = xgb_models[pitch_type]
        if pitch_type in ['Fastball', 'Sinker']:
            features = ['RelSpeed', 'SpinRate', 'differential_break', 'RelHeight', 'ABS_RelSide', 'Extension']
        else:
            features = ['RelSpeed', 'SpinRate', 'ABS_Horizontal', 'RelHeight', 'ABS_RelSide', 'Extension',
                        'InducedVertBreak']
        try:
            row_features = row[features].values.reshape(1, -1)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Missing or invalid features: {e}")
        xWhiff_rf = rf_model.predict_proba(row_features)[0][1]
        xWhiff_xg = xgb_model.predict_proba(row_features)[0][1]
        xWhiff = (xWhiff_rf + xWhiff_xg) / 2
        if pitch_type in ['Fastball', 'Sinker']:
            return (xWhiff / 0.18206374469443068) * 100
        elif pitch_type in ['Curveball', 'KnuckleCurve']:
            return (xWhiff / 0.30139757759674063) * 100
        elif pitch_type in ['Slider', 'Cutter']:
            return (xWhiff / 0.32823183402173944) * 100
        elif pitch_type in ['ChangeUp']:
            return (xWhiff / 0.32612872148563093) * 100
    else:
        raise HTTPException(status_code=400, detail="Invalid pitch type")



def query_trackman_data(athlete_id, start_date, end_date):
    connection_string = os.environ.get("NEON_DB_CONNECTION")
    if not connection_string:
        raise ValueError("NEON_DB_CONNECTION is not set!")
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = """
    SELECT 
        "pitchReleaseSpeed",
        "spinRate",
        "pitchType",
        "pitcherName",
        "releaseHeight",
        "releaseSide",
        "extension",
        "inducedVerticalBreak",
        "horizontalBreak",
        "locationSide",
        "locationHeight",
        "verticalApproachAngle",
        "createdAt"
    FROM "Trackman"
    WHERE "athleteId" = %s
      AND "createdAt" BETWEEN %s AND %s
    ORDER BY "createdAt" ASC;
    """
    cursor.execute(query, (athlete_id, start_date, end_date))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


@app.post("/calculate-stuff")
def calculate_stuff_endpoint(pitch: PitchData):
    data = pitch.dict()
    if data["Pitch_Type"] in ["Fastball", "Sinker"]:
        if data.get("differential_break") is None:
            if data.get("ABS_Horizontal") is None or data.get("InducedVertBreak") is None:
                raise HTTPException(status_code=400,
                                    detail="Missing ABS_Horizontal or InducedVertBreak to compute differential_break")
            data["differential_break"] = abs(data["InducedVertBreak"] - data["ABS_Horizontal"])
    row = pd.Series(data)
    result = calculate_stuff_plus_endpoint(row)
    return {"stuff_plus": result}


# Data model for generating the Trackman report
class ReportData(BaseModel):
    date_one: str
    date_two: str
    athleteId: str


@app.post("/generate-trackman-report")
def generate_trackman_report(report_request: ReportData):
    # 1. Convert dates and fetch data from the database using athleteId and date range.
    try:
        start_date = datetime.datetime.fromisoformat(report_request.date_one)
        end_date = datetime.datetime.fromisoformat(report_request.date_two)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    athlete_id = report_request.athleteId

    # Replace this simulated query with your actual database query.
    # The query should return records with fields matching the Prisma model:
    # For demonstration, we simulate a list of dictionaries.
    db_data = query_trackman_data(athlete_id, start_date, end_date)

    if not db_data:
        raise HTTPException(status_code=404, detail="No Trackman data found")

    # 2. Convert the fetched data into a DataFrame and apply column mapping.
    trackman_df = pd.DataFrame(db_data)
    column_mapping = {
        'pitchReleaseSpeed': 'RelSpeed',
        'spinRate': 'SpinRate',
        'pitchType': 'Pitch Type',
        'releaseHeight': 'RelHeight',
        'releaseSide': 'RelSide',
        'extension': 'Extension',
        'horizontalBreak': 'Horizontal Break (in)',
        'inducedVerticalBreak': 'InducedVertBreak',
        'locationSide': 'Location Side (ft)',
        'locationHeight': 'Location Height (ft)',
        'verticalApproachAngle': 'Vertical Approach Angle (°)',
        'pitcherName': 'Pitcher Name'
    }
    trackman_df = trackman_df.rename(columns=column_mapping)

    # 3. Compute extra columns required for analysis.
    trackman_df['ABS_Horizontal'] = trackman_df['Horizontal Break (in)'].abs()
    trackman_df['ABS_RelSide'] = trackman_df['RelSide'].abs()
    trackman_df['differential_break'] = (trackman_df['InducedVertBreak'] - trackman_df['ABS_Horizontal']).abs()

    # Compute the Stuff+ column and keep required columns.
    trackman_df = stuffappend(trackman_df)

    # 4. Generate plots (this will save all graphs into the "Graphs" folder).
    make_plots(trackman_df)

    # 5. Build the PDF report using FPDF exactly as in your original code.
    class PDFWithConditionalFooter(FPDF):
        # Create footer on every page except the first.
        def footer(self):
            if self.page_no() != 1:
                box_x = 0
                box_y = self.h - 20
                box_width = self.w
                box_height = 20
                self.set_fill_color(3, 20, 55)
                self.rect(box_x, box_y, box_width, box_height, style='F')
                self.image('./Logos/logoWriting.png', x=box_x + 10, y=box_y + 2, w=60)

    pdf = PDFWithConditionalFooter('P', 'mm', 'A4')
    pagewidth = pdf.w

    # Page 1: Title Page
    pdf.add_page()
    pdf.image('./Logos/logo.png', x=(pagewidth / 2) - 55, y=40, w=110)
    pdf.set_font('Helvetica', 'B', 25)
    pdf.ln(150)
    pdf.cell(0, 15, 'TrackMan Pitching Report', 0, 1, align='C')
    pdf.set_font('Helvetica', 'B', 20)
    pitcher_name = trackman_df["Pitcher Name"].iloc[0] if "Pitcher Name" in trackman_df.columns else "Unknown"
    pdf.cell(0, 5, f'Pitcher: {pitcher_name}', 0, 1, align='C')

    # Page 2: Graphs – Velocity, Spin Rate, Stuff+, and Pitch Location
    pitching_metrics_w = 170
    pdf.add_page()
    pdf.cell(5, 5, 'Velocity (MPH)', align='L')
    pdf.image('Graphs/velocity_plot.png', x=(pagewidth - pitching_metrics_w) / 2, y=19.5, w=pitching_metrics_w)
    pdf.ln(60)
    pdf.cell(5, 5, 'Spin Rate (RPM)', align='L')
    pdf.image('Graphs/spinrate_plot.png', x=(pagewidth - pitching_metrics_w) / 2, y=79.5, w=pitching_metrics_w)
    pdf.ln(60)
    pdf.cell(5, 5, 'Stuff+', align='L')
    pdf.image('Graphs/stuff+_plot.png', x=(pagewidth - pitching_metrics_w) / 2, y=137.5, w=pitching_metrics_w)
    pdf.ln(60)
    pdf.cell(5, 5, 'Pitch Location', align='L')
    pdf.image('Graphs/locationmap.png', x=(pagewidth - 200) / 2, y=197.5, w=200)

    # Page 3: Graphs – Movement Plot and Percentile Table
    pdf.add_page()
    pdf.cell(5, 5, 'Pitch Movement (in)', align='L')
    pdf.image('Graphs/movementplot.png', x=(pagewidth - 200) / 2, y=17.5, w=200)
    pdf.ln(160)
    pdf.cell(5, 5, 'Movement Stats', align='L')
    pdf.image('Graphs/percentiletable.png', x=(pagewidth - 200) / 2, y=180, w=200)

    # Page 4: Graph – Stuff+ Blob
    pdf.add_page()
    pdf.cell(5, 5, 'Your Stuff+ vs Pitches with Similar Velo', align='L')
    pdf.image('Graphs/stuffhexmap.png', x=(pagewidth - 200) / 2, y=17.5, w=200)

    # Output the PDF into a BytesIO buffer.
    buffer = BytesIO()
    pdf_data = pdf.output(dest='S').encode('latin-1')
    buffer = BytesIO(pdf_data)
    buffer.seek(0)

    # Delete temporary image files after generating the PDF.
    files_to_delete = [
        'Graphs/velocity_plot.png',
        'Graphs/spinrate_plot.png',
        'Graphs/stuff+_plot.png',
        'Graphs/locationmap.png',  # Make sure you save this with a .png extension
        'Graphs/movementplot.png',  # Save as .png instead of without extension
        'Graphs/percentiletable.png',  # Save as .png if needed
        'Graphs/stuffhexmap.png'
    ]
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Return the PDF as the response.
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=trackman_report.pdf"}
    )


@app.get("/")
def health_check():
    return {"Server": "Healthy"}
