import pandas as pd
import os
import shutil

# =====================================================================
# CONFIGURATION: Set this variable to the state you want (e.g., 'TN', 'IN')
# If you want to use the automatic selection, set to 'CALCULATE'
# =====================================================================
SELECTED_STATE = 'TX' 
# =====================================================================

# Dictionary to map abbreviations to full names for folder naming
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
    'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

def create_filtered_dataset(target_state_abbr, folder_suffix):
    """
    Creates a new directory and populates it with subset datasets 
    for the specific target_state_abbr.
    """
    state_full_name = STATE_NAMES.get(target_state_abbr, target_state_abbr)
    out_dir = f"Inputs_{folder_suffix}_{state_full_name}"
    
    # Create the output directory in the current working directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Creating dataset in directory: {out_dir} ---")

    # 1. BASE FILE: NHS_US_5miles_with_SCS.csv
    print("Processing Base file (5miles)...")
    df_base = pd.read_csv('NHS_US_5miles_with_SCS.csv')
    
    # Re-extract state abbr just in case
    state_series = df_base['FILE_NAME'].astype(str).str.split('_').str[0]
    df_base_sub = df_base[state_series == target_state_abbr].copy()
    
    # Save the base subset
    df_base_sub.to_csv(os.path.join(out_dir, 'NHS_US_5miles_with_SCS.csv'), index=False)
    
    # Get the valid sets of IDs for downstream filtering
    valid_segments = set(df_base_sub['SEGMENT_ID_OG'].dropna().unique())
    valid_scs = set(df_base_sub['NEAR_SCS_ID'].dropna().unique())
    
    print(f"Extracted {len(valid_segments)} unique segments and {len(valid_scs)} unique SCS IDs for {state_full_name}.")

    # 2. ADJACENT FILE: NHS_USA_5miles_Adjacent.csv
    print("Processing Adjacent file...")
    df_adj = pd.read_csv('NHS_USA_5miles_Adjacent.csv')
    # Both TARGET_FID and JOIN_FID must belong to the valid segment list
    df_adj_sub = df_adj[df_adj['TARGET_FID'].isin(valid_segments) & df_adj['JOIN_FID'].isin(valid_segments)]
    df_adj_sub.to_csv(os.path.join(out_dir, 'NHS_USA_5miles_Adjacent.csv'), index=False)

    # 3. SUBSTATION DISTANCE FILE: NHS_US_SubstationDistance.csv
    print("Processing Substation Distance file...")
    df_subdist = pd.read_csv('NHS_US_SubstationDistance.csv')
    # Filter by IN_FID
    df_subdist_sub = df_subdist[df_subdist['IN_FID'].isin(valid_segments)]
    df_subdist_sub.to_csv(os.path.join(out_dir, 'NHS_US_SubstationDistance.csv'), index=False)

    # 4. EXISTING SCS FILE: US_SCS_unique_existing.csv
    print("Processing Existing SCS file...")
    df_scs = pd.read_csv('US_SCS_unique_existing.csv')
    # Filter by SCS_ID matching NEAR_SCS_ID from base file
    df_scs_sub = df_scs[df_scs['SCS_ID'].isin(valid_scs)]
    df_scs_sub.to_csv(os.path.join(out_dir, 'US_SCS_unique_existing.csv'), index=False)

    # 5. SUBSTATIONS PROJECT FILE: SubstationsProject.csv
    print("Copying SubstationsProject file...")
    shutil.copy('SubstationsProject.csv', os.path.join(out_dir, 'SubstationsProject.csv'))

    print(f"Success! Data for {state_full_name} has been written to {out_dir}/")


def main(selection):
    print("Loading Base File to calculate state metrics...")
    df_base = pd.read_csv('NHS_US_5miles_with_SCS.csv')
    
    # Extract the State abbreviation from the FILE_NAME (e.g., 'IN_NHS_2021' -> 'IN')
    df_base['State_Abbr'] = df_base['FILE_NAME'].astype(str).str.split('_').str[0]
    
    # Convert metric columns to numeric (forces 'NOT AVAILABLE' or strings to NaN)
    # Note: Assuming 'Norm_index' represents the Affordability cost multiplier
    metric_cols = ['AADT_COM', 'Total_Electricity_Rate_Nom_2024', 'Norm_index']
    for col in metric_cols:
        df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
        
    # Group by State and calculate the averages to determine targets
    state_metrics = df_base.groupby('State_Abbr')[metric_cols].mean()
    
    # Calculate Representative State using Z-score Normalization and Euclidean Distance
    state_metrics_scaled = (state_metrics - state_metrics.mean()) / state_metrics.std()
    distance_from_national_avg = (state_metrics_scaled ** 2).sum(axis=1) ** 0.5
    representative_state_abbr = distance_from_national_avg.idxmin()
    
    # Identify target states
    targets = {
        'highest_aadt': (state_metrics['AADT_COM'].idxmax(), "HighestAADT"),
        'lowest_aadt': (state_metrics['AADT_COM'].idxmin(), "LowestAADT"),
        'highest_elec': (state_metrics['Total_Electricity_Rate_Nom_2024'].idxmax(), "HighestElectricity"),
        'lowest_elec': (state_metrics['Total_Electricity_Rate_Nom_2024'].idxmin(), "LowestElectricity"),
        'highest_affordability': (state_metrics['Norm_index'].idxmax(), "HighestAffordability"),
        'lowest_affordability': (state_metrics['Norm_index'].idxmin(), "LowestAffordability"),
        'representative': (representative_state_abbr, "RepresentativeState")
    }
    
    if selection not in targets:
        print(f"Invalid selection. Please choose from: {list(targets.keys())}")
        return

    target_state_abbr, folder_suffix = targets[selection]
    
    print(f"Based on your selection '{selection}', the calculated target state is {target_state_abbr}.")
    
    # Process the dataset
    create_filtered_dataset(target_state_abbr, folder_suffix)


if __name__ == "__main__":
    # =====================================================================
    # MODIFY THIS VARIABLE TO CHOOSE WHICH DATASET TO GENERATE
    # Options:
    # 'highest_aadt', 'lowest_aadt'
    # 'highest_elec', 'lowest_elec'
    # 'highest_affordability', 'lowest_affordability'
    # 'representative'
    # =====================================================================
    
    TARGET_DATASET = 'representative'  
    
    if SELECTED_STATE == 'CALCULATE':
        main(TARGET_DATASET)
    else:
        create_filtered_dataset(SELECTED_STATE, "Manual")














# """
# US State Infrastructure Metrics — 4-panel Nature Energy–style figure
# =====================================================================
# Panels
#   A  Median Commercial Traffic (AADT_COM)
#   B  Median Affordability Multiplier (Norm_index)
#   C  Median Total Electricity Rate (2024)
#   D  Median Nearest-Substation Distance (miles)

# Outputs (both variants):
#   US_State_Metrics_lollipop.png / .pdf
#   US_State_Metrics_bar.png      / .pdf

# Data sources
#   NHS_US_5miles_with_SCS.csv
#   NHS_US_SubstationDistance.csv
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np
# import os

# # ── Conversion ────────────────────────────────────────────────────────────────
# M_TO_MILES = 0.000621371

# # ── Typography baseline ───────────────────────────────────────────────────────
# plt.rcParams.update({
#     'font.family'       : 'sans-serif',
#     'font.sans-serif'   : ['Arial', 'Liberation Sans', 'DejaVu Sans'],
#     'pdf.fonttype'      : 42,
#     'ps.fonttype'       : 42,
#     'figure.dpi'        : 150,
#     'axes.linewidth'    : 0.9,
#     'xtick.major.width' : 0.8,
#     'ytick.major.width' : 0.8,
# })

# # ── Colour palette ─────────────────────────────────────────────────────────────
# # Above-median: full-saturation; below-median: 45% lighter version
# ABOVE = {'A': '#1A5EA8', 'B': '#C0392B', 'C': '#1A7D72', 'D': '#5B2182'}
# BELOW = {'A': '#91B8DC', 'B': '#E89C96', 'C': '#82C4BE', 'D': '#B18FCC'}
# MEDIAN_LINE = '#222222'
# GRID_COLOR  = '#E0E0E0'
# STEM_COLOR  = '#C8C8C8'
# TEXT_COLOR  = '#1A1A1A'   # near-black for all text
# LABEL_COLOR = '#333333'   # axis labels / tick labels

# # ─────────────────────────────────────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────────────────────────────────────

# def load_data(seg_path: str, sub_path: str) -> pd.DataFrame:
#     print(f"  Loading segments : {seg_path}")
#     seg = pd.read_csv(seg_path, low_memory=False)
#     seg['State'] = seg['FILE_NAME'].astype(str).str.split('_').str[0]

#     metric_cols = ['AADT_COM', 'Norm_index', 'Total_Electricity_Rate_Nom_2024']
#     for c in metric_cols:
#         seg[c] = pd.to_numeric(seg[c], errors='coerce')
#     seg = seg[['State', 'SEGMENT_ID_OG'] + metric_cols].copy()

#     print(f"  Loading substation: {sub_path}")
#     sub = pd.read_csv(sub_path, low_memory=False)
#     sub = sub[sub['NEAR_RANK'] == 1][['IN_FID', 'NEAR_DIST']].copy()
#     sub.rename(columns={'IN_FID': 'SEGMENT_ID_OG'}, inplace=True)
#     sub['Sub_Dist_Miles'] = sub['NEAR_DIST'] * M_TO_MILES

#     merged = seg.merge(sub[['SEGMENT_ID_OG', 'Sub_Dist_Miles']],
#                        on='SEGMENT_ID_OG', how='left')

#     all_cols = metric_cols + ['Sub_Dist_Miles']
#     national_medians = merged[all_cols].median()

#     state_med = (merged.groupby('State')[all_cols]
#                        .median()
#                        .dropna(how='all')
#                        .sort_index(ascending=True))
#     state_med.attrs['national_medians'] = national_medians

#     print(f"  States found     : {len(state_med)}")
#     print(f"  National medians :\n{national_medians.to_string()}\n")
#     return state_med


# # ─────────────────────────────────────────────────────────────────────────────
# # SHARED PANEL STYLING
# # ─────────────────────────────────────────────────────────────────────────────

# def _style_panel(ax, ylabel: str, title: str, letter: str,
#                  n_states: int, yformatter=None):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_color('#AAAAAA')
#     ax.spines['bottom'].set_color('#AAAAAA')

#     ax.yaxis.grid(True, linestyle=':', linewidth=0.65,
#                   color=GRID_COLOR, zorder=0)
#     ax.set_axisbelow(True)

#     ax.set_ylabel(ylabel, fontsize=10, color=LABEL_COLOR,
#                   fontweight='bold', labelpad=7)
#     ax.set_xlabel('US State / Territory (A → Z)', fontsize=9,
#                   color=LABEL_COLOR, labelpad=6)

#     # X-ticks: width proportional to n_states so labels don't overlap
#     ax.tick_params(axis='x', rotation=90, labelsize=8,
#                    colors=TEXT_COLOR, length=3, pad=3)
#     ax.tick_params(axis='y', labelsize=9, colors=TEXT_COLOR, length=3)

#     ax.set_title(f'$\\mathbf{{{letter}}}$  {title}', fontsize=10.5,
#                  loc='left', pad=8, color=TEXT_COLOR)

#     if yformatter is not None:
#         ax.yaxis.set_major_formatter(yformatter)


# def _add_median_annotation(ax, val, fmt: str):
#     ax.axhline(y=val, color=MEDIAN_LINE, linestyle='--',
#                linewidth=1.1, zorder=5, alpha=0.85)
#     ax.text(1.015, val, fmt.format(val),
#             transform=ax.get_yaxis_transform(),
#             color=MEDIAN_LINE, va='center', ha='left',
#             fontsize=8.5, fontweight='bold')


# # ─────────────────────────────────────────────────────────────────────────────
# # LOLLIPOP VARIANT
# # ─────────────────────────────────────────────────────────────────────────────

# def _draw_lollipop_panel(ax, states, values, panel_key: str,
#                           national_val: float,
#                           ylabel: str, title: str, letter: str,
#                           fmt_median: str = '{:.2f}',
#                           yformatter=None):
#     n = len(states)
#     x = np.arange(n)
#     above = values >= national_val

#     ax.vlines(x=x, ymin=0, ymax=values,
#               color=STEM_COLOR, linewidth=0.8, zorder=2)
#     ax.scatter(x[above],  values[above],
#                color=ABOVE[panel_key], s=32,
#                zorder=3, edgecolors='white', linewidths=0.5)
#     ax.scatter(x[~above], values[~above],
#                color=BELOW[panel_key], s=32,
#                zorder=3, edgecolors='white', linewidths=0.5)

#     _add_median_annotation(ax, national_val, fmt_median)
#     ax.set_xticks(x)
#     ax.set_xticklabels(states)
#     # Tighter margins so dots don't crowd edges
#     ax.set_xlim(-0.7, n - 0.3)
#     ax.set_ylim(bottom=0)
#     _style_panel(ax, ylabel, title, letter, n, yformatter)


# # ─────────────────────────────────────────────────────────────────────────────
# # BAR VARIANT
# # ─────────────────────────────────────────────────────────────────────────────

# def _draw_bar_panel(ax, states, values, panel_key: str,
#                      national_val: float,
#                      ylabel: str, title: str, letter: str,
#                      fmt_median: str = '{:.2f}',
#                      yformatter=None):
#     n = len(states)
#     x = np.arange(n)
#     above = values >= national_val
#     bar_w = 0.55  # slightly narrower than default for breathing room

#     for i, (v, ab) in enumerate(zip(values, above)):
#         color = ABOVE[panel_key] if ab else BELOW[panel_key]
#         ax.bar(i, v, width=bar_w, color=color, zorder=3,
#                linewidth=0, alpha=0.92)

#     _add_median_annotation(ax, national_val, fmt_median)
#     ax.set_xticks(x)
#     ax.set_xticklabels(states)
#     ax.set_xlim(-0.7, n - 0.3)
#     ax.set_ylim(bottom=0)
#     _style_panel(ax, ylabel, title, letter, n, yformatter)


# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE FACTORY
# # ─────────────────────────────────────────────────────────────────────────────

# # Panel config: (col_key, ylabel, title, letter, fmt_median, yformatter)
# def _panel_configs():
#     k_fmt = mticker.FuncFormatter(
#         lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else f'{v:.0f}')
#     return [
#         ('AADT_COM',
#          'Vehicles per Day',
#          'Median Commercial Traffic (AADT)',
#          'A', '{:,.0f}', k_fmt),
#         ('Norm_index',
#          'Index Score',
#          'Median Affordability Multiplier',
#          'B', '{:.2f}', None),
#         ('Total_Electricity_Rate_Nom_2024',
#          '¢ / kWh',
#          'Median Total Electricity Rate (2024)',
#          'C', '{:.2f}', None),
#         ('Sub_Dist_Miles',
#          'Miles',
#          'Median Nearest-Substation Distance',
#          'D', '{:.1f}', None),
#     ]

# PANEL_KEYS = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}


# def create_figure(state_med: pd.DataFrame,
#                   variant: str = 'lollipop',
#                   out_prefix: str = 'US_State_Metrics'):
#     """
#     variant: 'lollipop' | 'bar'
#     """
#     assert variant in ('lollipop', 'bar')
#     nat     = state_med.attrs['national_medians']
#     states  = np.array(state_med.index.tolist())
#     n       = len(states)

#     # Wide figure — extra horizontal space so 40+ x-labels don't collide
#     fig_w = max(18, n * 0.38)
#     fig, axes = plt.subplots(
#         2, 2,
#         figsize=(fig_w, 12),
#         gridspec_kw={'hspace': 0.60, 'wspace': 0.30},
#     )
#     ax_pos = {'A': axes[0, 0], 'B': axes[0, 1],
#               'C': axes[1, 0], 'D': axes[1, 1]}

#     draw_fn = _draw_lollipop_panel if variant == 'lollipop' else _draw_bar_panel

#     configs = _panel_configs()
#     panel_key_map = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}

#     for col_key, ylabel, title, letter, fmt_m, yfmt in configs:
#         vals = state_med[col_key].values.astype(float)
#         draw_fn(
#             ax=ax_pos[letter],
#             states=states,
#             values=vals,
#             panel_key=letter,
#             national_val=float(nat[col_key]),
#             ylabel=ylabel,
#             title=title,
#             letter=letter,
#             fmt_median=fmt_m,
#             yformatter=yfmt,
#         )

#     # ── Figure-level header ───────────────────────────────────────────────────
#     fig.text(
#         0.5, 0.998,
#         'State-Level Infrastructure & Cost Metrics — NHS Commercial Segments',
#         ha='center', va='top', fontsize=12,
#         fontweight='bold', color=TEXT_COLOR,
#     )
#     fig.text(
#         0.5, 0.977,
#         'Dashed line = national median across all segments  ·  '
#         'Faded colour = below national median',
#         ha='center', va='top', fontsize=8.5,
#         color=LABEL_COLOR, style='italic',
#     )

#     suffix = 'lollipop' if variant == 'lollipop' else 'bar'
#     for ext in ('png', 'pdf'):
#         path = f'{out_prefix}_{suffix}.{ext}'
#         fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
#         print(f"  Saved: {path}")
#     plt.close(fig)


# # ─────────────────────────────────────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     SEG_FILE = 'NHS_US_5miles_with_SCS.csv'
#     SUB_FILE = 'NHS_US_SubstationDistance.csv'

#     for f in (SEG_FILE, SUB_FILE):
#         if not os.path.exists(f):
#             raise FileNotFoundError(f"Missing input: {f}")

#     print("=== Loading data ===")
#     state_med = load_data(SEG_FILE, SUB_FILE)

#     print("=== Generating lollipop figure ===")
#     create_figure(state_med, variant='lollipop',
#                   out_prefix='US_State_Metrics')

#     print("=== Generating bar figure ===")
#     create_figure(state_med, variant='bar',
#                   out_prefix='US_State_Metrics')

#     print("\nDone. Output files:")
#     for s in ('lollipop', 'bar'):
#         for e in ('png', 'pdf'):
#             print(f"  US_State_Metrics_{s}.{e}")