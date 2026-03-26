# === Environment Setup ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
from pathlib import Path
from collections import Counter

# Professional publication style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Color palette
COLORS = {
    'baseline': '#9CA3AF',     # Gray
    'timer': '#2563EB',        # Blue
    'success': '#16A34A',      # Green
    'failure': '#DC2626',      # Red
    'semantic_collision': '#8B5CF6',  # Purple
    'negation_recency': '#F59E0B',    # Amber
    'terminology_drift': '#EC4899',    # Pink
    'real_world': '#14B8A6',          # Teal
}

# Paths
ROOT = Path('.').resolve()
DATA_EVAL = ROOT / 'data' / 'evaluation'
RESULTS_P5 = ROOT / 'results' / 'phase5'
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

print(f'✓ Environment configured')
print(f'  Root: {ROOT}')
print(f'  Results: {RESULTS_P5}')
# === Load Phase 1-5 Results ===

# Phase 1 & 3
with open(DATA_EVAL / 'phase_1_baseline_results.json') as f:
    phase1 = json.load(f)
with open(DATA_EVAL / 'phase_3_results.json') as f:
    phase3 = json.load(f)

# Phase 5 - Load all scenario CSVs
scenarios = {}
for csv_file in RESULTS_P5.glob('*_results.csv'):
    scenario_name = csv_file.stem.replace('_results', '')
    scenarios[scenario_name] = pd.read_csv(csv_file)
    print(f'  Loaded: {scenario_name} ({len(scenarios[scenario_name])} queries)')

# Phase 5 Summary
summary = pd.read_csv(RESULTS_P5 / 'summary_table.csv')

# Merge all scenarios into single dataframe
all_results = pd.concat([
    df.assign(scenario=name) for name, df in scenarios.items()
], ignore_index=True)

print(f'\n✓ Loaded {len(all_results)} total queries across {len(scenarios)} scenarios')
# Quick summary stats
print('=== Dataset Summary ===')
print(f"Total Queries: {len(all_results)}")
print(f"\nBy Scenario:")
print(all_results.groupby('scenario').size().to_string())
print(f"\nBy Intent:")
print(all_results.groupby('intent').size().to_string())
# === Figure 1: Research Phase Progression ===
import seaborn as sns
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Compile metrics across phases
phases = {
    'Phase 1\n(Naive RAG)': {'recall5': phase1['results']['naive']['Recall@5'], 'type': 'Baseline'},
    'Phase 1\n(Hybrid)': {'recall5': phase1['results']['poc']['Recall@5'], 'type': 'Increment'},
    'Phase 3\n(Cross-Encoder)': {'recall5': phase3['results']['Phase 3 (Reranking)']['Recall@5'], 'type': 'Increment'}
}

# Add Phase 5 TIMER result (overall accuracy on hard negatives)
overall_row = summary[summary['Scenario'] == '**OVERALL**']
if len(overall_row) > 0:
    timer_acc = float(overall_row['TIMER Acc'].values[0].replace('%', '')) / 100
    phases['Phase 5\n(TIMER-Graph)'] = {'recall5': timer_acc, 'type': 'Novel Contribution'}

df_prog = pd.DataFrame([
    {'Phase': p, 'Accuracy': data['recall5'], 'Type': data['type']}
    for p, data in phases.items()
])

# Journal-worthy palette (Cool gray, azure blue, deep royal)
pal = {'Baseline': '#ced4da', 'Increment': '#48cae4', 'Novel Contribution': '#023e8a'}

# Draw bars with seaborn
sns.barplot(
    data=df_prog, x='Phase', y='Accuracy', hue='Type', 
    palette=pal, edgecolor=".2", linewidth=1.5,
    dodge=False, ax=ax
)

# Customize spines (clean, academic look)
sns.despine(left=True, bottom=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels and improvement annotations
for i, row in df_prog.iterrows():
    # Bar height text
    ax.text(i, row['Accuracy'] + 0.02, f"{row['Accuracy']:.0%}", 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#343a40')
    
    # Delta improvements
    if i > 0:
        prev = df_prog.iloc[i-1]['Accuracy']
        delta = row['Accuracy'] - prev
        if delta > 0:
            mid_x = i - 0.5
            mid_y = (row['Accuracy'] + prev) / 2
            ax.annotate(f'+{delta:.0%}', xy=(mid_x, mid_y), fontsize=11, 
                       color='#d90429', fontweight='bold', ha='center',
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

# Styling
ax.set_ylabel('Recall@5 / Hard Negative Accuracy', fontsize=13, fontweight='medium', labelpad=15)
ax.set_xlabel('')
ax.set_ylim(0, 1.15)
ax.axhline(y=0.5, color='#adb5bd', linestyle=':', linewidth=2, zorder=0)
ax.text(3.4, 0.51, 'Random Chance', color='#6c757d', fontsize=10, ha='right', va='bottom')

# Title 
ax.set_title('Accuracy Progression: Naive RAG to TIMER-Graph', fontsize=15, fontweight='bold', pad=20)

# Legend positioning
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='', loc='upper left', frameon=True, fontsize=11, edgecolor='silver', facecolor='white')

# Add abstract annotation box
final_imp = timer_acc - phase1["results"]["naive"]["Recall@5"]
ax.text(0.98, 0.05, f'Absolute Accuracy Gain: +{final_imp:.0%}',
       transform=ax.transAxes, ha='right', va='bottom', fontsize=11, fontweight='bold', color='#023e8a',
       bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#023e8a', alpha=0.9, pad=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval1_phase_progression.png', bbox_inches='tight', facecolor='white', dpi=300)
plt.savefig(PLOTS_DIR / 'eval1_phase_progression.pdf', bbox_inches='tight', dpi=300)
print(f'✓ Saved: {PLOTS_DIR}/eval1_phase_progression.png')

# === Figure 2: Grouped Bar Chart - Baseline vs TIMER ===
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Prepare data for Seaborn (long format)
df_scenarios = summary[summary['Scenario'] != '**OVERALL**'].copy()
df_scenarios['Baseline'] = df_scenarios['Baseline Acc'].str.replace('%', '').astype(float)
df_scenarios['TIMER'] = df_scenarios['TIMER Acc'].str.replace('%', '').astype(float)

# Scenario mapping for cleaner labels
scenario_map = {
    'semantic_collision': 'Semantic\nCollision',
    'negation_recency': 'Negation\nRecency',
    'terminology_drift': 'Terminology\nDrift',
    'real_world_mining': 'Real-World\nMining'
}
df_scenarios['Display Name'] = df_scenarios['Scenario'].map(lambda x: f"{scenario_map.get(x, x)}\n(n={df_scenarios[df_scenarios['Scenario']==x]['N'].values[0]})")

df_long = pd.melt(df_scenarios, id_vars=['Display Name', 'Sig'], value_vars=['Baseline', 'TIMER'],
                  var_name='Model', value_name='Accuracy')
df_long['Model'] = df_long['Model'].replace({'Baseline': 'Semantic Baseline', 'TIMER': 'TIMER-Graph'})

# Engaging Color Palette (Orange and Teal)
pal = {'Semantic Baseline': '#d95f02', 'TIMER-Graph': '#1b9e77'}

# Plot
sns.barplot(data=df_long, x='Display Name', y='Accuracy', hue='Model', 
            palette=pal, edgecolor=".2", linewidth=1.5, ax=ax)

# Style
sns.despine(ax=ax, left=True)
ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=10)
ax.set_xlabel('', fontsize=12)
ax.set_title('Hard Negative Stress Test Results ($n=200$)', fontsize=14, fontweight='bold', pad=20)

# Add value labels and significance markers
for i, scenario in enumerate(df_scenarios['Display Name']):
    # Get values for this scenario
    b_val = df_scenarios.iloc[i]['Baseline']
    t_val = df_scenarios.iloc[i]['TIMER']
    sig = df_scenarios.iloc[i]['Sig']
    
    # Position for bars (using index i and width offset)
    # Seaborn bar centers are at integer positions
    ax.text(i - 0.2, b_val + 1, f'{b_val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i + 0.2, t_val + 1, f'{t_val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Significance marker (ns or stars)
    sig_text = sig if sig != 'ns' else 'ns'
    ax.text(i, max(b_val, t_val) + 10, sig_text, ha='center', va='bottom', 
            fontsize=11, fontweight='bold', color='#4a4a4a')

# Headroom and Legend
ax.set_ylim(0, 130) # Increased to 130 for significance text
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
          frameon=True, edgecolor='silver', facecolor='white')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval2_scenario_comparison.png', bbox_inches='tight', facecolor='white', dpi=300)
plt.savefig(PLOTS_DIR / 'eval2_scenario_comparison.pdf', bbox_inches='tight', dpi=300)
print(f'✓ Saved: {PLOTS_DIR}/eval2_scenario_comparison.png')

# === Figure 3: Per-Query Win/Loss Heatmap ===
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Prepare data for heatmap
all_results['baseline_correct_int'] = all_results['baseline_correct'].astype(int)
all_results['timer_correct_int'] = all_results['timer_correct'].astype(int)

# Sort exactly as requested
scenario_order = {
    'semantic_collision': 0, 'collision_multi_temporal': 0,
    'negation_recency': 1, 'negation_subsequent': 1,
    'terminology_drift': 2,
    'real_world_mining': 3, 'mining_base': 3
}
all_results['scenario_idx'] = all_results['scenario'].map(lambda x: scenario_order.get(x, 99))
heatmap_data = all_results.sort_values(['scenario_idx', 'query_id'])

# Reshape into 10x20
base_matrix = heatmap_data['baseline_correct_int'].values.reshape(10, 20)
timer_matrix = heatmap_data['timer_correct_int'].values.reshape(10, 20)

from matplotlib.colors import ListedColormap
# Custom colormap: orange for wrong, green for correct
cmap = ListedColormap(['#d95f02', '#1b9e77']) 

# Plot Baseline
axes[0].imshow(base_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
axes[0].set_title('Semantic Baseline', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Query row (20 per row)', fontsize=11, labelpad=30)

# Plot TIMER
axes[1].imshow(timer_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
axes[1].set_title('TIMER-Graph', fontsize=11, fontweight='bold')

# Add grid lines and grouping boundaries
import numpy as np
for ax in axes:
    ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Boundary lines for scenarios
    ax.axhline(2.5, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(4.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(5.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)

# Scenario labels
axes[0].text(-2.2, 1.25, 'SC', ha='right', va='center', fontsize=9, color='gray')
axes[0].text(-2.2, 3.25, 'NR', ha='right', va='center', fontsize=9, color='gray')
axes[0].text(-2.2, 4.5, 'TD', ha='right', va='center', fontsize=9, color='gray')
axes[0].text(-2.2, 7.5, 'RWM', ha='right', va='center', fontsize=9, color='gray')

# Summary stats
base_acc = heatmap_data['baseline_correct_int'].mean() * 100
timer_acc = heatmap_data['timer_correct_int'].mean() * 100

axes[0].text(0.5, -0.1, f'Accuracy: {base_acc:.1f}%', transform=axes[0].transAxes, 
             ha='center', va='top', fontsize=11, fontweight='bold', color='#d95f02')
axes[1].text(0.5, -0.1, f'Accuracy: {timer_acc:.1f}%', transform=axes[1].transAxes, 
             ha='center', va='top', fontsize=11, fontweight='bold', color='#1b9e77')

plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
fig.suptitle('Per-Query Correctness ($n=200$)', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(PLOTS_DIR / 'eval3_query_heatmap.png', bbox_inches='tight', facecolor='white', dpi=300)
plt.savefig(PLOTS_DIR / 'eval3_query_heatmap.pdf', bbox_inches='tight', dpi=300)
print(f'✓ Saved: {PLOTS_DIR}/eval3_query_heatmap.png')

# === Figure 4: Intent Detection Confusion Matrix ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4a: Confusion matrix
ax = axes[0]

# Create confusion data
ground_truth = all_results['intent'].values
detected = all_results['timer_intent_detected'].values

labels = ['historical', 'current']
confusion = np.zeros((2, 2))
for gt, det in zip(ground_truth, detected):
    if gt in labels and det in labels:
        i = labels.index(gt)
        j = labels.index(det)
        confusion[i, j] += 1

# Plot heatmap
im = ax.imshow(confusion, cmap='Blues')

# Labels
ax.set_xticks([0, 1])
ax.set_xticklabels(['Historical', 'Current'], fontsize=11)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Historical', 'Current'], fontsize=11)
ax.set_xlabel('Detected Intent', fontsize=12)
ax.set_ylabel('Ground Truth Intent', fontsize=12)

# Add values
for i in range(2):
    for j in range(2):
        color = 'white' if confusion[i, j] > confusion.max()/2 else 'black'
        ax.text(j, i, f'{int(confusion[i, j])}', ha='center', va='center', 
               fontsize=16, fontweight='bold', color=color)

# Calculate accuracy
accuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()
ax.set_title(f'Intent Classification Confusion Matrix\n(Accuracy: {accuracy:.0%})', fontsize=13, fontweight='bold')

# 4b: Intent distribution by scenario
ax = axes[1]

# Group by scenario and intent
intent_dist = all_results.groupby(['scenario', 'intent']).size().unstack(fill_value=0)

# Stacked bar
intent_dist.plot(kind='bar', stacked=True, ax=ax, 
                color=[COLORS['failure'], COLORS['success']], edgecolor='black')

ax.set_ylabel('Number of Queries', fontsize=12)
ax.set_xlabel('Scenario', fontsize=12)
ax.set_title('Query Intent Distribution by Scenario', fontsize=13, fontweight='bold')
ax.legend(title='Intent', fontsize=10)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval4_intent_analysis.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval4_intent_analysis.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval4_intent_analysis.png')

# === Figure 5: TIMER Win Analysis ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Identify improvement cases
all_results['timer_win'] = (all_results['baseline_correct'] == False) & (all_results['timer_correct'] == True)
all_results['both_correct'] = (all_results['baseline_correct'] == True) & (all_results['timer_correct'] == True)
all_results['both_wrong'] = (all_results['baseline_correct'] == False) & (all_results['timer_correct'] == False)
all_results['timer_loss'] = (all_results['baseline_correct'] == True) & (all_results['timer_correct'] == False)

# 5a: Pie chart of outcomes
ax = axes[0]
outcome_counts = [
    all_results['both_correct'].sum(),
    all_results['timer_win'].sum(),
    all_results['both_wrong'].sum(),
    all_results['timer_loss'].sum()
]
labels = ['Both Correct', 'TIMER Wins', 'Both Wrong', 'TIMER Loses']
colors = ['#86EFAC', '#22C55E', '#FCA5A5', '#EF4444']
explode = (0, 0.1, 0, 0)  # Explode TIMER wins

ax.pie(outcome_counts, labels=labels, autopct='%1.0f%%', colors=colors,
      explode=explode, startangle=90, textprops={'fontsize': 10})
ax.set_title('Query Outcome Distribution', fontsize=13, fontweight='bold')

# 5b: TIMER wins by scenario
ax = axes[1]
wins_by_scenario = all_results[all_results['timer_win']].groupby('scenario').size()
total_by_scenario = all_results.groupby('scenario').size()
win_rate = (wins_by_scenario / total_by_scenario * 100).fillna(0)

bars = ax.bar(range(len(win_rate)), win_rate.values, color=COLORS['success'], edgecolor='black')
ax.set_xticks(range(len(win_rate)))
ax.set_xticklabels([s[:15] for s in win_rate.index], rotation=15, ha='right')
ax.set_ylabel('TIMER Win Rate (%)', fontsize=12)
ax.set_title('TIMER Improvement Rate by Scenario', fontsize=13, fontweight='bold')

for bar, val in zip(bars, win_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
           f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# 5c: TIMER wins by intent
ax = axes[2]
wins_by_intent = all_results[all_results['timer_win']].groupby('intent').size()
total_by_intent = all_results.groupby('intent').size()
win_rate_intent = (wins_by_intent / total_by_intent * 100).fillna(0)

intent_colors = {'historical': COLORS['failure'], 'current': COLORS['success']}
bars = ax.bar(win_rate_intent.index, win_rate_intent.values, 
             color=[intent_colors.get(i, 'gray') for i in win_rate_intent.index], edgecolor='black')
ax.set_ylabel('TIMER Win Rate (%)', fontsize=12)
ax.set_title('TIMER Improvement Rate by Intent Type', fontsize=13, fontweight='bold')

for bar, val in zip(bars, win_rate_intent.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
           f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval5_improvement_analysis.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval5_improvement_analysis.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval5_improvement_analysis.png')

# === Examine failure cases ===
failures = all_results[all_results['timer_correct'] == False].copy()

print(f'=== TIMER Failure Cases ({len(failures)} queries) ===')
print()

if len(failures) > 0:
    for _, row in failures.iterrows():
        print(f"Query: {row['query_id']} ({row['scenario']})")
        print(f"  Text: {row['query_text'][:80]}..." if len(str(row['query_text'])) > 80 else f"  Text: {row['query_text']}")
        print(f"  Ground Truth Intent: {row['intent']}")
        print(f"  Detected Intent: {row['timer_intent_detected']}")
        print(f"  Expected: {row['expected_note']}")
        print(f"  Retrieved: {row['timer_retrieval']}")
        print()
else:
    print("No failure cases found! TIMER achieved 100% accuracy.")
# === Figure 6: Failure Analysis Visualization ===
fig, ax = plt.subplots(figsize=(10, 6))

if len(failures) > 0:
    # Failure by scenario
    failure_counts = failures.groupby('scenario').size()
    total_counts = all_results.groupby('scenario').size()
    failure_rate = (failure_counts / total_counts * 100).fillna(0)
    
    colors = [COLORS.get(s.replace('_', ' ').lower(), 'gray') for s in failure_rate.index]
    bars = ax.bar(failure_rate.index, failure_rate.values, color=colors, edgecolor='black')
    
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('TIMER-Graph Error Rate by Scenario', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    for bar, val, count in zip(bars, failure_rate.values, failure_counts.reindex(failure_rate.index, fill_value=0)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.0f}%\n({count})', ha='center', fontsize=10)
else:
    # Perfect accuracy message
    ax.text(0.5, 0.5, '100% Accuracy Achieved\nNo Failures to Analyze',
           ha='center', va='center', fontsize=20, fontweight='bold',
           color=COLORS['success'], transform=ax.transAxes)
    ax.set_title('TIMER-Graph Error Analysis', fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval6_failure_analysis.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval6_failure_analysis.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval6_failure_analysis.png')

# === Figure 7: Summary Statistics Table ===
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Prepare summary table
summary_display = summary[['Scenario', 'N', 'Baseline Acc', 'TIMER Acc', 'Δ Acc', 'TIMER Wins']].copy()
summary_display.columns = ['Scenario', 'N', 'Baseline', 'TIMER', 'Δ', 'Wins']

# Create table
table = ax.table(
    cellText=summary_display.values,
    colLabels=summary_display.columns,
    cellLoc='center',
    loc='center',
    colColours=['#E5E7EB'] * len(summary_display.columns)
)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Highlight header
for i in range(len(summary_display.columns)):
    table[(0, i)].set_text_props(fontweight='bold')
    table[(0, i)].set_facecolor('#1F2937')
    table[(0, i)].set_text_props(color='white')

# Highlight TIMER column
for i in range(1, len(summary_display) + 1):
    table[(i, 3)].set_facecolor('#DBEAFE')  # TIMER column
    
# Highlight overall row
if '**OVERALL**' in summary_display['Scenario'].values:
    overall_idx = summary_display[summary_display['Scenario'] == '**OVERALL**'].index[0] + 1
    for j in range(len(summary_display.columns)):
        table[(overall_idx, j)].set_facecolor('#FEF3C7')
        table[(overall_idx, j)].set_text_props(fontweight='bold')

ax.set_title('Hard Negative Stress Test Results Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval7_summary_table.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval7_summary_table.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval7_summary_table.png')

# List all generated files
print('\n=== Generated Evaluation Visualizations ===')
for f in sorted(PLOTS_DIR.glob('eval*.png')):
    print(f'  ✓ {f.name}')

