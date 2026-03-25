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
fig, ax = plt.subplots(figsize=(12, 6))

# Compile metrics across phases
phases = {
    'Phase 1\n(Naive)': {'recall5': phase1['results']['naive']['Recall@5'], 'type': 'baseline'},
    'Phase 1\n(Hybrid)': {'recall5': phase1['results']['poc']['Recall@5'], 'type': 'improvement'},
    'Phase 3\n(Cross-Encoder)': {'recall5': phase3['results']['Phase 3 (Reranking)']['Recall@5'], 'type': 'improvement'},
}

# Add Phase 5 TIMER result (overall accuracy on hard negatives)
overall_row = summary[summary['Scenario'] == '**OVERALL**']
if len(overall_row) > 0:
    timer_acc = float(overall_row['TIMER Acc'].values[0].replace('%', '')) / 100
    phases['Phase 5\n(TIMER-Graph)'] = {'recall5': timer_acc, 'type': 'novel'}

phase_names = list(phases.keys())
recall_values = [phases[p]['recall5'] for p in phase_names]
phase_types = [phases[p]['type'] for p in phase_names]

# Color by type
color_map = {'baseline': '#E5E7EB', 'improvement': '#93C5FD', 'novel': '#22C55E'}
bar_colors = [color_map[t] for t in phase_types]

# Create bars
bars = ax.bar(phase_names, recall_values, color=bar_colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, recall_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
           f'{val:.0%}', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Add improvement arrows
for i in range(1, len(recall_values)):
    delta = recall_values[i] - recall_values[i-1]
    if delta > 0:
        mid_x = i - 0.5
        mid_y = (recall_values[i] + recall_values[i-1]) / 2
        ax.annotate(f'+{delta:.0%}', xy=(mid_x, mid_y), fontsize=10, 
                   color=COLORS['success'], fontweight='bold', ha='center')

# Styling
ax.set_ylabel('Accuracy (Recall@5 / Hard Neg Accuracy)', fontsize=12)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
ax.set_title('Research Journey: From Naive RAG to TIMER-Graph', fontsize=16, fontweight='bold')

# Legend for phase types
legend_elements = [
    mpatches.Patch(facecolor='#E5E7EB', edgecolor='black', label='Baseline'),
    mpatches.Patch(facecolor='#93C5FD', edgecolor='black', label='Incremental Improvement'),
    mpatches.Patch(facecolor='#22C55E', edgecolor='black', label='Novel Contribution'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Add annotation box
ax.text(0.98, 0.05, f'Final Improvement:\n+{(timer_acc - phase1["results"]["naive"]["Recall@5"]):.0%} over naive',
       transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval1_phase_progression.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval1_phase_progression.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval1_phase_progression.png')
plt.show()
# === Figure 2: Grouped Bar Chart - Baseline vs TIMER ===
fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data - exclude overall
df_scenarios = summary[summary['Scenario'] != '**OVERALL**'].copy()
df_scenarios['Baseline'] = df_scenarios['Baseline Acc'].str.replace('%', '').astype(float)
df_scenarios['TIMER'] = df_scenarios['TIMER Acc'].str.replace('%', '').astype(float)

# Plot settings
x = np.arange(len(df_scenarios))
width = 0.35

# Create grouped bars
bars1 = ax.bar(x - width/2, df_scenarios['Baseline'], width, 
               label='Semantic Baseline', color=COLORS['baseline'], edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, df_scenarios['TIMER'], width,
               label='TIMER-Graph', color=COLORS['timer'], edgecolor='black', linewidth=1.2)

# Add value labels
def add_labels(bars, fontweight='normal'):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight=fontweight)

add_labels(bars1)
add_labels(bars2, fontweight='bold')

# Add improvement annotations
for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    improvement = b2.get_height() - b1.get_height()
    if improvement > 0:
        ax.annotate(f'+{improvement:.0f}%', 
                   xy=(x[i], max(b1.get_height(), b2.get_height()) + 10),
                   fontsize=11, color=COLORS['success'], fontweight='bold', ha='center')

# Styling
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Scenario Type', fontsize=12)
ax.set_title('Hard Negative Stress Test: TIMER-Graph vs Semantic Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_scenarios['Scenario'], rotation=0)
ax.set_ylim(0, 115)
ax.legend(loc='upper right', fontsize=11)

# Add random baseline line
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
ax.text(len(df_scenarios)-0.5, 52, 'Random Chance (50%)', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval2_scenario_comparison.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval2_scenario_comparison.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval2_scenario_comparison.png')
plt.show()
# === Figure 3: Per-Query Win/Loss Heatmap ===
fig, ax = plt.subplots(figsize=(14, 10))

# Prepare data for heatmap
all_results['baseline_correct_int'] = all_results['baseline_correct'].astype(int)
all_results['timer_correct_int'] = all_results['timer_correct'].astype(int)

# Create matrix: rows=queries, columns=[Baseline, TIMER]
heatmap_data = all_results[['query_id', 'scenario', 'baseline_correct_int', 'timer_correct_int']].copy()
heatmap_data = heatmap_data.sort_values(['scenario', 'query_id'])

# Create display matrix
matrix = heatmap_data[['baseline_correct_int', 'timer_correct_int']].values

# Custom colormap: red for wrong, green for correct
cmap = LinearSegmentedColormap.from_list('correct', ['#FEE2E2', '#DCFCE7'])

# Plot
im = ax.imshow(matrix, cmap=cmap, aspect='auto')

# Labels
ax.set_xticks([0, 1])
ax.set_xticklabels(['Semantic\nBaseline', 'TIMER-Graph'], fontsize=12, fontweight='bold')
ax.set_yticks(range(len(heatmap_data)))
ax.set_yticklabels([f"{row['query_id']} ({row['scenario'][:10]})" for _, row in heatmap_data.iterrows()], fontsize=8)

# Add cell text
for i in range(len(matrix)):
    for j in range(2):
        text = '✓' if matrix[i, j] == 1 else '✗'
        color = COLORS['success'] if matrix[i, j] == 1 else COLORS['failure']
        ax.text(j, i, text, ha='center', va='center', fontsize=10, color=color, fontweight='bold')

# Highlight TIMER wins (improvement cases)
for i in range(len(matrix)):
    if matrix[i, 0] == 0 and matrix[i, 1] == 1:  # Baseline wrong, TIMER correct
        ax.add_patch(plt.Rectangle((0.5, i-0.5), 1, 1, fill=False, edgecolor=COLORS['success'], linewidth=2))

ax.set_title('Per-Query Correctness: Baseline vs TIMER-Graph\n(Green highlight = TIMER improvement)', 
            fontsize=14, fontweight='bold')

# Add summary stats
baseline_acc = matrix[:, 0].mean() * 100
timer_acc = matrix[:, 1].mean() * 100
improvements = ((matrix[:, 0] == 0) & (matrix[:, 1] == 1)).sum()

stats_text = f'Baseline: {baseline_acc:.0f}%\nTIMER: {timer_acc:.0f}%\nImprovements: {improvements}'
ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'),
       verticalalignment='center')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eval3_query_heatmap.png', bbox_inches='tight', facecolor='white')
plt.savefig(PLOTS_DIR / 'eval3_query_heatmap.pdf', bbox_inches='tight')
print(f'✓ Saved: {PLOTS_DIR}/eval3_query_heatmap.png')
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
# === Figure 7: Summary Statistics Table ===
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Prepare summary table
summary_display = summary.copy()
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
plt.show()
# List all generated files
print('\n=== Generated Evaluation Visualizations ===')
for f in sorted(PLOTS_DIR.glob('eval*.png')):
    print(f'  ✓ {f.name}')

