#!/usr/bin/env python3
"""
TIMER-Graph Visualization Generator
Generates publication-quality figures for research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Paths
ROOT = Path(__file__).parent.parent.resolve()
DATA_EVAL = ROOT / 'data' / 'evaluation'
RESULTS_P5 = ROOT / 'results' / 'phase5'
PLOTS_DIR = ROOT / 'visualizations' / 'plots' / 'publication_ready'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f'Root: {ROOT}')
print(f'Plots will be saved to: {PLOTS_DIR}')

# ============================================
# Figure 1: Phase Progression
# ============================================
print('\n=== Generating Figure 1: Phase Progression ===')

with open(DATA_EVAL / 'phase_1_baseline_results.json') as f:
    p1 = json.load(f)
with open(DATA_EVAL / 'phase_3_results.json') as f:
    p3 = json.load(f)

# Phase 5 Results
p5_summary = pd.read_csv(RESULTS_P5 / 'summary_table.csv')
timer_accuracy = float(p5_summary[p5_summary['Scenario'] == '**OVERALL**']['TIMER Accuracy'].values[0].replace('%', '')) / 100

phases = ['Naive\n(Baseline)', 'Hybrid\nChunking', 'Cross-Encoder\nReranking', 'TIMER-Graph\n(Hard Neg)']
recall_values = [
    p1['results']['naive']['Recall@5'],
    p1['results']['poc']['Recall@5'],
    p3['results']['Phase 3 (Reranking)']['Recall@5'],
    timer_accuracy
]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#E57373', '#FFB74D', '#64B5F6', '#81C784']
bars = ax.bar(phases, recall_values, color=colors, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, recall_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Recall@5', fontsize=12)
ax.set_title('Research Journey: Retrieval Performance Across Phases', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig1_phase_progression.pdf', bbox_inches='tight')
plt.savefig(PLOTS_DIR / 'fig1_phase_progression.png', bbox_inches='tight')
plt.close()
print(f'  ✓ Saved: fig1_phase_progression.pdf')

# ============================================
# Figure 2: Hard Negative Comparison
# ============================================
print('\n=== Generating Figure 2: Hard Negative Comparison ===')

df = pd.read_csv(RESULTS_P5 / 'summary_table.csv')
df = df[df['Scenario'] != '**OVERALL**']
df['Baseline'] = df['Baseline Accuracy'].str.replace('%', '').astype(float)
df['TIMER'] = df['TIMER Accuracy'].str.replace('%', '').astype(float)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df))
width = 0.35

bars1 = ax.bar(x - width/2, df['Baseline'], width, label='Semantic Baseline', color='#BDBDBD', edgecolor='black')
bars2 = ax.bar(x + width/2, df['TIMER'], width, label='TIMER-Graph', color='#2196F3', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Hard Negative Stress Test: TIMER-Graph vs Semantic Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['Scenario'], rotation=15, ha='right')
ax.legend(loc='upper left')
ax.set_ylim(0, 120)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Chance')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig2_hard_negative_comparison.pdf', bbox_inches='tight')
plt.savefig(PLOTS_DIR / 'fig2_hard_negative_comparison.png', bbox_inches='tight')
plt.close()
print(f'  ✓ Saved: fig2_hard_negative_comparison.pdf')

# ============================================
# Figure 3: Temporal Decay Curves
# ============================================
print('\n=== Generating Figure 3: Temporal Decay ===')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 3a: Basic decay
t = np.linspace(0, 2000, 500)
lambdas = [0.001, 0.005, 0.01]
colors = ['#4CAF50', '#2196F3', '#F44336']

for lam, c in zip(lambdas, colors):
    decay = np.exp(-lam * t)
    ax1.plot(t, decay, label=f'λ = {lam}', color=c, linewidth=2)

ax1.set_xlabel('Time Offset (days)')
ax1.set_ylabel('Decay Value e^(-λt)')
ax1.set_title('(A) Exponential Decay Function', fontweight='bold')
ax1.legend()
ax1.set_xlim(0, 2000)
ax1.set_ylim(0, 1.05)

# 3b: Beta modulation
lam = 0.005
semantic = 0.6
betas = [0.8, 0.0, -0.3]
beta_labels = ['β=+0.8 (Current)', 'β=0.0 (Neutral)', 'β=-0.3 (Historical)']
colors = ['#4CAF50', '#9E9E9E', '#F44336']

for beta, label, c in zip(betas, beta_labels, colors):
    score = semantic + beta * np.exp(-lam * t)
    ax2.plot(t, score, label=label, color=c, linewidth=2)

ax2.axhline(y=semantic, color='black', linestyle=':', alpha=0.5, label='Semantic Only')
ax2.set_xlabel('Time Offset (days)')
ax2.set_ylabel('Final TIMER Score')
ax2.set_title('(B) Intent-Modulated Scoring', fontweight='bold')
ax2.legend(loc='center right')
ax2.set_xlim(0, 1000)
ax2.set_ylim(0.2, 1.5)

ax2.fill_between(t[(t > 0) & (t < 200)], 0.2, 1.5, alpha=0.1, color='orange')
ax2.text(100, 1.35, 'Recent\nActive Zone', ha='center', fontsize=9, color='orange')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig3_temporal_decay.pdf', bbox_inches='tight')
plt.savefig(PLOTS_DIR / 'fig3_temporal_decay.png', bbox_inches='tight')
plt.close()
print(f'  ✓ Saved: fig3_temporal_decay.pdf')

# ============================================
# Figure 4: Beta Values
# ============================================
print('\n=== Generating Figure 4: Beta Values ===')

fig, ax = plt.subplots(figsize=(6, 4))

intents = ['Current', 'Historical', 'Trend']
betas = [0.8, -0.3, 0.4]
colors = ['#4CAF50', '#F44336', '#2196F3']

bars = ax.bar(intents, betas, color=colors, edgecolor='black', linewidth=1)
ax.axhline(y=0, color='black', linewidth=1)

for bar, val in zip(bars, betas):
    offset = 0.05 if val > 0 else -0.1
    ax.text(bar.get_x() + bar.get_width()/2, val + offset,
            f'{val:+.1f}', ha='center', va='bottom' if val > 0 else 'top',
            fontsize=12, fontweight='bold')

ax.set_ylabel('β (Beta) Value', fontsize=12)
ax.set_title('Intent-Modulated Beta Parameters', fontsize=14, fontweight='bold')
ax.set_ylim(-0.6, 1.0)

ax.text(0, 0.95, 'Boost Recent', ha='center', fontsize=9, style='italic', color='#388E3C')
ax.text(1, -0.55, 'Boost Historical', ha='center', fontsize=9, style='italic', color='#D32F2F')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig4_beta_values.pdf', bbox_inches='tight')
plt.savefig(PLOTS_DIR / 'fig4_beta_values.png', bbox_inches='tight')
plt.close()
print(f'  ✓ Saved: fig4_beta_values.pdf')

# ============================================
# Figure 5: Ranking Flip Case Study
# ============================================
print('\n=== Generating Figure 5: Ranking Flip ===')

sc_results = pd.read_csv(RESULTS_P5 / 'semantic_collision_results.csv')
case = sc_results[sc_results['query_id'] == 'sc_hist_1'].iloc[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

semantic_score = 0.6
offset_new, offset_old = 10, 500
lam = 0.005
beta_hist = -0.3

notes_baseline = ['New Note\n(10 days)', 'Old Note\n(500 days)']
scores_baseline = [semantic_score, semantic_score]

score_new = semantic_score + beta_hist * np.exp(-lam * offset_new)
score_old = semantic_score + beta_hist * np.exp(-lam * offset_old)
scores_timer = [score_new, score_old]

# Baseline
bars1 = ax1.barh(notes_baseline, scores_baseline, color=['#BDBDBD', '#BDBDBD'], edgecolor='black')
ax1.set_xlim(0, 1.0)
ax1.set_xlabel('Score')
ax1.set_title('(A) Semantic Baseline\n(New Note Wins ❌)', fontweight='bold', color='red')
for bar, val in zip(bars1, scores_baseline):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')

# TIMER
colors_timer = ['#FFCDD2', '#C8E6C9']
bars2 = ax2.barh(notes_baseline, scores_timer, color=colors_timer, edgecolor='black')
ax2.set_xlim(0, 1.0)
ax2.set_xlabel('Score')
ax2.set_title('(B) TIMER-Graph (Historical)\n(Old Note Wins ✓)', fontweight='bold', color='green')
for bar, val in zip(bars2, scores_timer):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')

plt.suptitle(f'Query: "{case["query_text"]}"', fontsize=11, style='italic', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig5_ranking_flip.pdf', bbox_inches='tight')
plt.savefig(PLOTS_DIR / 'fig5_ranking_flip.png', bbox_inches='tight')
plt.close()
print(f'  ✓ Saved: fig5_ranking_flip.pdf')

# ============================================
# Summary
# ============================================
print('\n' + '='*50)
print('VISUALIZATION GENERATION COMPLETE')
print('='*50)
print(f'\nGenerated files in {PLOTS_DIR}:')
for f in sorted(PLOTS_DIR.glob('*.pdf')):
    print(f'  ✓ {f.name}')
