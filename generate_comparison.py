#!/usr/bin/env python3
"""Generate side-by-side comparison of Video 1 vs Video 2."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BG = '#0e1117'
CARD_BG = '#1a1d23'
ACCENT = '#ff6b35'
ACCENT2 = '#00d4aa'
TEXT = '#e0e0e0'
MUTED = '#888888'

# Load data
games = {}
for label, folder in [("Game 1\n(Screen Recording)", "output_v3"), ("Game 2\n(KUP 1H)", "output_kup")]:
    with open(f"{folder}/player_stats.json") as f:
        stats = json.load(f)
    with open(f"{folder}/advanced_stats.json") as f:
        adv = json.load(f)
    games[label] = {"stats": stats, "adv": adv, "player": stats["players"][0]}

fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle("Player #15 — Game Comparison", fontsize=24, fontweight='bold', color=TEXT, y=0.97)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                       left=0.08, right=0.92, top=0.91, bottom=0.05)

labels = list(games.keys())
colors_g = [ACCENT, ACCENT2]

# ── Row 1: Work Rate Comparison ──
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(CARD_BG)
phases = ['idle', 'jogging', 'running', 'sprinting']
phase_labels = ['Idle\n(<1 m/s)', 'Jogging\n(1-3 m/s)', 'Running\n(3-5 m/s)', 'Sprinting\n(>5 m/s)']
x = np.arange(4)
width = 0.3

for i, (label, data) in enumerate(games.items()):
    vals = [data["adv"]["work_rate_phases"][p] for p in phases]
    bars = ax1.bar(x + i*width - width/2, vals, width, label=label.replace('\n', ' '),
                   color=colors_g[i], alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold', color=colors_g[i])

ax1.set_xticks(x)
ax1.set_xticklabels(phase_labels, fontsize=10, color=MUTED)
ax1.set_ylabel('% of Time', fontsize=11, color=MUTED)
ax1.set_title("Work Rate Comparison", fontsize=16, fontweight='bold', color=TEXT, pad=15)
ax1.legend(fontsize=10, facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
ax1.tick_params(colors=MUTED)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color(MUTED)
ax1.spines['bottom'].set_color(MUTED)

# ── Row 2 Left: Field Position Distribution ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(CARD_BG)
thirds = ['defensive', 'middle', 'attacking']
third_labels = ['Defensive', 'Middle', 'Attacking']
x2 = np.arange(3)
for i, (label, data) in enumerate(games.items()):
    vals = [data["player"]["time_in_thirds"][t] for t in thirds]
    ax2.bar(x2 + i*width - width/2, vals, width, label=label.replace('\n', ' '),
            color=colors_g[i], alpha=0.85, edgecolor='none')
    for j, val in enumerate(vals):
        if val > 0:
            ax2.text(x2[j] + i*width - width/2, val + 1, f'{val:.0f}%',
                     ha='center', fontsize=9, fontweight='bold', color=colors_g[i])

ax2.set_xticks(x2)
ax2.set_xticklabels(third_labels, fontsize=10, color=MUTED)
ax2.set_ylabel('% of Time', fontsize=11, color=MUTED)
ax2.set_title("Field Position Distribution", fontsize=14, fontweight='bold', color=TEXT, pad=10)
ax2.tick_params(colors=MUTED)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(MUTED)
ax2.spines['bottom'].set_color(MUTED)

# ── Row 2 Right: Movement Direction ──
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(CARD_BG)
categories = ['Lateral %', 'Vertical %']
x3 = np.arange(2)
for i, (label, data) in enumerate(games.items()):
    lat = data["adv"]["movement_direction"]["lateral_pct"]
    vert = data["adv"]["movement_direction"]["vertical_pct"]
    vals = [lat, vert]
    bars = ax3.bar(x3 + i*width - width/2, vals, width, label=label.replace('\n', ' '),
                   color=colors_g[i], alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', fontsize=11, fontweight='bold', color=colors_g[i])

ax3.set_xticks(x3)
ax3.set_xticklabels(categories, fontsize=11, color=MUTED)
ax3.set_ylabel('% of Movement', fontsize=11, color=MUTED)
ax3.set_title("Movement Direction", fontsize=14, fontweight='bold', color=TEXT, pad=10)
ax3.tick_params(colors=MUTED)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(MUTED)
ax3.spines['bottom'].set_color(MUTED)
# Target line for vertical
ax3.axhline(30, color='#ffff00', ls='--', lw=1, alpha=0.4)
ax3.text(1.7, 31, 'Target: 30%+', fontsize=8, color='#ffff00', alpha=0.6)

# ── Row 3: Key Metrics Radar-style comparison ──
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor(CARD_BG)

metrics_list = [
    ("Purposefulness", "movement_purposefulness", "score", 1.0),
    ("Nearest Teammate (m)", "spacing", "avg_nearest_teammate_m", 15.0),
    ("Positional Consistency", "positional_discipline", "consistency_score", 1.0),
    ("Sprint %", "work_rate_phases", "sprinting", 30.0),
    ("Vertical Movement %", "movement_direction", "vertical_pct", 50.0),
]

metric_names = [m[0] for m in metrics_list]
x4 = np.arange(len(metrics_list))

for i, (label, data) in enumerate(games.items()):
    vals = []
    for name, cat, key, max_val in metrics_list:
        val = data["adv"][cat][key]
        vals.append(val)

    bars = ax4.bar(x4 + i*width - width/2, vals, width, label=label.replace('\n', ' '),
                   color=colors_g[i], alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontsize=9, fontweight='bold', color=colors_g[i])

ax4.set_xticks(x4)
ax4.set_xticklabels(metric_names, fontsize=10, color=MUTED)
ax4.set_title("Key Metrics Comparison", fontsize=14, fontweight='bold', color=TEXT, pad=10)
ax4.tick_params(colors=MUTED)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color(MUTED)
ax4.spines['bottom'].set_color(MUTED)

plt.savefig('output_v3/game_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Comparison chart saved to output_v3/game_comparison.png")
