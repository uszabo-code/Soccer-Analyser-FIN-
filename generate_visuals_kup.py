#!/usr/bin/env python3
"""Generate comprehensive visualizations for Video 1 (output_kup)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

OUT = "output_kup"

# Load data
with open(f"{OUT}/player_stats.json") as f:
    stats = json.load(f)
with open(f"{OUT}/advanced_stats.json") as f:
    adv = json.load(f)

player = stats["players"][0]  # Target player
heatmap = np.array(player["heatmap"])

# ── Color palette ──
BG = '#0e1117'
CARD_BG = '#1a1d23'
ACCENT = '#ff6b35'
ACCENT2 = '#00d4aa'
TEXT = '#e0e0e0'
MUTED = '#888888'
FIELD_GREEN = '#2d5a27'
FIELD_LINE = '#ffffff'

def draw_field(ax, flip=False):
    """Draw a soccer field on the given axes."""
    ax.set_facecolor(FIELD_GREEN)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Field outline
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=FIELD_LINE, lw=1.5, alpha=0.6)
    # Center line
    ax.axvline(0.5, color=FIELD_LINE, lw=1, alpha=0.4)
    # Center circle
    circle = Circle((0.5, 0.5), 0.08, fill=False, color=FIELD_LINE, lw=1, alpha=0.4)
    ax.add_patch(circle)
    # Penalty areas
    for x_start in [0, 0.83]:
        w = 0.17
        ax.plot([x_start, x_start+w, x_start+w, x_start], [0.2, 0.2, 0.8, 0.8],
                color=FIELD_LINE, lw=1, alpha=0.4)
    # Third lines
    for x in [1/3, 2/3]:
        ax.axvline(x, color=FIELD_LINE, lw=0.5, alpha=0.2, ls='--')
    ax.set_xticks([])
    ax.set_yticks([])

# ═══════════════════════════════════════════════════════
# FIGURE 1: Player Dashboard (4-panel overview)
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle(f"Player #{15} — Game Analysis Dashboard",
             fontsize=22, fontweight='bold', color=TEXT, y=0.97)
fig.text(0.5, 0.935, f"Position: {player['inferred_position'].title()} | Distance: {player['total_distance_m']:.0f}m | Max Speed: {player['max_speed_mps']:.1f} m/s | Sprints: {player['sprint_count']}",
         fontsize=12, color=MUTED, ha='center')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25,
                       left=0.06, right=0.94, top=0.9, bottom=0.05)

# ── Panel 1: Heatmap ──
ax1 = fig.add_subplot(gs[0, 0])
draw_field(ax1)
# Overlay heatmap
if heatmap.max() > 0:
    hm_norm = heatmap / heatmap.max()
    rows, cols = hm_norm.shape
    cmap = LinearSegmentedColormap.from_list('heat', ['#2d5a2700', '#ffff00', '#ff6600', '#ff0000'])
    extent = [0, 1, 0, 1]
    ax1.imshow(hm_norm, extent=extent, origin='lower', cmap=cmap, alpha=0.7,
               aspect='auto', interpolation='gaussian')
ax1.set_title("Position Heatmap", fontsize=14, fontweight='bold', color=TEXT, pad=10)
# Third labels
for x, label in [(1/6, 'DEF'), (0.5, 'MID'), (5/6, 'ATT')]:
    ax1.text(x, 0.02, label, ha='center', fontsize=8, color=FIELD_LINE, alpha=0.5)

# ── Panel 2: Work Rate Phases ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD_BG)
phases = adv["work_rate_phases"]
labels = ['Idle\n(<1 m/s)', 'Jogging\n(1-3 m/s)', 'Running\n(3-5 m/s)', 'Sprinting\n(>5 m/s)']
values = [phases['idle'], phases['jogging'], phases['running'], phases['sprinting']]
colors = ['#555555', '#4ecdc4', '#ff9f43', '#ee5a24']
bars = ax2.bar(range(4), values, color=colors, width=0.6, edgecolor='none')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold', color=TEXT)
ax2.set_xticks(range(4))
ax2.set_xticklabels(labels, fontsize=9, color=MUTED)
ax2.set_ylabel('% of Time', fontsize=11, color=MUTED)
ax2.set_ylim(0, max(values) * 1.2)
ax2.set_title("Work Rate Breakdown", fontsize=14, fontweight='bold', color=TEXT, pad=10)
ax2.tick_params(colors=MUTED)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(MUTED)
ax2.spines['bottom'].set_color(MUTED)

# ── Panel 3: Movement Direction ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD_BG)
lat = adv["movement_direction"]["lateral_pct"]
vert = adv["movement_direction"]["vertical_pct"]
# Arrow-based visualization
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
# Lateral arrows (left-right)
lat_size = lat / 100
ax3.annotate('', xy=(-lat_size, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=ACCENT, lw=3*lat_size))
ax3.annotate('', xy=(lat_size, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=ACCENT, lw=3*lat_size))
# Vertical arrows (up-down = toward goal)
vert_size = vert / 100
ax3.annotate('', xy=(0, vert_size*2), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=3*max(vert_size, 0.3)))
ax3.annotate('', xy=(0, -vert_size*2), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=3*max(vert_size, 0.3)))
ax3.text(0, -1.3, f'Lateral: {lat:.0f}%', ha='center', fontsize=14, fontweight='bold', color=ACCENT)
ax3.text(0, 1.3, f'Vertical: {vert:.0f}%', ha='center', fontsize=14, fontweight='bold', color=ACCENT2)
ax3.text(0, -1.0, '← Side to side →', ha='center', fontsize=9, color=MUTED)
ax3.text(0, 1.05, '↑ Toward goal ↑', ha='center', fontsize=9, color=MUTED)
ax3.set_title("Movement Direction", fontsize=14, fontweight='bold', color=TEXT, pad=10)
ax3.set_xticks([])
ax3.set_yticks([])
for spine in ax3.spines.values():
    spine.set_visible(False)

# ── Panel 4: Key Metrics Summary ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_BG)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

metrics = [
    ("Movement\nPurposefulness", adv["movement_purposefulness"]["score"], 1.0, ACCENT2,
     adv["movement_purposefulness"]["interpretation"]),
    ("Nearest\nTeammate", adv["spacing"]["avg_nearest_teammate_m"], 10.0, ACCENT,
     f'{adv["spacing"]["avg_nearest_teammate_m"]:.1f}m — {adv["spacing"]["interpretation"]}'),
    ("Positional\nConsistency", adv["positional_discipline"]["consistency_score"], 1.0, '#9b59b6',
     f'X-var: {adv["positional_discipline"]["x_variability_m"]:.1f}m, Y-var: {adv["positional_discipline"]["y_variability_m"]:.1f}m'),
    ("Field Position\n(Att Third %)", player["time_in_thirds"]["attacking"], 100, '#3498db',
     f'{player["time_in_thirds"]["attacking"]:.0f}% attacking, {player["time_in_thirds"]["middle"]:.0f}% middle'),
]

for i, (label, value, max_val, color, desc) in enumerate(metrics):
    y = 8.5 - i * 2.5
    # Label
    ax4.text(0.3, y, label, fontsize=10, fontweight='bold', color=TEXT, va='center')
    # Bar background
    bar_x, bar_y, bar_w, bar_h = 3.5, y-0.3, 6, 0.6
    ax4.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w, bar_h,
                  boxstyle="round,pad=0.1", facecolor='#333', edgecolor='none'))
    # Bar fill
    fill_w = bar_w * min(value / max_val, 1.0)
    ax4.add_patch(FancyBboxPatch((bar_x, bar_y), fill_w, bar_h,
                  boxstyle="round,pad=0.1", facecolor=color, edgecolor='none', alpha=0.8))
    # Value text
    ax4.text(bar_x + bar_w + 0.1, y, f'{value:.1f}' if max_val <= 1 else f'{value:.1f}',
             fontsize=10, fontweight='bold', color=color, va='center')
    # Description
    ax4.text(3.5, y-0.7, desc, fontsize=7, color=MUTED, va='center')

ax4.set_title("Key Metrics", fontsize=14, fontweight='bold', color=TEXT, pad=10)
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_visible(False)

plt.savefig(f'{OUT}/dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Dashboard saved")

# ═══════════════════════════════════════════════════════
# FIGURE 2: Field Position + Movement Trail
# ═══════════════════════════════════════════════════════
fig2, (ax_field, ax_speed) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig2.suptitle(f"Player #{15} — Position & Speed Profile",
              fontsize=18, fontweight='bold', color=TEXT, y=0.97)

# Movement trail on field
draw_field(ax_field)
# Extract positions from detections
with open(f"{OUT}/detections.json") as f:
    det = json.load(f)
target_track = player["player_id"].split(" ")[0].replace("Track-", "")
positions = []
for d in det["person_detections"]:
    if str(d["track_id"]) == target_track:
        cx = (d["bbox"][0] + d["bbox"][2]) / 2 / stats["frame_width"]
        cy = (d["bbox"][1] + d["bbox"][3]) / 2 / stats["frame_height"]
        positions.append((cx, 1-cy, d["frame_num"]))  # flip y

if positions:
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    frame_nums = [p[2] for p in positions]
    # Color by time
    colors_t = np.linspace(0, 1, len(xs))
    scatter = ax_field.scatter(xs, ys, c=colors_t, cmap='plasma', s=3, alpha=0.6, zorder=5)
    # Start and end markers
    ax_field.plot(xs[0], ys[0], 'o', color='#00ff00', markersize=10, zorder=6, label='Start')
    ax_field.plot(xs[-1], ys[-1], 's', color='#ff0000', markersize=10, zorder=6, label='End')
    ax_field.legend(loc='lower right', fontsize=8, facecolor=CARD_BG, edgecolor=MUTED,
                    labelcolor=TEXT)
ax_field.set_title("Movement Trail (color = time progression)", fontsize=12,
                   fontweight='bold', color=TEXT, pad=10)

# Speed over time
ax_speed.set_facecolor(CARD_BG)
# Compute speeds from positions
if len(positions) > 1:
    speeds = []
    times = []
    fps = stats["fps"]
    fw = stats["field_width_m"]
    fh = fw * stats["frame_height"] / stats["frame_width"]
    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i-1][0]) * fw
        dy = (positions[i][1] - positions[i-1][1]) * fh
        dt = (positions[i][2] - positions[i-1][2]) / fps
        if dt > 0:
            speed = np.sqrt(dx**2 + dy**2) / dt
            speeds.append(min(speed, 12.0))
            times.append(positions[i][2] / fps)

    # Smooth speeds
    window = 10
    smoothed = np.convolve(speeds, np.ones(window)/window, mode='valid')
    t_smooth = times[window-1:]

    ax_speed.fill_between(t_smooth, smoothed, alpha=0.3, color=ACCENT)
    ax_speed.plot(t_smooth, smoothed, color=ACCENT, lw=1.5)

    # Sprint threshold line
    ax_speed.axhline(5.0, color='#ee5a24', ls='--', lw=1, alpha=0.6, label='Sprint threshold')
    ax_speed.axhline(3.0, color='#4ecdc4', ls='--', lw=1, alpha=0.4, label='Run threshold')

    # Mark sprint episodes
    for ep in player.get("sprint_episodes", []):
        t_start = ep["start_frame"] / fps
        t_end = ep["end_frame"] / fps
        ax_speed.axvspan(t_start, t_end, color='#ee5a24', alpha=0.2)

    ax_speed.set_xlabel("Time (seconds)", fontsize=10, color=MUTED)
    ax_speed.set_ylabel("Speed (m/s)", fontsize=10, color=MUTED)
    ax_speed.legend(fontsize=8, facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)

ax_speed.set_title("Speed Profile Over Time", fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_speed.tick_params(colors=MUTED)
ax_speed.spines['top'].set_visible(False)
ax_speed.spines['right'].set_visible(False)
ax_speed.spines['left'].set_color(MUTED)
ax_speed.spines['bottom'].set_color(MUTED)

plt.savefig(f'{OUT}/position_speed.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Position & Speed saved")

# ═══════════════════════════════════════════════════════
# FIGURE 3: Spacing Analysis
# ═══════════════════════════════════════════════════════
fig3, ax_sp = plt.subplots(figsize=(10, 6), facecolor=BG)
ax_sp.set_facecolor(CARD_BG)

spacing = adv["spacing"]
categories = ['Nearest\nTeammate', 'Distance from\nTeam Center', 'Team\nSpread']
values_sp = [spacing["avg_nearest_teammate_m"], spacing["avg_distance_from_centroid_m"], spacing["avg_team_spread_m"]]
ideal_ranges = [(5, 10), (10, 15), (30, 40)]  # rough ideal ranges for U13-15
colors_sp = []
for val, (lo, hi) in zip(values_sp, ideal_ranges):
    if lo <= val <= hi:
        colors_sp.append(ACCENT2)
    else:
        colors_sp.append(ACCENT)

bars = ax_sp.bar(range(3), values_sp, color=colors_sp, width=0.5, edgecolor='none')
for bar, val in zip(bars, values_sp):
    ax_sp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}m', ha='center', fontsize=14, fontweight='bold', color=TEXT)

# Show ideal ranges
for i, (lo, hi) in enumerate(ideal_ranges):
    ax_sp.plot([i-0.25, i+0.25], [lo, lo], '--', color=ACCENT2, alpha=0.5, lw=1)
    ax_sp.plot([i-0.25, i+0.25], [hi, hi], '--', color=ACCENT2, alpha=0.5, lw=1)
    ax_sp.text(i+0.3, (lo+hi)/2, f'Ideal\n{lo}-{hi}m', fontsize=7, color=MUTED, va='center')

ax_sp.set_xticks(range(3))
ax_sp.set_xticklabels(categories, fontsize=11, color=MUTED)
ax_sp.set_ylabel('Distance (meters)', fontsize=11, color=MUTED)
ax_sp.set_title(f"Player #{15} — Spacing Analysis", fontsize=16, fontweight='bold', color=TEXT, pad=15)
ax_sp.tick_params(colors=MUTED)
ax_sp.spines['top'].set_visible(False)
ax_sp.spines['right'].set_visible(False)
ax_sp.spines['left'].set_color(MUTED)
ax_sp.spines['bottom'].set_color(MUTED)

plt.savefig(f'{OUT}/spacing.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Spacing saved")

print("\n🎉 All visualizations generated in output_kup/")
