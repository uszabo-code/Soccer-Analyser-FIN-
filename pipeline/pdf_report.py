"""Stage 5b: Generate a Veo-style PDF report with narrative-first visualizations.

Data flow:
  player_stats.json  ──┐
  analysis.json      ──┤
  identity.json      ──┼──► pdf_report.run() ──► output_dir/report.pdf
  detections.json    ──┘     (4 pages, A4, 150 DPI)

Pages:
  1 — Player Card: jersey #, metric tiles, KDE heatmap, work rate bar
  2 — Performance Summary: LLM overview + sprint timeline + coaching text
  3 — Coaching Moments: 3–5 LLM-sourced moment cards with timestamps
  4 — Game Maps: pass map + shot map (Phase 2 placeholders)

Visual language: Veo-parity — warm cream (#edeae4) bg, white rounded cards,
forest green (#2d5a3d) accent, DejaVu Sans, mplsoccer pitch with #ddd lines.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless — no display required

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ── constants ─────────────────────────────────────────────────────────────────

BG       = "#edeae4"   # warm cream page background
WHITE    = "#ffffff"   # card face
GREEN    = "#2d5a3d"   # forest green accent / data colour
GREEN_LT = "#d4e6db"   # light sage fill
GRAY     = "#d8d4cf"   # inactive bars / grid lines
GRAY_MID = "#9e9a95"   # secondary text
GRAY_DRK = "#4a4845"   # body text

PITCH_LINE   = "#cccccc"
SPRINT_COLOR = GREEN
PROX_COLOR   = "#7bbf99"

A4_W, A4_H = 8.27, 11.69   # inches at 96 DPI
DPI = 150

# Work rate speed thresholds (m/s) — aligned with config.SPRINT_SPEED_THRESHOLD=4.0
IDLE_MAX    = 0.5
JOG_MAX     = 2.5
RUN_MAX     = 4.0
# sprint = > RUN_MAX (i.e. > SPRINT_SPEED_THRESHOLD)


# ── public API ────────────────────────────────────────────────────────────────

def run(
    stats_path: str,
    identity_path: str,
    analysis_path: str,
    detections_path: str,
    output_dir: str,
) -> str:
    """Generate report.pdf alongside report.md.  Returns output path."""
    print("\n[Stage 5b] PDF Report")

    # ── load data ─────────────────────────────────────────────────────────────
    try:
        with open(stats_path) as f:
            stats_data = json.load(f)
        with open(identity_path) as f:
            id_data = json.load(f)
        with open(analysis_path) as f:
            analysis = json.load(f)
        with open(detections_path) as f:
            dets_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR: could not load required data: {exc}")
        return ""

    # ── locate target player ──────────────────────────────────────────────────
    target_stats = next(
        (p for p in stats_data["players"] if p.get("is_target")), None
    )
    if target_stats is None:
        print("  WARNING: target player not found in stats — skipping PDF")
        return ""

    jersey_num    = id_data.get("target_jersey")
    target_ids    = set(id_data.get("target_track_ids", []))
    fps           = float(dets_data.get("fps", 30.0))
    frame_width   = float(dets_data.get("width", 1920))
    frame_height  = float(dets_data.get("height", 1080))
    total_frames  = int(dets_data.get("total_frames", 0))

    # ── collect target pixel positions (for KDE) ──────────────────────────────
    print("  Extracting target player positions …")
    x_pitch, y_pitch = _target_positions_pitch(
        dets_data["person_detections"], target_ids, frame_width, frame_height
    )

    # ── work rate (per-player, derived from speed data) ────────────────────────
    work_rate = _compute_work_rate(target_stats)

    # ── render PDF ────────────────────────────────────────────────────────────
    output_path = os.path.join(output_dir, "report.pdf")
    tmp_path    = output_path + ".tmp"

    try:
        with PdfPages(tmp_path) as pdf:
            _page1_player_card(pdf, target_stats, analysis, jersey_num, x_pitch, y_pitch, work_rate)
            _page2_performance(pdf, target_stats, analysis, fps, total_frames)
            _page3_coaching_moments(pdf, analysis)
            _page4_game_maps(pdf)

        os.replace(tmp_path, output_path)   # atomic rename — no partial files visible
        print(f"  PDF saved to {output_path}")
        return output_path

    except Exception as exc:
        # Clean up any partial file
        for p in (tmp_path, output_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        print(f"  ERROR generating PDF: {exc}")
        return ""


# ── coordinate helpers ────────────────────────────────────────────────────────

def _target_positions_pitch(person_detections, target_ids, frame_w, frame_h):
    """Return (x_pitch, y_pitch) numpy arrays in mplsoccer pitch coordinates.

    Pixel space: x ∈ [0, frame_w], y ∈ [0, frame_h] (standard broadcast, goal line horizontal).
    Pitch space: x ∈ [0, 105] m (length), y ∈ [0, 68] m (width).

    Camera assumes landscape view — pitch_length maps to frame_width (horizontal axis),
    pitch_width maps to frame_height (vertical axis).
    """
    PITCH_L, PITCH_W = 105.0, 68.0
    xs, ys = [], []
    for det in person_detections:
        if det["track_id"] in target_ids:
            b = det["bbox"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            xs.append(cx / frame_w * PITCH_L)
            ys.append(cy / frame_h * PITCH_W)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _compute_work_rate(target_stats):
    """Derive per-player work-rate breakdown from available speed stats.

    Uses sprint_count, total_time_visible_s, avg_speed_mps, and max_speed_mps
    to estimate the fraction of time in each zone.  This is an approximation
    until TODO-004 (per-frame speed bins in features.py) is implemented.
    """
    total_s   = target_stats.get("total_time_visible_s", 0) or 1
    avg_mps   = target_stats.get("avg_speed_mps", 0)
    sprint_ct = target_stats.get("sprint_count", 0)

    # Approximate sprint fraction: each sprint ~2s average duration at 5+ m/s
    sprint_s = sprint_ct * 2.0
    sprint_pct = min(sprint_s / total_s * 100, 35)

    # Allocate remaining time proportional to avg speed
    remaining = 100 - sprint_pct
    if avg_mps >= RUN_MAX:
        run_pct  = min(remaining * 0.40, 30)
        jog_pct  = remaining * 0.40
        idle_pct = remaining - run_pct - jog_pct
    elif avg_mps >= JOG_MAX:
        run_pct  = remaining * 0.25
        jog_pct  = remaining * 0.45
        idle_pct = remaining - run_pct - jog_pct
    else:
        run_pct  = remaining * 0.10
        jog_pct  = remaining * 0.35
        idle_pct = remaining - run_pct - jog_pct

    return {
        "idle":     max(idle_pct, 0),
        "jogging":  max(jog_pct, 0),
        "running":  max(run_pct, 0),
        "sprinting": sprint_pct,
    }


# ── figure / style helpers ────────────────────────────────────────────────────

def _new_page(title=None):
    """Create a new A4 figure with cream background and optional page title."""
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "text.color":     GRAY_DRK,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
    })
    fig = plt.figure(figsize=(A4_W, A4_H))
    fig.patch.set_facecolor(BG)
    if title:
        fig.text(
            0.5, 0.975, title,
            ha="center", va="top",
            fontsize=8, color=GRAY_MID,
            fontweight="normal",
        )
    return fig


def _card_ax(fig, rect, title=None):
    """Add a white rounded-card axes at rect=[left, bottom, width, height] (figure fractions)."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(WHITE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White rounded rectangle as card background
    fancy = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor=WHITE,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(fancy)
    if title:
        ax.text(
            0.04, 0.92, title,
            transform=ax.transAxes,
            fontsize=8, fontweight="bold",
            color=GRAY_MID, va="top",
        )
    return ax


def _metric_tile(fig, rect, label, value, unit=""):
    """Draw a single metric tile (label above, big value below)."""
    ax = _card_ax(fig, rect)
    ax.text(0.5, 0.72, label,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color=GRAY_MID)
    ax.text(0.5, 0.38, str(value),
            transform=ax.transAxes, ha="center", va="center",
            fontsize=20, fontweight="bold", color=GREEN)
    ax.text(0.5, 0.10, unit,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color=GRAY_MID)
    return ax


def _pitch_ax(fig, rect):
    """Draw a mplsoccer pitch inside rect and return (pitch, ax)."""
    from mplsoccer import Pitch
    ax = fig.add_axes(rect)
    ax.set_facecolor(WHITE)
    pitch = Pitch(
        pitch_color="white",
        line_color=PITCH_LINE,
        line_zorder=2,
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
    )
    pitch.draw(ax=ax)
    return pitch, ax


def _wrap_text(ax, text, x, y, width, fontsize=8, color=GRAY_DRK, max_lines=10, **kwargs):
    """Draw text with manual word-wrapping at `width` characters.

    Line height is derived from the axes' actual height so text spacing is
    consistent regardless of card size.
    """
    import textwrap
    lines = textwrap.wrap(text or "", width=width)
    # Compute line height in axes-fraction coordinates:
    #   fontsize points → inches: fontsize / 72
    #   axes height in inches: ax.get_position().height * fig_height_in
    fig = ax.get_figure()
    ax_h_in = ax.get_position().height * fig.get_size_inches()[1]
    line_h = (fontsize * 1.5 / 72) / max(ax_h_in, 0.01)
    for i, line in enumerate(lines[:max_lines]):
        ax.text(x, y - i * line_h, line, fontsize=fontsize,
                color=color, transform=ax.transAxes, va="top",
                clip_on=True, **kwargs)
    return len(lines)


# ── PAGE 1: Player Card ───────────────────────────────────────────────────────

def _page1_player_card(pdf, target_stats, analysis, jersey_num, x_pitch, y_pitch, work_rate):
    fig = _new_page()

    # ── Layout (bottom → top, all in figure fractions) ────────────────────────
    #   [0.03, 0.03] footer
    #   [0.04, 0.05, 0.92, 0.10] field zone bar
    #   [0.04, 0.17, 0.92, 0.10] work rate bar
    #   [0.04, 0.29, 0.92, 0.27] heatmap (left desc + right pitch)
    #   [0.04, 0.58, 0.92, 0.13] narrative summary
    #   [x,    0.73, tile_w, 0.09] metric tiles
    #   [0.04, 0.84, 0.92, 0.10] header

    # ── header ────────────────────────────────────────────────────────────────
    header_ax = fig.add_axes([0.04, 0.84, 0.92, 0.10])
    header_ax.set_facecolor(BG)
    header_ax.axis("off")
    jersey_str = f"#{jersey_num}" if jersey_num is not None else "Player"
    header_ax.text(0.0, 1.0, jersey_str,
                   fontsize=40, fontweight="bold", color=GREEN,
                   va="top", transform=header_ax.transAxes)
    pos  = target_stats.get("inferred_position", "")
    team = target_stats.get("team", "")
    tv_s   = target_stats.get("total_time_visible_s", 0)
    tv_min = int(tv_s // 60)
    tv_sec = int(tv_s % 60)
    header_ax.text(0.0, 0.14,
                   f"{pos}  ·  {team}  ·  tracked {tv_min}m {tv_sec:02d}s",
                   fontsize=9, color=GRAY_MID, va="bottom",
                   transform=header_ax.transAxes)

    # ── 4 metric tiles ────────────────────────────────────────────────────────
    tile_y, tile_h = 0.73, 0.09
    tile_w, tile_gap = 0.215, 0.013
    tile_x0 = 0.04
    tiles = [
        ("Distance",  f"{target_stats.get('total_distance_m', 0):.0f}", "m"),
        ("Sprints",   str(target_stats.get("sprint_count", 0)),          ""),
        ("Max Speed", f"{target_stats.get('max_speed_mps', 0):.1f}",    "m/s"),
        ("Avg Speed", f"{target_stats.get('avg_speed_mps', 0):.1f}",    "m/s"),
    ]
    for i, (lbl, val, unit) in enumerate(tiles):
        x = tile_x0 + i * (tile_w + tile_gap)
        _metric_tile(fig, [x, tile_y, tile_w, tile_h], lbl, val, unit)

    # ── narrative summary card ─────────────────────────────────────────────────
    has_llm = analysis.get("player_summary") is not None
    summary = analysis.get("player_summary") or {}
    narrative = ""
    if has_llm and not summary.get("error") and not summary.get("parse_error"):
        narrative = summary.get("summary", "")
    if not narrative:
        thirds = target_stats.get("time_in_thirds", {})
        dominant = max(thirds, key=thirds.get, default="middle") if thirds else "middle"
        narrative = (
            f"Player spent most time in the {dominant} third "
            f"({thirds.get(dominant, 0):.0f}%), covering "
            f"{target_stats.get('total_distance_m', 0):.0f} m "
            f"with {target_stats.get('sprint_count', 0)} sprints recorded."
        )

    narr_ax = _card_ax(fig, [0.04, 0.58, 0.92, 0.13])
    # Green left accent bar
    narr_ax.add_patch(patches.FancyBboxPatch(
        (0, 0), 0.008, 1, boxstyle="square,pad=0",
        facecolor=GREEN, edgecolor="none", transform=narr_ax.transAxes,
    ))
    _wrap_text(narr_ax, narrative, 0.025, 0.84, width=108,
               fontsize=8.5, color=GRAY_DRK, max_lines=5, style="italic")

    # ── heatmap card ──────────────────────────────────────────────────────────
    thirds = target_stats.get("time_in_thirds", {})
    dominant = max(thirds, key=thirds.get, default="middle") if thirds else "middle"

    # Left: field zone stats
    desc_ax = _card_ax(fig, [0.04, 0.29, 0.33, 0.27], title="Position Map")
    stat_lines = [
        ("Defensive",  f"{thirds.get('defensive', 0):.1f}%"),
        ("Middle",     f"{thirds.get('middle', 0):.1f}%"),
        ("Attacking",  f"{thirds.get('attacking', 0):.1f}%"),
        ("",           ""),
        ("Frames",     f"{len(x_pitch):,}"),
    ]
    for i, (label, val) in enumerate(stat_lines):
        y_pos = 0.74 - i * 0.135
        if label:
            desc_ax.text(0.07, y_pos, label,
                         transform=desc_ax.transAxes, fontsize=8,
                         color=GRAY_MID, va="top")
            desc_ax.text(0.93, y_pos, val,
                         transform=desc_ax.transAxes, fontsize=8,
                         color=GREEN, fontweight="bold", va="top", ha="right")

    # Right: pitch with KDE or scatter
    pitch, p_ax = _pitch_ax(fig, [0.39, 0.29, 0.57, 0.27])
    if len(x_pitch) >= 50:
        pitch.kdeplot(x_pitch, y_pitch, ax=p_ax,
                      cmap="Greens", levels=10, fill=True, alpha=0.75)
    elif len(x_pitch) > 0:
        pitch.scatter(x_pitch, y_pitch, ax=p_ax,
                      color=GREEN, s=4, alpha=0.5, zorder=3)
    else:
        p_ax.text(0.5, 0.5, "Insufficient tracking data",
                  transform=p_ax.transAxes, ha="center", va="center",
                  fontsize=8, color=GRAY_MID)

    # ── work rate bar ─────────────────────────────────────────────────────────
    wr_ax = _card_ax(fig, [0.04, 0.17, 0.92, 0.10], title="Work Rate")
    _draw_work_rate_bar(wr_ax, work_rate)

    # ── field zone bar ────────────────────────────────────────────────────────
    zone_ax = _card_ax(fig, [0.04, 0.05, 0.92, 0.10], title="Field Zone Distribution")
    _draw_thirds_bar(zone_ax, thirds)

    # ── footer ────────────────────────────────────────────────────────────────
    fig.text(0.96, 0.015, "Soccer Analyzer",
             ha="right", va="bottom", fontsize=7, color=GRAY_MID)

    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def _draw_work_rate_bar(ax, work_rate):
    """Stacked horizontal bar: idle/jog/run/sprint with green shading."""
    labels  = ["Idle", "Jogging", "Running", "Sprinting"]
    colors  = ["#e0dbd5", "#b8d4c4", "#7bbf99", GREEN]
    values  = [
        work_rate.get("idle", 0),
        work_rate.get("jogging", 0),
        work_rate.get("running", 0),
        work_rate.get("sprinting", 0),
    ]
    bar_y, bar_h = 0.30, 0.28
    left = 0.04
    total = sum(values) or 1
    for val, col, lbl in zip(values, colors, labels):
        w = (val / total) * 0.92
        rect = patches.FancyBboxPatch(
            (left, bar_y), w, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=col, edgecolor="none",
            transform=ax.transAxes, clip_on=True,
        )
        ax.add_patch(rect)
        if w > 0.08:
            ax.text(
                left + w / 2, bar_y + bar_h / 2,
                f"{val:.0f}%",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=WHITE if col in (GREEN, "#7bbf99") else GRAY_DRK,
            )
        # label below
        ax.text(
            left + w / 2, bar_y - 0.12,
            lbl, transform=ax.transAxes,
            ha="center", va="top", fontsize=6, color=GRAY_MID,
        )
        left += w


def _draw_thirds_bar(ax, thirds):
    """Stacked thirds bar (defensive / middle / attacking)."""
    labels = ["Defensive", "Middle", "Attacking"]
    colors = ["#d4e6db", "#a8c8b5", GREEN]
    values = [
        thirds.get("defensive", 0),
        thirds.get("middle", 0),
        thirds.get("attacking", 0),
    ]
    bar_y, bar_h = 0.30, 0.28
    left = 0.04
    total = sum(values) or 1
    for val, col, lbl in zip(values, colors, labels):
        w = (val / total) * 0.92
        rect = patches.FancyBboxPatch(
            (left, bar_y), w, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=col, edgecolor="none",
            transform=ax.transAxes, clip_on=True,
        )
        ax.add_patch(rect)
        if w > 0.08:
            ax.text(
                left + w / 2, bar_y + bar_h / 2,
                f"{val:.0f}%",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=WHITE if col == GREEN else GRAY_DRK,
            )
        ax.text(
            left + w / 2, bar_y - 0.12,
            lbl, transform=ax.transAxes,
            ha="center", va="top", fontsize=6, color=GRAY_MID,
        )
        left += w


# ── PAGE 2: Performance Summary ───────────────────────────────────────────────

def _page2_performance(pdf, target_stats, analysis, fps, total_frames):
    fig = _new_page()
    has_llm = analysis.get("player_summary") is not None
    summary = analysis.get("player_summary") or {}

    # ── detailed narrative (Page 2 uses the longer narrative, not the summary
    #    which already appears on Page 1) ───────────────────────────────────────
    ov_ax = _card_ax(fig, [0.04, 0.79, 0.92, 0.19], title="Detailed Analysis")
    if has_llm and not summary.get("error") and not summary.get("parse_error"):
        # Prefer detailed_narrative; fall back to summary if absent
        overview_txt = summary.get("detailed_narrative") or summary.get("summary") or ""
    else:
        overview_txt = (
            "LLM analysis not available — re-run with ANTHROPIC_API_KEY set."
        )
    # Wrap at ~100 chars; max 6 lines fits the card height
    _wrap_text(ov_ax, overview_txt, 0.03, 0.84, width=100, fontsize=7.5, max_lines=6)

    # ── sprint timeline ────────────────────────────────────────────────────────
    sprint_eps = target_stats.get("sprint_episodes", [])
    sp_ax = _card_ax(fig, [0.04, 0.60, 0.92, 0.16], title="Sprint Timeline")
    if sprint_eps and total_frames > 0:
        _draw_episode_timeline(sp_ax, sprint_eps, total_frames, fps,
                               color=SPRINT_COLOR, label="Sprint")
    else:
        sp_ax.text(0.5, 0.50, "No sprints detected",
                   transform=sp_ax.transAxes, ha="center", va="center",
                   fontsize=9, color=GRAY_MID)

    # ── ball proximity timeline ────────────────────────────────────────────────
    prox_eps = target_stats.get("ball_proximity_episodes", None)
    if prox_eps is not None:
        pr_ax = _card_ax(fig, [0.04, 0.42, 0.92, 0.15], title="Ball Proximity")
        if prox_eps and total_frames > 0:
            _draw_episode_timeline(pr_ax, prox_eps, total_frames, fps,
                                   color=PROX_COLOR, label="Near ball")
        else:
            pr_ax.text(0.5, 0.50, "No ball proximity episodes detected",
                       transform=pr_ax.transAxes, ha="center", va="center",
                       fontsize=9, color=GRAY_MID)
        strengths_y = 0.20
    else:
        strengths_y = 0.36

    # ── strengths / areas ─────────────────────────────────────────────────────
    strengths = summary.get("strengths", []) if has_llm else []
    areas     = summary.get("areas_to_improve", []) if has_llm else []
    if strengths or areas:
        card_h = strengths_y - 0.04
        str_ax  = _card_ax(fig, [0.04, 0.04, 0.44, card_h], title="Strengths")
        area_ax = _card_ax(fig, [0.52, 0.04, 0.44, card_h], title="Areas to Improve")

        # Each bullet is wrapped at ~42 chars to fit the half-width card.
        # We show at most 3 bullets to leave room for multi-line wrapping.
        BULLET_WRAP = 42
        BULLET_MAX_LINES = 3   # lines per bullet before truncating

        def _draw_bullets(ax, items):
            import textwrap
            fig_obj = ax.get_figure()
            ax_h_in = ax.get_position().height * fig_obj.get_size_inches()[1]
            line_h = (7.0 * 1.4 / 72) / max(ax_h_in, 0.01)
            y = 0.82
            for item in items[:3]:
                # First line gets the bullet, continuation lines indent
                wrapped = textwrap.wrap(item, width=BULLET_WRAP)
                for j, line in enumerate(wrapped[:BULLET_MAX_LINES]):
                    prefix = "• " if j == 0 else "  "
                    ax.text(0.05, y, prefix + line,
                            transform=ax.transAxes, fontsize=7,
                            color=GRAY_DRK, va="top", clip_on=True)
                    y -= line_h
                y -= line_h * 0.4   # extra gap between bullets

        _draw_bullets(str_ax, strengths)
        _draw_bullets(area_ax, areas)

    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def _draw_episode_timeline(ax, episodes, total_frames, fps, color, label):
    """Draw a horizontal timeline bar with coloured episode segments."""
    timeline_y, timeline_h = 0.38, 0.22
    total_s = total_frames / fps

    # Background track
    bg = patches.FancyBboxPatch(
        (0.04, timeline_y), 0.92, timeline_h,
        boxstyle="round,pad=0.005",
        facecolor=GRAY, edgecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(bg)

    # Episode segments
    for ep in episodes:
        start_s = ep.get("start_frame", 0) / fps
        end_s   = ep.get("end_frame", 0) / fps
        x0 = 0.04 + (start_s / total_s) * 0.92
        w  = max((end_s - start_s) / total_s * 0.92, 0.003)
        seg = patches.FancyBboxPatch(
            (x0, timeline_y), w, timeline_h,
            boxstyle="round,pad=0.001",
            facecolor=color, edgecolor="none",
            transform=ax.transAxes,
        )
        ax.add_patch(seg)

    # Axis labels: 0:00 … match duration
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        t_s = frac * total_s
        ax.text(
            0.04 + frac * 0.92, timeline_y - 0.10,
            f"{int(t_s//60)}:{int(t_s%60):02d}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color=GRAY_MID,
        )

    # Count label
    ax.text(
        0.5, 0.88,
        f"{len(episodes)} {label.lower()} episode{'s' if len(episodes) != 1 else ''}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7, color=GRAY_MID,
    )


# ── PAGE 3: Coaching Moments ──────────────────────────────────────────────────

def _page3_coaching_moments(pdf, analysis):
    fig = _new_page()

    improvements = analysis.get("improvements") or {}
    suggestions  = improvements.get("suggestions", []) if isinstance(improvements, dict) else []

    # Page title
    fig.text(0.5, 0.96, "Coaching Moments",
             ha="center", va="top", fontsize=14,
             fontweight="bold", color=GREEN)
    fig.text(0.5, 0.925, "Key moments identified by AI analysis",
             ha="center", va="top", fontsize=9, color=GRAY_MID)

    if not suggestions:
        # Fallback placeholder
        ph_ax = _card_ax(fig, [0.10, 0.55, 0.80, 0.20])
        ph_ax.text(0.5, 0.6, "No coaching moments available",
                   transform=ph_ax.transAxes, ha="center", va="center",
                   fontsize=11, color=GRAY_MID)
        ph_ax.text(0.5, 0.38, "Re-run with ANTHROPIC_API_KEY to generate AI analysis",
                   transform=ph_ax.transAxes, ha="center", va="center",
                   fontsize=8, color=GRAY_MID)
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        return

    # Show 3 moments per page; taller cards so text never overlaps.
    #
    # Card interior layout (axes fraction, card_h = 0.24):
    #   0.92 ── timestamp (bold green)
    #   0.78 ── description, italic, max 3 lines
    #   ── divider at 0.40 ──
    #   0.36 ── recommendation, bold green, max 3 lines
    #
    # With card_h=0.24 → ax height ≈ 2.8 in → line_h ≈ 0.059
    # 3 desc lines end at 0.78 − 2×0.059 = 0.66  (well above divider at 0.40)
    # 3 rec lines end at 0.36 − 2×0.059 = 0.24   (stays inside card)

    n      = min(len(suggestions), 3)
    card_h = 0.24
    gap    = 0.028
    top    = 0.88

    for i, sug in enumerate(suggestions[:n]):
        y      = top - i * (card_h + gap)
        card_ax = _card_ax(fig, [0.04, y - card_h, 0.92, card_h])

        # Left green border
        card_ax.add_patch(patches.FancyBboxPatch(
            (0, 0), 0.010, 1,
            boxstyle="square,pad=0",
            facecolor=GREEN, edgecolor="none",
            transform=card_ax.transAxes,
        ))

        # Timestamp
        ts_start = sug.get("timestamp_start", "")
        ts_end   = sug.get("timestamp_end", "")
        ts_str   = f"{ts_start} – {ts_end}" if ts_start else ""
        card_ax.text(0.03, 0.92, ts_str,
                     transform=card_ax.transAxes, fontsize=8,
                     fontweight="bold", color=GREEN, va="top", clip_on=True)

        # Description (italic, max 3 lines, anchored near top)
        desc = sug.get("description", "")
        _wrap_text(card_ax, desc, 0.03, 0.78,
                   width=92, fontsize=7.5, color=GRAY_DRK,
                   max_lines=3, style="italic")

        # Thin divider between description and recommendation
        card_ax.axhline(0.40, xmin=0.03, xmax=0.97,
                        color=GRAY, linewidth=0.5)

        # Recommendation (bold green, max 3 lines, anchored at bottom half)
        rec = sug.get("recommendation", "")
        if rec:
            _wrap_text(card_ax, f"→ {rec}", 0.03, 0.36,
                       width=92, fontsize=7.5, color=GREEN,
                       max_lines=3, fontweight="bold")

    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


# ── PAGE 4: Game Maps (Phase 2 placeholders) ──────────────────────────────────

def _page4_game_maps(pdf):
    fig = _new_page()

    fig.text(0.5, 0.96, "Game Maps",
             ha="center", va="top", fontsize=14,
             fontweight="bold", color=GREEN)
    fig.text(0.5, 0.925, "Advanced event maps — coming in Phase 2",
             ha="center", va="top", fontsize=9, color=GRAY_MID)

    for title, y_top in [("Pass Map", 0.88), ("Shot Map", 0.44)]:
        card_ax = _card_ax(fig, [0.06, y_top - 0.38, 0.88, 0.38], title=title)

        # Placeholder pitch
        try:
            from mplsoccer import Pitch
            inner_ax = fig.add_axes([0.14, (y_top - 0.38) + 0.04, 0.72, 0.26])
            inner_ax.set_facecolor(WHITE)
            pitch = Pitch(
                pitch_color="white",
                line_color="#e0dbd5",
                line_zorder=2,
                pitch_type="custom",
                pitch_length=105,
                pitch_width=68,
            )
            pitch.draw(ax=inner_ax)
        except Exception:
            pass

        # Phase 2 badge
        badge_ax = card_ax.inset_axes([0.70, 0.75, 0.25, 0.16])
        badge_ax.set_facecolor(GREEN)
        badge_ax.set_xticks([])
        badge_ax.set_yticks([])
        badge_ax.text(0.5, 0.5, "PHASE 2",
                      transform=badge_ax.transAxes,
                      ha="center", va="center",
                      fontsize=7, fontweight="bold", color=WHITE)

        card_ax.text(0.5, 0.15,
                     "Requires ball event detection",
                     transform=card_ax.transAxes,
                     ha="center", va="center",
                     fontsize=8, color=GRAY_MID)

    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)
