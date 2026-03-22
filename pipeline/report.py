"""Stage 5: Generate the final Markdown report."""

import json
import os
from datetime import datetime


def run(analysis_path: str, stats_path: str, identity_path: str,
        output_dir: str) -> str:
    """
    Generate a formatted Markdown report.

    Returns path to the output report file.
    """
    print(f"\n[Stage 5] Report Generation")

    with open(analysis_path) as f:
        analysis = json.load(f)
    with open(stats_path) as f:
        stats_data = json.load(f)
    with open(identity_path) as f:
        id_data = json.load(f)

    target_player = analysis.get("target_player", "Unknown Player")
    has_llm = analysis.get("player_summary") is not None

    lines = []
    lines.append(f"# Soccer Game Analysis Report")
    lines.append(f"")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Target Player**: {target_player}")
    lines.append(f"**Team**: {id_data.get('target_team', 'Unknown')}")
    lines.append(f"")

    # Find target player stats
    target_stats = None
    for p in stats_data["players"]:
        if p.get("is_target"):
            target_stats = p
            break

    # --- Section 1: Performance Summary ---
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 1. Performance Summary")
    lines.append(f"")

    if has_llm and not analysis["player_summary"].get("error"):
        summary = analysis["player_summary"]
        if summary.get("parse_error"):
            lines.append(summary.get("raw_text", "Analysis unavailable."))
        else:
            lines.append(f"### Overview")
            lines.append(f"")
            lines.append(summary.get("summary", ""))
            lines.append(f"")

            if summary.get("detailed_narrative"):
                lines.append(summary["detailed_narrative"])
                lines.append(f"")

            if summary.get("strengths"):
                lines.append(f"### Strengths")
                lines.append(f"")
                for s in summary["strengths"]:
                    lines.append(f"- {s}")
                lines.append(f"")

            if summary.get("areas_to_improve"):
                lines.append(f"### Areas to Improve")
                lines.append(f"")
                for a in summary["areas_to_improve"]:
                    lines.append(f"- {a}")
                lines.append(f"")
    else:
        lines.append(_stats_only_summary(target_stats))

    # --- Section 2: Improvement Suggestions ---
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 2. Improvement Suggestions")
    lines.append(f"")

    if has_llm and not analysis.get("improvements", {}).get("error"):
        improvements = analysis.get("improvements", {})
        suggestions = improvements.get("suggestions", [])

        if not suggestions and improvements.get("raw_text"):
            lines.append(improvements["raw_text"])
        else:
            for i, sug in enumerate(suggestions, 1):
                ts = f"[{sug.get('timestamp_start', '?')} - {sug.get('timestamp_end', '?')}]"
                lines.append(f"### {i}. {ts}")
                lines.append(f"")
                lines.append(f"**What happened**: {sug.get('description', 'N/A')}")
                lines.append(f"")
                lines.append(f"**Suggestion**: {sug.get('recommendation', 'N/A')}")
                lines.append(f"")
                lines.append(f"**Why**: {sug.get('reasoning', 'N/A')}")
                lines.append(f"")
    else:
        lines.append("*LLM analysis not available. Key moments from tracking data:*")
        lines.append("")
        if target_stats and target_stats.get("key_moments"):
            for km in target_stats["key_moments"][:10]:
                tr = km.get("time_range", {})
                lines.append(f"- **[{tr.get('start_time', '?')} - {tr.get('end_time', '?')}]** "
                              f"{km.get('moment_type', '')}: {km.get('description', '')}")
            lines.append("")

    # --- Section 3: Team Strategies ---
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 3. Team Strategies")
    lines.append(f"")

    if has_llm and not analysis.get("team_strategy", {}).get("error"):
        strategy = analysis.get("team_strategy", {})
        observations = strategy.get("observations", [])

        if not observations and strategy.get("raw_text"):
            lines.append(strategy["raw_text"])
        else:
            for i, obs in enumerate(observations, 1):
                lines.append(f"### {i}. {obs.get('title', 'Observation')}")
                lines.append(f"")
                lines.append(obs.get("description", ""))
                lines.append(f"")
                refs = obs.get("timestamp_refs", [])
                if refs:
                    lines.append(f"**Key moments**: {', '.join(refs)}")
                    lines.append(f"")
                lines.append(f"**Recommendation**: {obs.get('recommendation', 'N/A')}")
                lines.append(f"")
    else:
        lines.append("*LLM analysis not available. Set ANTHROPIC_API_KEY and re-run with --stage 4.*")
        lines.append("")

    # --- Appendix: Raw Statistics ---
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Appendix: Player Statistics")
    lines.append(f"")

    if target_stats:
        lines.append(f"### {target_player}")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Inferred Position | {target_stats.get('inferred_position', 'N/A')} |")
        lines.append(f"| Total Distance | {target_stats.get('total_distance_m', 0):.0f} m |")
        lines.append(f"| Average Speed | {target_stats.get('avg_speed_mps', 0):.1f} m/s |")
        lines.append(f"| Max Speed | {target_stats.get('max_speed_mps', 0):.1f} m/s |")
        lines.append(f"| Sprint Count | {target_stats.get('sprint_count', 0)} |")
        lines.append(f"| Time Visible | {target_stats.get('total_time_visible_s', 0):.0f} s |")
        lines.append(f"| Frames Tracked | {target_stats.get('frames_visible', 0)} |")
        lines.append(f"")

        thirds = target_stats.get("time_in_thirds", {})
        if thirds:
            lines.append(f"**Field Position Distribution**:")
            lines.append(f"- Defensive third: {thirds.get('defensive', 0):.1f}%")
            lines.append(f"- Middle third: {thirds.get('middle', 0):.1f}%")
            lines.append(f"- Attacking third: {thirds.get('attacking', 0):.1f}%")
            lines.append(f"")

    # Other players summary table
    other_players = [p for p in stats_data["players"] if not p.get("is_target")]
    if other_players:
        lines.append(f"### Other Tracked Players")
        lines.append(f"")
        lines.append(f"| Player | Team | Position | Distance (m) | Sprints |")
        lines.append(f"|--------|------|----------|-------------|---------|")
        for p in other_players[:15]:  # Limit to 15
            lines.append(
                f"| {p.get('player_id', '?')} "
                f"| {p.get('team', '?')} "
                f"| {p.get('inferred_position', '?')} "
                f"| {p.get('total_distance_m', 0):.0f} "
                f"| {p.get('sprint_count', 0)} |"
            )
        lines.append(f"")

    # Footer
    lines.append(f"---")
    lines.append(f"*Generated by Soccer Analyzer*")

    # Write report
    report = "\n".join(lines)
    output_path = os.path.join(output_dir, "report.md")

    with open(output_path, "w") as f:
        f.write(report)

    print(f"  Report saved to {output_path}")
    print(f"  ({len(report)} characters, {len(lines)} lines)")
    return output_path


def _stats_only_summary(stats: dict) -> str:
    """Generate a basic summary from stats when LLM is unavailable."""
    if not stats:
        return "*No player statistics available.*\n"

    pos = stats.get("inferred_position", "unknown position")
    dist = stats.get("total_distance_m", 0)
    sprints = stats.get("sprint_count", 0)
    avg_speed = stats.get("avg_speed_mps", 0)
    max_speed = stats.get("max_speed_mps", 0)
    thirds = stats.get("time_in_thirds", {})

    dominant_third = max(thirds.items(), key=lambda x: x[1])[0] if thirds else "middle"

    return (
        f"*LLM analysis not available. Basic stats summary:*\n\n"
        f"The player appears to play as a **{pos}**, spending most time in the "
        f"**{dominant_third} third** ({thirds.get(dominant_third, 0):.0f}%). "
        f"They covered approximately **{dist:.0f} meters** with an average speed of "
        f"**{avg_speed:.1f} m/s** (max: **{max_speed:.1f} m/s**). "
        f"**{sprints} sprint(s)** were detected during the game.\n\n"
        f"*Set ANTHROPIC_API_KEY and re-run with `--stage 4` for detailed AI analysis.*\n"
    )
