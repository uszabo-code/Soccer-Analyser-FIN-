"""Stage 4: LLM-powered analysis using Claude API."""

import json
import os
import time

import config


def run(stats_path: str, identity_path: str, output_dir: str) -> str:
    """
    Generate AI-powered analysis using Claude API.

    Returns path to the output analysis JSON file.
    """
    print(f"\n[Stage 4] LLM Analysis")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set. Skipping LLM analysis.")
        print("  Set the key and re-run with --stage 4 to generate AI analysis.")
        print("  Get a key at: https://console.anthropic.com/")
        return _save_empty_analysis(output_dir)

    try:
        import anthropic
    except ImportError:
        print("  WARNING: anthropic package not installed. Skipping LLM analysis.")
        return _save_empty_analysis(output_dir)

    # Load data
    with open(stats_path) as f:
        stats_data = json.load(f)
    with open(identity_path) as f:
        id_data = json.load(f)

    # Find target player stats
    target_stats = None
    team_stats = []
    for p in stats_data["players"]:
        if p.get("is_target"):
            target_stats = p
        if p.get("team") == id_data.get("target_team"):
            team_stats.append(p)

    if target_stats is None:
        print("  WARNING: Could not find target player in stats. Using first player.")
        target_stats = stats_data["players"][0] if stats_data["players"] else {}

    # Load prompt templates
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

    client = anthropic.Anthropic(api_key=api_key)

    # --- Call 1: Player Summary ---
    print("  Generating player performance summary...")
    player_summary = _call_claude(
        client=client,
        prompt_template_path=os.path.join(prompts_dir, "player_summary.txt"),
        replacements={"player_data": json.dumps(target_stats, indent=2)},
    )

    # Wait for per-minute token window to reset before next call
    time.sleep(65)

    # --- Call 2: Improvement Suggestions ---
    print("  Generating improvement suggestions...")
    key_moments_data = target_stats.get("key_moments", [])
    improvements = _call_claude(
        client=client,
        prompt_template_path=os.path.join(prompts_dir, "improvement.txt"),
        replacements={
            "player_data": json.dumps(target_stats, indent=2),
            "key_moments": json.dumps(key_moments_data, indent=2),
        },
    )

    # Wait for per-minute token window to reset before next call
    time.sleep(65)

    # --- Call 3: Team Strategy ---
    print("  Generating team strategy observations...")
    # Trim team stats to avoid token limit — send summaries, not raw moments/heatmaps
    team_summary = []
    for p in team_stats[:15]:
        team_summary.append({
            "player_id": p.get("player_id"),
            "team": p.get("team"),
            "inferred_position": p.get("inferred_position"),
            "total_distance_m": round(p.get("total_distance_m", 0), 1),
            "avg_speed_mps": round(p.get("avg_speed_mps", 0), 2),
            "max_speed_mps": round(p.get("max_speed_mps", 0), 2),
            "sprint_count": p.get("sprint_count", 0),
            "time_in_thirds": p.get("time_in_thirds", {}),
            "total_time_visible_s": round(p.get("total_time_visible_s", 0), 1),
        })
    team_strategy = _call_claude(
        client=client,
        prompt_template_path=os.path.join(prompts_dir, "team_strategy.txt"),
        replacements={"team_data": json.dumps(team_summary, indent=2)},
    )

    # Save
    output_path = os.path.join(output_dir, "analysis.json")
    output_data = {
        "player_summary": player_summary,
        "improvements": improvements,
        "team_strategy": team_strategy,
        "target_player": target_stats.get("player_id", "Unknown"),
        "llm_model": config.CLAUDE_MODEL,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved to {output_path}")
    return output_path


def _call_claude(client, prompt_template_path: str, replacements: dict,
                 max_retries: int = 5) -> dict:
    """Call Claude API with a prompt template and parse JSON response."""
    with open(prompt_template_path) as f:
        prompt = f.read()

    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                temperature=config.CLAUDE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text

            # Try to parse as JSON
            # Handle case where response has markdown code fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())

        except json.JSONDecodeError:
            # If JSON parsing fails, return raw text in a wrapper
            return {"raw_text": text, "parse_error": True}
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e)
            if attempt < max_retries - 1:
                # Rate-limit errors need ~60s for the per-minute token window to reset.
                # Short exponential backoff is fine for transient network errors.
                wait = 65 if is_rate_limit else 2 ** attempt
                print(f"    API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error after {max_retries} attempts: {e}")
                return {"error": str(e)}

    return {"error": "Max retries exceeded"}


def _save_empty_analysis(output_dir: str) -> str:
    """Save an empty analysis file when LLM is unavailable."""
    output_path = os.path.join(output_dir, "analysis.json")
    output_data = {
        "player_summary": None,
        "improvements": None,
        "team_strategy": None,
        "target_player": "Unknown",
        "llm_model": None,
        "note": "LLM analysis skipped. Set ANTHROPIC_API_KEY and re-run with --stage 4",
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_path
