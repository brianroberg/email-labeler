# Newsletter Assessment TUI

## Problem

The email-labeler daemon writes newsletter classification results to `data/newsletter_assessments.jsonl`. There's no way to browse, filter, or inspect these results interactively. Reviewing the raw JSONL is tedious, especially when examining the chain-of-thought reasoning the model recorded during classification.

## Approach

Build a Textual TUI that loads the JSONL file and provides a three-panel List → Detail interface with tier and theme filtering.

## Architecture

Two flat modules following the project's existing convention:

- **`tui_data.py`** — Pure data layer: loads JSONL, parses into frozen dataclasses, provides filtering functions. No Textual dependency.
- **`tui.py`** — Textual App: newsletter list, story list, detail panel, filter controls. Entry point via `[project.scripts]`.

### Data Models (`tui_data.py`)

```python
@dataclass(frozen=True)
class Story:
    title: str
    text: str
    scores: dict[str, int] | None
    average_score: float | None
    tier: str | None          # "excellent", "good", "fair", "poor"
    themes: list[str]
    quality_cot: str
    theme_cot: str

@dataclass(frozen=True)
class Assessment:
    timestamp: str
    message_id: str
    thread_id: str
    sender: str               # "from" field in JSONL
    subject: str
    overall_tier: str | None
    stories: list[Story]
```

### Loading & Filtering

- `load_assessments(path: str) -> list[Assessment]` — reads JSONL, skips malformed lines with a warning.
- `filter_by_tier(assessments, tier: str) -> list[Assessment]` — filters by `overall_tier`.
- `filter_by_theme(assessments, theme: str) -> list[Assessment]` — keeps assessments where any story has the given theme.
- `available_tiers(assessments) -> list[str]` — distinct tiers present in data.
- `available_themes(assessments) -> list[str]` — distinct themes present in data.

### TUI Layout

```
┌─ Header ───────────────────────────────────────┐
│ Tier: [All ▾]  Theme: [All ▾]   12 of 47      │
├─ Newsletter List ──────────────────────────────┤
│  Subject                    Tier      Date     │
│ > Feb Update - Penn State   excellent Feb 19   │
│   Jan Report - Ohio State   good      Jan 15   │
├─ Stories ──────────────────────────────────────┤
│  Title                      Tier      Avg      │
│ ▸ Sarah's Journey           excellent 4.75     │
│   Campus Outreach Week      fair      2.50     │
├─ Detail ───────────────────────────────────────┤
│  Scores: simple=5 concrete=4 personal=5 dyn=5  │
│  Themes: christlikeness, disciple-making        │
│  ── Quality CoT ──                              │
│  The story follows one person (Sarah) clearly...│
│  ── Theme CoT ──                                │
│  Sarah's transformation reflects increasing...  │
└─────────────────────────────────────────────────┘
```

### Key Bindings

- `↑`/`↓` — navigate lists
- `Tab` — cycle focus between newsletter list, story list, detail panel
- `t` — cycle tier filter (All → excellent → good → fair → poor → All)
- `h` — cycle theme filter (All → scripture → christlikeness → ... → All)
- `q` — quit

### Behavior

- Selecting a newsletter populates the story list.
- Selecting a story populates the detail panel (scores, themes, CoT text).
- Changing a filter updates the newsletter list and resets story/detail selections.
- The detail panel scrolls for long CoT text.

## Entry Point

`[project.scripts]` entry in `pyproject.toml`: `tui = "tui:main"`. Accepts an optional positional argument for the JSONL file path, defaulting to `data/newsletter_assessments.jsonl`.

## Testing Strategy (Red/Green TDD)

### `tests/test_tui_data.py` — Data layer (bulk of TDD)

- `TestLoadAssessments` — load valid JSONL, empty file, malformed lines, missing file
- `TestFilterByTier` — filter by each tier, "all" returns everything, null tiers
- `TestFilterByTheme` — filter by each theme, matches any story's themes
- `TestAvailableTiers` / `TestAvailableThemes` — extract unique values

### `tests/test_tui.py` — Textual pilot tests (lighter)

- App launches and shows data
- Selecting a newsletter shows its stories
- Selecting a story shows its detail/CoT
- Tier filter narrows the list
- Theme filter narrows the list
- `q` quits the app

## Dependencies

Add to `pyproject.toml`:
- `textual>=0.85.0` (core dependency)
- `textual-dev>=1.0.0` (dev dependency, for testing)

## Files to Create/Modify

- **Create**: `tui_data.py`, `tui.py`, `tests/test_tui_data.py`, `tests/test_tui.py`
- **Modify**: `pyproject.toml` (add textual dependencies and scripts entry)
