# Newsletter Story Classification

## Problem

Ministry newsletters are sent to `newsletters@dm.org` and arrive in the same Gmail inbox the email-labeler already monitors. These newsletters contain stories about campus ministry work, and we want to automatically assess story quality and categorize stories by organizational mission themes.

## Approach

Branch-in-loop: detect newsletter emails early in the daemon's processing flow via a deterministic `To:` header check. Newsletter emails skip the existing priority classification (NEEDS_RESPONSE / FYI / LOW_PRIORITY) and instead enter a dedicated newsletter pipeline.

## Pipeline

### 1. Detection & Routing

- After fetching a thread, check if the `To:` header contains the configured newsletter recipient address (`newsletters@dm.org`).
- If matched, route to the newsletter pipeline. Otherwise, continue with existing priority classification.
- Detection is deterministic (no LLM call).
- Configured in `config.toml`:

```toml
[newsletter]
recipient = "newsletters@dm.org"
output_file = "data/newsletter_assessments.jsonl"
```

### 2. Story Extraction (1 LLM call per email)

- Cloud LLM receives the full newsletter body.
- Returns a list of individual stories, each with a short title and extracted text.
- Non-story content (headers, footers, donation appeals, event calendars, administrative announcements) is skipped.
- If no stories are found, the email is labeled `agent/newsletter/no-stories` and no further LLM calls are made.

### 3. Quality Assessment (1 LLM call per story)

Each story is scored on four dimensions (1-5 scale):

| Dimension    | High score (5)                                                              | Low score (1)                                                              |
| ------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Simple**   | Focuses on one key idea or progression; no tangents or extraneous details   | Tries to cover too many things; scattered, unfocused                       |
| **Concrete** | Narrates particular events involving particular people at particular places | Abstract, generalized; talks about concepts rather than showing moments    |
| **Personal** | Centers around one or a few people; their concerns drive the story          | Mainly about events, numbers, programs, or ideas disconnected from people  |
| **Dynamic**  | Describes how a person changes over time; shows a before and after          | Static; describes a situation or person without any arc or transformation  |

Overall quality tier derived from average score:

| Tier          | Criteria       |
| ------------- | -------------- |
| **Excellent** | Average >= 4.0 |
| **Good**      | Average >= 3.0 |
| **Fair**      | Average >= 2.0 |
| **Poor**      | Average < 2.0  |

Chain-of-thought reasoning is captured and stored for later examination.

### 4. Theme Classification (1 LLM call per story)

Each story is tagged with one or more themes derived from the Board of Directors' Ends Statement:

| Theme Label       | Ends Statement Bullet                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `scripture`       | Study the Scriptures correctly, apply them to all of life, and teach them to others       |
| `christlikeness`  | Exhibit increasing Christlikeness in response to the Gospel                               |
| `church`          | Serve not only in their campus fellowship but also in a Bible-believing local church      |
| `vocation-family` | Honor God in their vocation and family relationships                                      |
| `disciple-making` | Continue to make disciples wherever God takes them                                        |

- Multiple themes per story are allowed.
- If a story doesn't clearly fit any theme, it gets no theme tags.
- Chain-of-thought reasoning is captured and stored.

## Output

### Gmail Labels

12 new labels, all pre-created manually in Gmail:

- `agent/newsletter` — marker label for all newsletter emails
- `agent/newsletter/excellent`, `agent/newsletter/good`, `agent/newsletter/fair`, `agent/newsletter/poor` — quality tier of best story
- `agent/newsletter/no-stories` — newsletter contained no extractable stories
- `agent/newsletter/theme/scripture`, `agent/newsletter/theme/christlikeness`, `agent/newsletter/theme/church`, `agent/newsletter/theme/vocation-family`, `agent/newsletter/theme/disciple-making` — theme labels

Per email, applied labels are:
- `agent/processed` (existing)
- `agent/newsletter`
- One quality tier label (from the best story's tier)
- One or more theme labels (union across all stories)

All newsletter emails are archived (removed from inbox) after labeling.

### Structured Storage

Detailed per-story results appended to a JSONL file (`data/newsletter_assessments.jsonl`):

```json
{
  "timestamp": "2026-02-19T14:30:00Z",
  "message_id": "abc123",
  "thread_id": "thread456",
  "from": "john.smith@dm.org",
  "subject": "February Campus Update - Penn State",
  "overall_tier": "excellent",
  "stories": [
    {
      "title": "Sarah's Journey from Skeptic to Small Group Leader",
      "scores": {
        "simple": 5,
        "concrete": 4,
        "personal": 5,
        "dynamic": 5
      },
      "average_score": 4.75,
      "tier": "excellent",
      "themes": ["christlikeness", "disciple-making"],
      "quality_cot": "The story follows one person clearly...",
      "theme_cot": "Sarah's transformation reflects increasing Christlikeness..."
    }
  ]
}
```

## Error Handling

| Scenario                              | Behavior                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------- |
| Story extraction LLM call fails       | Email skipped, retried next cycle                                         |
| Quality assessment fails for a story  | That story gets no scores; other stories still processed                  |
| Theme classification fails for a story| That story gets no themes; quality scores still retained                  |
| No extractable stories                | Labeled `agent/newsletter/no-stories`, marked processed, archived         |
| JSONL write fails                     | Log error, still apply Gmail labels                                       |
| `To:` header missing/unparseable      | Falls through to existing priority classification                         |

## Concurrency

- All newsletter LLM calls use the cloud LLM and share the existing `cloud_parallel` concurrency limit.
- For a newsletter with N stories, the pipeline makes 1 + 2N cloud LLM calls (1 extraction + N quality + N theme).

## Label Verification

The daemon's startup `verify_labels()` check is extended to verify all 12 new newsletter labels exist in Gmail.
