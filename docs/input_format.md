# Input format

Commands accept UTF-8 JSONL with one `DecisionItem` per line. The required
fields are `item_id`, `source`, `question`, `choices`, and `gold_keys`.

```json
{
  "item_id": "example-001",
  "context_kind": "passage",
  "source": "Background text presented to the decision model.",
  "question": "Which option is best?",
  "choices": [
    {"key": "A", "text": "First option"},
    {"key": "B", "text": "Second option"}
  ],
  "gold_keys": ["A"],
  "metadata": {}
}
```

`context_kind` is optional and may be `question_only`, `passage`, or `dialogue`;
it defaults to `passage`. `metadata` is an optional free-form object.

Choices must have unique keys, and every gold key must name an available
choice. A single gold key produces one bounded-choice decision unit. Multiple
gold keys produce one binary membership unit per choice so absent options can
be evaluated as fixed targets.

## Insertion boundaries

The validator derives deterministic sentence boundaries for passage and
dialogue inputs. For `question_only` inputs, additions are placed before the
source. Advanced callers may supply `insertion_boundaries` explicitly:

```json
"insertion_boundaries": [
  {"id": "after-intro", "offset": 31, "left": "", "right": ""}
]
```

Offsets are character positions in the unchanged `source`. `left` and `right`
are optional local-context hints. Context additions are inserted at these
positions; the original source, question, choices, and gold keys are not
rewritten.
