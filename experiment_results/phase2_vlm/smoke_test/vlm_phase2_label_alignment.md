# VLM Phase 2 Label Alignment: Stop Sign Support

This document details the alignment of the production VLM node configurations with the Phase 2 classification benchmark (which includes `STOP_SIGN`, not present in the original production taxonomy).

## 1. Prompt Alignment
In production, the VLM is queried with the full 12-gesture taxonomy (including `ROLL_BACK`, `SWIPE_DOWN`, etc.) and does not list `STOP_SIGN`. 

For the Phase 2 evaluation, we introduce a **Benchmark Mode**. When enabled, the model prompt is dynamically swapped to restrict the classification choice space exclusively to the validation categories:

```text
You will be given frames from a short video.
Each video contains one of the following gestures:

* SWIPE_LEFT
* SWIPE_RIGHT
* ROLL_FWD
* STOP_SIGN

If none match, answer UNKNOWN.
Respond with ONLY one label.
No explanation.
No reasoning.
No extra text.
```

---

## 2. Parser Alignment
To handle the transition, the parser has been expanded to support the validation label mappings.

| Benchmark Class | Mapped Label ID | Recognized Substrings / Patterns |
|---|---|---|
| **SWIPE_LEFT** | `0` | `swipe left`, `swiping left`, `swipe_left`, `swiping_left`, `swipe` & `left` |
| **SWIPE_RIGHT** | `1` | `swipe right`, `swiping right`, `swipe_right`, `swiping_right`, `swipe` & `right` |
| **ROLL_FWD** | `2` | `rolling hand forward`, `roll forward`, `roll fwd`, `rolling forward`, `roll_fwd`, `roll` |
| **STOP_SIGN** | `3` | `stop sign`, `stop hand`, `stop gesture`, `open palm`, `stop signal`, `stop_sign`, `stop` |
| **UNKNOWN** | `-1` / `UNKNOWN` | Any output not matching the above |

---

## 3. Production Node Alignment (Non-destructive)
To ensure production compatibility without permanently modifying the existing node, the benchmark mode will be controlled via an environment variable `VLM_BENCHMARK_MODE=1`.

When `VLM_BENCHMARK_MODE` is enabled:
1. The label taxonomy overrides production settings.
2. The prompt swaps to the restricted benchmark prompt.
3. The parser incorporates the improved substring matching rules.
4. If disabled or unset, the node executes in production mode.
