# Qwen3-VL 8B Closed-Set Parser Report

## Fallback Mapping Rules

In closed-set mode, all model predictions must map to one of the 4 valid classes. The fallback rules applied when no direct match is found are:

1. **Empty Response**: Mapped to `STOP_SIGN`.
2. **Keyword Match 'left'/'l'**: Mapped to `SWIPE_LEFT`.
3. **Keyword Match 'right'/'r'**: Mapped to `SWIPE_RIGHT`.
4. **Keyword Match 'roll'/'fwd'/'circular'**: Mapped to `ROLL_FWD`.
5. **Unmatched / Default Fallback**: Mapped to `STOP_SIGN`.

## Logged Fallback Events during Smoke Test

No fallback events occurred during the test; all outputs matched canonical labels exactly.
