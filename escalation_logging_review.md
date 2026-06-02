# Escalation Logging Review Report

This report evaluates how pre-escalation predictions (top-1 LSTM labels and confidence scores) are captured and logged during uncertainty-triggered escalations.

## 1. Audit Findings
- **Initial Implementation**: 
  - For escalated samples (LSTM confidence $\le 0.60$), the `lstm_inference_node.py` discarded the actual top-1 LSTM predicted label and populated the `UnknownGesture.label` field with the hardcoded string `"UNCERTAIN"`.
  - As a result, both the operator hint and the primary escalation log (`hybrid_escalation_log.csv`) registered only `"UNCERTAIN"` instead of the actual LSTM guess.
- **Why this was a Gap**: 
  - This prevented detailed post-hoc analysis. We could not verify *what* the LSTM predicted before getting confused, making it impossible to perform confusion matrix analysis on pre-escalation guesses.

## 2. Implemented Correction
We modified the sequence callback in `lstm_inference_node.py`:
- **Change**: When confidence falls below the parameter threshold, we store the actual predicted label string (e.g. `"Swipe Left"`, `"Stop Sign"`) inside `label_str` and set `msg_unknown.label = label_str`.
- **Outcome**: 
  1. The VLM bridge node now receives the actual top-1 prediction.
  2. The actual top-1 LSTM prediction and its corresponding confidence are correctly logged in the columns `lstm_prediction` and `confidence` of `hybrid_escalation_log.csv`.
  3. Pre-escalation predictions are fully preserved, enabling comparative confusion matrix calculations.
