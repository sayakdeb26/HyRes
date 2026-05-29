builtin_interfaces/Time stamp
string session_id        # unique ID for this gesture session (generated at LSTM)
int32  window_id         # ID of the LSTM window that produced this event
string label             # best-guess label from LSTM, e.g. UNKNOWN or a class name
float32 confidence       # LSTM probability for that label
string hint              # optional textual hint, may be empty
string source            # e.g. "lstm"
