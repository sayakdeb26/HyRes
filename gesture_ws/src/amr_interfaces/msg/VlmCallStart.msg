# VlmCallStart.msg
# Published by bridge_node when VLM inference is initiated
# Used by metrics_logger to capture accurate t_vlm_request_ns

builtin_interfaces/Time stamp
string session_id
string clip_path
