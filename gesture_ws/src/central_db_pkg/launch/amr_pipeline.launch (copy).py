# ~/amr_gesture_ws/src/central_db_pkg/launch/amr_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def node_delay(seconds, node):
    return TimerAction(period=seconds, actions=[node])

def generate_launch_description():
    # Arguments (easy overrides at launch time)
    cam_dev   = DeclareLaunchArgument('video_device',    default_value='/dev/video0')
    cam_size  = DeclareLaunchArgument('image_size',      default_value='[320, 240]')
    cam_fmt   = DeclareLaunchArgument('pixel_format',    default_value='YUYV')   # stable path; MJPG→rgb8 can crash
    cam_enc   = DeclareLaunchArgument('output_encoding', default_value='rgb8')
    cam_fps   = DeclareLaunchArgument('frame_rate',      default_value='30')

    sim_in    = DeclareLaunchArgument('simplify_in',     default_value='/image_raw')
    sim_out   = DeclareLaunchArgument('simplify_out',    default_value='/image_raw_30hz')
    sim_hz    = DeclareLaunchArgument('target_hz',       default_value='30')
    sim_w     = DeclareLaunchArgument('resize_width',    default_value='320')
    sim_h     = DeclareLaunchArgument('resize_height',   default_value='240')

    kp_src    = DeclareLaunchArgument('kp_source',       default_value='ros')
    kp_topic  = DeclareLaunchArgument('kp_image_topic',  default_value=LaunchConfiguration('simplify_out'))
    kp_stride = DeclareLaunchArgument('kp_stride',       default_value='1')
    kp_debug  = DeclareLaunchArgument('kp_debug',        default_value='false')

    weights   = DeclareLaunchArgument('lstm_weights',    default_value=f'{"/home/sayak/amr_gesture_ws/models/lstm/jester20b_12cls/model_30x84.onnx"}')
    labels    = DeclareLaunchArgument('lstm_labels',     default_value=f'{"/home/sayak/amr_gesture_ws/models/lstm/jester20b_12cls/labels.txt"}')
    norm      = DeclareLaunchArgument('lstm_normalizer', default_value=f'{"/home/sayak/amr_gesture_ws/models/lstm/jester20b_12cls/normalizer.json"}')
    model_T   = DeclareLaunchArgument('lstm_model_frames', default_value='30')
    min_T     = DeclareLaunchArgument('lstm_min_frames',   default_value='12')
    conf_thr  = DeclareLaunchArgument('lstm_conf_threshold', default_value='0.80')
    bg_label  = DeclareLaunchArgument('lstm_bg_label',      default_value='NO_GESTURE')
    bg_as_unk = DeclareLaunchArgument('lstm_bg_as_unknown', default_value='true')

    # Recorder defaults
    rec_image   = DeclareLaunchArgument('rec_image_topic', default_value=LaunchConfiguration('simplify_out'))
    rec_buf_s   = DeclareLaunchArgument('rec_buffer_seconds', default_value='4.0')
    rec_fps     = DeclareLaunchArgument('rec_target_fps', default_value='30')
    rec_w       = DeclareLaunchArgument('rec_width', default_value='320')
    rec_h       = DeclareLaunchArgument('rec_height', default_value='240')
    rec_outdir  = DeclareLaunchArgument('rec_out_dir', default_value='/home/sayak/amr_gesture_ws/data/clips')
    rec_pubjson = DeclareLaunchArgument('rec_publish_json_topic', default_value='/recorder/clip_ready')
    rec_preset  = DeclareLaunchArgument('rec_preset', default_value='veryfast')
    rec_crf     = DeclareLaunchArgument('rec_crf', default_value='23')

    # Bridge defaults
    br_timeout  = DeclareLaunchArgument('confirm_timeout_s', default_value='6.0')
    br_req_s    = DeclareLaunchArgument('request_seconds', default_value='2.0')
    br_keep_min = DeclareLaunchArgument('keep_clips_minutes', default_value='30')
    br_min_conf = DeclareLaunchArgument('min_vlm_conf', default_value='0.0')
    br_ui_req   = DeclareLaunchArgument('ui_request_topic', default_value='/vlm/confirm_request')
    br_ui_rep   = DeclareLaunchArgument('ui_reply_topic',   default_value='/ui/confirm_reply')
    br_rec_req  = DeclareLaunchArgument('recorder_request_topic', default_value='/recorder/request')
    br_rec_rdy  = DeclareLaunchArgument('recorder_ready_topic',   default_value='/recorder/clip_ready')

    # Environment: fixed domain + FastVLM on CPU by default (safe on your RTX 5070 Blackwell)
    set_domain = SetEnvironmentVariable(name='ROS_DOMAIN_ID', value='7')
    set_model  = SetEnvironmentVariable(name='VLM_MODEL_ID',  value='apple/FastVLM-1.5B')
    force_cpu  = SetEnvironmentVariable(name='CUDA_VISIBLE_DEVICES', value='')  # comment this if you run nightly CUDA

    # 1) CAMERA
    cam = Node(
        package='v4l2_camera', executable='v4l2_camera_node', output='screen',
        parameters=[{
            'video_device':       LaunchConfiguration('video_device'),
            'image_size':         LaunchConfiguration('image_size'),
            'pixel_format':       LaunchConfiguration('pixel_format'),
            'output_encoding':    LaunchConfiguration('output_encoding'),
            'frame_rate':         LaunchConfiguration('frame_rate'),
        }]
    )

    # 2) FRAME SIMPLIFIER
    simplifier = Node(
        package='frame_simplifier_pkg', executable='simplifier_node', output='screen',
        parameters=[{
            'in_topic':    LaunchConfiguration('simplify_in'),
            'out_topic':   LaunchConfiguration('simplify_out'),
            'target_hz':   LaunchConfiguration('target_hz'),
            'resize_width':  LaunchConfiguration('resize_width'),
            'resize_height': LaunchConfiguration('resize_height'),
        }]
    )

    # 3) KEYPOINT EXTRACTOR
    keypoints = Node(
        package='keypoint_extractor_pkg', executable='keypoint_extractor_node', output='screen',
        parameters=[{
            'source':           LaunchConfiguration('kp_source'),
            'image_topic':      LaunchConfiguration('kp_image_topic'),
            'stride':           LaunchConfiguration('kp_stride'),
            'debug':            LaunchConfiguration('kp_debug'),
            'publish_annotated': True,
            'annotated_topic':  '/keypoints_annotated',
        }]
    )

    # 4) LSTM ONNX
    lstm = Node(
        package='gesture_classifiers_pkg', executable='lstm_onnx_node', output='screen',
        parameters=[{
            'weights_path':   LaunchConfiguration('lstm_weights'),
            'labels_path':    LaunchConfiguration('lstm_labels'),
            'normalizer_path':LaunchConfiguration('lstm_normalizer'),
            'model_frames':   LaunchConfiguration('lstm_model_frames'),
            'min_frames':     LaunchConfiguration('lstm_min_frames'),
            'conf_threshold': LaunchConfiguration('lstm_conf_threshold'),
            'background_label': LaunchConfiguration('lstm_bg_label'),
            'treat_background_as_unknown': LaunchConfiguration('lstm_bg_as_unknown'),
        }]
    )

    # 5) RECORDER (rolling buffer, emits /recorder/clip_ready)
    recorder = Node(
        package='vlm_recorder_pkg', executable='recorder_node', output='screen',
        parameters=[{
            'image_topic':        LaunchConfiguration('rec_image_topic'),
            'buffer_seconds':     LaunchConfiguration('rec_buffer_seconds'),
            'target_fps':         LaunchConfiguration('rec_target_fps'),
            'width':              LaunchConfiguration('rec_width'),
            'height':             LaunchConfiguration('rec_height'),
            'out_dir':            LaunchConfiguration('rec_out_dir'),
            'publish_json_topic': LaunchConfiguration('rec_publish_json_topic'),
            'preset':             LaunchConfiguration('rec_preset'),
            'crf':                LaunchConfiguration('rec_crf'),
        }]
    )

    # 6) VLM (FastVLM service /vlm/infer)
    vlm = Node(
        package='vlm_ros', executable='vlm_node', output='screen',
        parameters=[{}]
    )

    # 7) VLM BRIDGE (unknown → recorder → /vlm/infer → UI → /intents)
    bridge = Node(
        package='vlm_bridge_pkg', executable='bridge_node', output='screen',
        parameters=[{
            'confirm_timeout_s':        LaunchConfiguration('confirm_timeout_s'),
            'request_seconds':          LaunchConfiguration('request_seconds'),
            'keep_clips_minutes':       LaunchConfiguration('keep_clips_minutes'),
            'min_vlm_conf':             LaunchConfiguration('min_vlm_conf'),
            'ui_request_topic':         LaunchConfiguration('ui_request_topic'),
            'ui_reply_topic':           LaunchConfiguration('ui_reply_topic'),
            'recorder_request_topic':   LaunchConfiguration('recorder_request_topic'),
            'recorder_ready_topic':     LaunchConfiguration('recorder_ready_topic'),
        }]
    )

    # 8) UI KIOSK
    ui = Node(
        package='ui_kiosk_pkg', executable='ui_kiosk_node', output='screen'
    )

    # 9) CENTRAL DB
    db = Node(
        package='central_db_pkg', executable='central_db_node', output='screen'
    )

    # Light staging so downstream nodes see topics
    return LaunchDescription([
        # Args
        cam_dev, cam_size, cam_fmt, cam_enc, cam_fps,
        sim_in, sim_out, sim_hz, sim_w, sim_h,
        kp_src, kp_topic, kp_stride, kp_debug,
        weights, labels, norm, model_T, min_T, conf_thr, bg_label, bg_as_unk,
        rec_image, rec_buf_s, rec_fps, rec_w, rec_h, rec_outdir, rec_pubjson, rec_preset, rec_crf,
        br_timeout, br_req_s, br_keep_min, br_min_conf, br_ui_req, br_ui_rep, br_rec_req, br_rec_rdy,

        # Env
        set_domain, set_model, force_cpu,

        # Order: camera → simplifier → keypoints → lstm → recorder → vlm → bridge → ui/db
        cam,
        node_delay(1.0, simplifier),
        node_delay(2.0, keypoints),
        node_delay(2.5, lstm),
        node_delay(3.0, recorder),
        node_delay(3.5, vlm),
        node_delay(4.0, bridge),
        node_delay(4.0, ui),
        node_delay(4.0, db),
    ])

