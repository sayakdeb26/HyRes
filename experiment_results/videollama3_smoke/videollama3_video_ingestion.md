# Video-LLaMA 3 Video Ingestion Design

Video-LLaMA 3 utilizes native video processing powered by `decord` under the hood. However, the benchmark dataset consists of individual image frame sequences (JPEGs) stored in subdirectories.

To support this natively without degrading video quality or violating temporal dependencies:
1. **On-the-fly MP4 Compilation**: For each sample, the frame sequence directory is compiled into a temporary H.264 MP4 file using `ffmpeg`.
   ```bash
   ffmpeg -y -f image2 -framerate 10 -i <input_dir>/%05d.jpg -c:v libx264 -pix_fmt yuv420p <output_path>
   ```
2. **Native Loader Integration**: The path to this temporary MP4 is passed directly to the `VideoLLaMA3Processor` conversation structure under type `"video"`.
3. **Temporal Ingestion Parameters**:
   - `fps`: 2 frames/sec
   - `max_frames`: 16 frames
   These settings strike a balance between temporal reasoning capability and memory efficiency.
4. **Cleanup**: The temporary MP4 files are removed immediately after inference to prevent disk bloat.
