# modules/executor/frame_processor.py
"""
Iterator wrapper around a video file that yields frames at a controlled rate.

  - Supports real-time frame skipping (realtime_skip=True) so slow inference
    does not fall behind wall-clock video time.
  - Sampling is controlled by target_fps; frames are read every N source frames.

Usage:
    processor = FrameProcessor(video_path, target_fps=10.0)
    for frame_data in processor:
        frame = frame_data["frame"]
        # ... process frame ...
        processor.report_processing_time(elapsed)
    processor.release()
"""

import cv2
import time
from typing import Dict, Any


class FrameProcessor:
    def __init__(self, video_path: str, realtime_skip: bool = True, target_fps: float = 10.0):
        """
        Args:
            video_path:    Path to the input video file.
            realtime_skip: If True, skip frames to stay aligned with wall-clock time.
            target_fps:    Maximum inference rate (capped at the source video FPS).
        """
        self.video_path = video_path
        self.realtime_skip = realtime_skip
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps

        # Clamp target_fps to the source FPS; compute the read stride.
        self.target_fps = min(target_fps, self.fps)
        self._frame_interval = max(1, round(self.fps / self.target_fps))

        # Iteration state.
        self._current_frame_id = 0
        self._last_proc_time = 0.0
        self._inference_count = 0
        self._start_time = None
        self._next_read_pos = 0  # Next frame index to read from the capture.
        self._previous_video_time_sec = None
        self._schedule_records = []
        self._stride_skipped_frames = 0
        self._latency_skipped_frames = 0

        print(f"[FrameProcessor] Video loaded: {video_path}")
        print(f"  Resolution: {self.width}x{self.height} | "
              f"FPS: {self.fps:.1f} (target: {self.target_fps:.1f}, interval: {self._frame_interval}) | "
              f"Frames: {self.total_frames} | Duration: {self.duration:.1f}s")

    # ================================================================
    # Iterator protocol
    # ================================================================
    def __iter__(self):
        self._start_time = time.time()
        self._next_read_pos = 0
        self._previous_video_time_sec = None
        self._schedule_records = []
        self._stride_skipped_frames = 0
        self._latency_skipped_frames = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """
        Returns the next sampled frame, skipping ahead if real-time mode is active.

        Returns:
            {
                "frame": np.ndarray (BGR, H x W x 3),
                "frame_id": int,
                "video_time_sec": float,
                "fps": float,
                "progress_pct": float,
                "inference_count": int,
            }
        """
        scheduled_read_pos = self._next_read_pos
        selected_read_pos = scheduled_read_pos

        # In real-time mode, advance to the frame that matches wall-clock elapsed time.
        if self.realtime_skip and self._start_time is not None:
            wall_elapsed = time.time() - self._start_time
            target_frame = int(wall_elapsed * self.fps)
            # Snap to the nearest sampling boundary.
            target_frame = (target_frame // self._frame_interval) * self._frame_interval
            selected_read_pos = max(scheduled_read_pos, target_frame)
            self._next_read_pos = selected_read_pos

        # End of video.
        if self._next_read_pos >= self.total_frames:
            raise StopIteration

        # Seek only when we need to jump forward (avoids unnecessary seeks).
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if self._next_read_pos > current_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._next_read_pos)

        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        # OpenCV's reported position after a seek is codec dependent. Keep the
        # requested source position as the canonical schedule identity so every
        # downstream variant can replay the exact same frame/timestamp pair.
        source_frame_zero_based = int(selected_read_pos)
        self._current_frame_id = source_frame_zero_based + 1
        video_time_sec = source_frame_zero_based / self.fps
        self._inference_count += 1

        if self._previous_video_time_sec is None:
            delta_t_sec = 0.0
            stride_skipped_frames = 0
            latency_skipped_frames = 0
        else:
            delta_t_sec = video_time_sec - self._previous_video_time_sec
            # The scheduled stride is deliberate downsampling; any additional
            # jump beyond it is caused by the wall-clock-aware catch-up path.
            stride_skipped_frames = max(0, self._frame_interval - 1)
            latency_skipped_frames = max(0, selected_read_pos - scheduled_read_pos)

        self._stride_skipped_frames += stride_skipped_frames
        self._latency_skipped_frames += latency_skipped_frames
        self._previous_video_time_sec = video_time_sec

        # Schedule the next read position based on the sampling stride.
        self._next_read_pos = source_frame_zero_based + self._frame_interval

        progress = 100.0 * self._current_frame_id / self.total_frames

        frame_record = {
            "frame": frame,
            "frame_id": self._current_frame_id,
            "source_frame_zero_based": source_frame_zero_based,
            "video_time_sec": round(video_time_sec, 6),
            "delta_t_sec": round(delta_t_sec, 6),
            "stride_skipped_frames": int(stride_skipped_frames),
            "latency_skipped_frames": int(latency_skipped_frames),
            "fps": self.fps,
            "progress_pct": round(progress, 1),
            "inference_count": self._inference_count,
        }
        self._schedule_records.append({
            "frame_index": self._current_frame_id,
            "source_frame_zero_based": source_frame_zero_based,
            "video_timestamp_sec": round(video_time_sec, 6),
            "delta_t_sec": round(delta_t_sec, 6),
            "stride_skipped_frames": int(stride_skipped_frames),
            "latency_skipped_frames": int(latency_skipped_frames),
            "total_skipped_frames": int(stride_skipped_frames + latency_skipped_frames),
            "target_fps": self.target_fps,
            "source_fps": self.fps,
            "source_total_frames": self.total_frames,
            "realtime_skip": self.realtime_skip,
        })
        return frame_record

    # ================================================================
    # Timing feedback from the caller
    # ================================================================
    def report_processing_time(self, elapsed_sec: float):
        """ Records the most recent per-frame processing time (informational only). """
        self._last_proc_time = elapsed_sec

    # ================================================================
    # Progress display
    # ================================================================
    def print_progress(self, extra: str = ""):
        elapsed = time.time() - self._start_time if self._start_time else 0
        video_time = (self._current_frame_id - 1) / self.fps
        msg = (
            f"\r  [{self._current_frame_id}/{self.total_frames}] "
            f"{100*self._current_frame_id/self.total_frames:5.1f}% | "
            f"t={video_time:.1f}s | "
            f"infer={self._inference_count} | "
            f"wall={elapsed:.1f}s"
        )
        if extra:
            msg += f" | {extra}"
        print(msg, end="", flush=True)

    # ================================================================
    # Video info
    # ================================================================
    def get_video_info(self) -> dict:
        return {
            "video_path": self.video_path,
            "fps": self.fps,
            "target_fps": self.target_fps,
            "realtime_skip": self.realtime_skip,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration, 2),
        }

    def get_run_stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 0
        tail_unprocessed_frames = max(0, self.total_frames - self._current_frame_id)
        return {
            "total_inferences": self._inference_count,
            "wall_time_sec": round(elapsed, 2),
            "avg_proc_time_sec": round(elapsed / max(self._inference_count, 1), 4),
            "frames_processed_ratio": f"{self._inference_count}/{self.total_frames}",
            "stride_skipped_frames": self._stride_skipped_frames,
            "latency_skipped_frames": self._latency_skipped_frames,
            "total_skipped_frames": self._stride_skipped_frames + self._latency_skipped_frames,
            "tail_unprocessed_frames": tail_unprocessed_frames,
            "total_unprocessed_frames": max(0, self.total_frames - self._inference_count),
        }

    def get_schedule_records(self) -> list:
        """Return the immutable frame schedule observed during this execution."""
        return list(self._schedule_records)

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        stats = self.get_run_stats()
        print(f"\n[FrameProcessor] Released. Stats: {stats}")

    def __del__(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
