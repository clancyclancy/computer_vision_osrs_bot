"""
Performance metrics aggregation and overlay utilities.

This module provides a lightweight tracker for real-time pipelines. It maintains
exponential moving averages (EMA) for:
- FPS (frames per second)
- Model inference time (s)
- Template matching time (s)
- Draw/display time (s)

Use PerformanceTracker to:
- Smooth noisy per-frame timings with a tunable alpha (smoothing factor)
- Update metrics incrementally during the main loop
- Render human-readable stats for on-screen overlays via get_overlay_stats()

Notes:
- Times are tracked in seconds internally; the overlay formats them in milliseconds.
- EMA initialization uses the first observed value to avoid cold-start bias.
"""


import time

class PerformanceTracker:
    """
    Tracks and calculates performance metrics for specific activities.

    This class facilitates the tracking of various performance components such as
    frames per second (FPS), model inference time, template match time, and display
    time using an exponential moving average. It is designed for scenarios where
    real-time or iterative updates of these metrics are required, providing efficient
    and summarized statistics.

    :ivar alpha: Smoothing factor for exponential moving average calculations.
    :ivar fps_avg: Averaged frames per second based on updates.
    :ivar model_inference_avg: Average model inference time, in seconds.
    :ivar template_avg: Average template match time, in seconds.
    :ivar display_avg: Average draw and display time, in seconds.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps_avg = 0.0
        self.model_inference_avg = 0.0
        self.template_avg = 0.0
        self.display_avg = 0.0

        self.time_of_new_action = 0.0
        self.clicked_object_label = "None"



    def update_display_time(self, display_time):
        self.display_avg = (1 - self.alpha) * self.display_avg + self.alpha * display_time

    def update_fps(self, fps):
        self.fps_avg = fps if self.fps_avg == 0 else (1 - self.alpha) * self.fps_avg + self.alpha * fps

    def update_model_time(self, t):
        self.model_inference_avg = t if self.model_inference_avg == 0 else (1 - self.alpha) * self.model_inference_avg + self.alpha * t

    def update_template_time(self, t):
        self.template_avg = t if self.template_avg == 0 else (1 - self.alpha) * self.template_avg + self.alpha * t

    def get_overlay_stats(self):
        return [
            f"{self.fps_avg:.1f}: FPS",
            "",
            f"{self.model_inference_avg * 1000:.1f} ms: Model Inference Time",
            f"{self.template_avg * 1000:.1f} ms: Template Match Time",
            f"{self.display_avg * 1000:.1f} ms:   Draw & Display Time",
            "",
            f"{time.perf_counter() - self.time_of_new_action: .1f} s: Time Since Last Action",
            f"{self.clicked_object_label}: Last Action Object"
        ]
