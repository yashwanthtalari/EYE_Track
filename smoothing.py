class GazeSmoother:
    """
    Stabilizes cursor movement using standard Exponential Moving Average (EMA)
    and an adaptive smoothing filter that dynamically adjusts lag vs. jitter.
    """
    def __init__(self, alpha=0.15, use_adaptive=True, threshold_px=50.0):
        self.alpha_default = alpha
        self.use_adaptive = use_adaptive
        self.threshold_px = threshold_px
        
        self.smoothed_x = None
        self.smoothed_y = None

    def smooth(self, x, y):
        """
        Applies smoothing to input coordinates (x, y).
        Returns smoothed (x, y) coordinates.
        """
        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = x
            self.smoothed_y = y
            return x, y

        if self.use_adaptive:
            # Calculate distance between new point and previous smoothed point
            dist = ((x - self.smoothed_x) ** 2 + (y - self.smoothed_y) ** 2) ** 0.5
            
            # Dynamic alpha calculation:
            # If distance is small (jitter), alpha is small (heavy smoothing).
            # If distance is large (saccade/jump), alpha is large (fast response, low latency).
            if dist > self.threshold_px:
                # Fast catch-up
                alpha = min(0.8, self.alpha_default + (dist - self.threshold_px) / 200.0)
            else:
                # High stability
                alpha = max(0.05, self.alpha_default * (dist / self.threshold_px))
        else:
            alpha = self.alpha_default

        # Exponential moving average formula
        self.smoothed_x = alpha * x + (1 - alpha) * self.smoothed_x
        self.smoothed_y = alpha * y + (1 - alpha) * self.smoothed_y

        return self.smoothed_x, self.smoothed_y

    def reset(self):
        self.smoothed_x = None
        self.smoothed_y = None
