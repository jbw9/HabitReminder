#!/usr/bin/env python3
"""
Camera Background Thread
Captures frames, processes them through DetectorManager, sends alerts
via queue, and optionally prepares annotated preview frames.
"""

import subprocess
import threading
import time
import cv2

from app.preview_window import draw_overlays


def _request_avfoundation_camera_permission():
    """
    Explicitly request camera authorization via the AVFoundation API and
    block until the user responds (or 60 s elapses).

    Returns True if the camera is authorized, False if denied/restricted.
    Falls back to True on any unexpected error so that cv2 still gets a
    chance to open the device.

    WHY THIS EXISTS:
      OpenCV's AVFoundation backend does NOT call
      -[AVCaptureDevice requestAccessForMediaType:completionHandler:] before
      starting a capture session.  On macOS 12+ (and especially Sequoia),
      if the session is created before TCC authorization is confirmed, the
      OS delivers only black / zero frames — even after the user later clicks
      "Allow" — because the *existing* session was started in an unauthorized
      state.  By requesting and awaiting authorization here first, we ensure
      the capture session is opened only when access is genuinely granted.
    """
    try:
        import objc
        from Foundation import NSBundle

        avf_path = '/System/Library/Frameworks/AVFoundation.framework'
        avf_bundle = NSBundle.bundleWithPath_(avf_path)
        if not avf_bundle.isLoaded():
            avf_bundle.load()

        AVCaptureDevice = objc.lookUpClass('AVCaptureDevice')
        AVMediaTypeVideo = 'vide'  # NSString constant for video

        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        # 0 = not determined, 1 = restricted, 2 = denied, 3 = authorized
        if status == 3:
            return True   # already authorized — open camera immediately
        if status in (1, 2):
            return False  # restricted / denied — don't bother opening

        # status == 0: show the TCC dialog and wait up to 60 s for a response
        event = threading.Event()
        result = [False]

        def handler(granted):
            result[0] = bool(granted)
            event.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeVideo, handler
        )
        event.wait(timeout=60)
        return result[0]

    except Exception:
        # If anything goes wrong with the ObjC bridge, fall through and
        # let cv2 try to open the device anyway (old behaviour).
        return True

# Preview frame size (for inline menu bar preview)
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180


class CameraThread:
    """Background thread for camera capture and detector processing."""

    def __init__(self, detector_manager, alert_queue, fps=30,
                 camera_width=640, camera_height=480):
        self.detector_manager = detector_manager
        self.alert_queue = alert_queue
        self.fps = fps
        self.camera_width = camera_width
        self.camera_height = camera_height

        self._thread = None
        self._running = False
        self._camera = None
        self._frame_timestamp_ms = 0
        self._initializing = False  # True while _initialize_camera() is running

        # Preview state — main thread reads latest_preview_frame
        self._preview_enabled = False
        self.latest_preview_frame = None  # BGR numpy array (320x180)

    @property
    def running(self):
        return self._running

    @property
    def preview_enabled(self):
        return self._preview_enabled

    def set_preview(self, enabled):
        """Toggle preview frame generation on/off."""
        self._preview_enabled = enabled
        if not enabled:
            self.latest_preview_frame = None

    def start(self):
        """Start the camera thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the camera thread and release resources."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.latest_preview_frame = None

    def _initialize_camera(self):
        # Ask for authorization BEFORE opening any capture session.
        # cv2's AVFoundation backend does not do this itself; without it,
        # macOS returns only black frames until the existing session is
        # torn down and rebuilt — which is too late if the user already
        # responded to a dialog that was attached to a dead session.
        authorized = _request_avfoundation_camera_permission()
        if not authorized:
            raise RuntimeError(
                "Camera access denied. Enable it in "
                "System Settings → Privacy & Security → Camera."
            )

        # Use AVFoundation explicitly — the backend macOS ties camera
        # permissions to. Verify a test read succeeds before accepting the
        # device; some indices (e.g. 0 for Continuity Camera stub) report
        # isOpened()=True but deliver no frames at all.
        for index in [0, 1]:
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                    self._camera = cap
                    return
            cap.release()
        raise RuntimeError("Could not open camera")

    def _release_camera(self):
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _run(self):
        """Main camera loop (background thread)."""
        self._initializing = True
        try:
            self._initialize_camera()
        except RuntimeError as e:
            self.alert_queue.put({
                'detector': 'system',
                'message': str(e),
                'severity': 'high',
            })
            self._running = False
            return
        finally:
            self._initializing = False

        frame_delay = 1.0 / self.fps
        dead_frames = 0  # consecutive failed/black frames
        ever_had_valid_frame = False  # True once we've seen real camera data

        while self._running:
            loop_start = time.time()

            # Guard: if _initialize_camera() failed inside the dead-frame
            # recovery path, self._camera will be None.  Treat that as a
            # dead frame so the recovery logic retries after the threshold.
            if self._camera is None:
                ret, frame = False, None
            else:
                ret, frame = self._camera.read()

            # A frame is "dead" if read failed or it's a black buffer
            # (happens when camera opened before permission was granted).
            is_dead = not ret or frame is None or frame.sum() == 0
            if is_dead:
                dead_frames += 1
                # Recover in two situations:
                #   1. Camera WAS working then died (permission revoked,
                #      hardware disconnect, AVFoundation hiccup) — retry
                #      after 90 consecutive dead frames (~3 s at 30 fps).
                #   2. Camera opened but NEVER delivered a valid frame —
                #      this happens after a rapid close/reopen (e.g. the
                #      auto-restart on first launch) when AVFoundation
                #      needs more time to settle.  Wait 5 s (150 frames)
                #      before retrying so we don't hammer the driver, but
                #      DO retry — without this the thread would spin
                #      forever and the preview would stay permanently blank.
                #
                # Note: the original "ever_had_valid_frame" guard was meant
                # to avoid reopening during a TCC permission dialog.  That
                # guard is no longer needed here because
                # _request_avfoundation_camera_permission() blocks until the
                # dialog is resolved, so by the time we reach this loop TCC
                # is already settled and it is safe to retry.
                worked_then_died = ever_had_valid_frame and dead_frames >= 90
                startup_stalled = (not ever_had_valid_frame) and dead_frames >= 150
                if worked_then_died or startup_stalled:
                    self._release_camera()
                    try:
                        self._initialize_camera()
                    except RuntimeError:
                        pass
                    dead_frames = 0
                time.sleep(frame_delay)
                continue
            ever_had_valid_frame = True
            dead_frames = 0

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Wall-clock timestamp — always monotonically increasing even
            # when the camera thread is stopped and restarted.
            self._frame_timestamp_ms = int(time.time() * 1000)

            # Run detectors
            alerts = self.detector_manager.process_frame(
                rgb_frame, self._frame_timestamp_ms
            )
            for alert in alerts:
                # Play audio immediately on the camera thread — no queue delay
                subprocess.Popen(
                    ['afplay', '/System/Library/Sounds/Sosumi.aiff'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.alert_queue.put(alert)

            # Build preview frame if enabled
            if self._preview_enabled:
                mouth_counter, mouth_threshold = self.detector_manager.get_mouth_counter()
                nail_counter, nail_threshold = self.detector_manager.get_nail_counter()
                annotated = draw_overlays(
                    frame,
                    self.detector_manager.last_face_landmarks,
                    self.detector_manager.last_hand_landmarks,
                    self.detector_manager.get_all_statuses(),
                    self.detector_manager.enabled_detector_keys(),
                    mouth_counter=mouth_counter,
                    mouth_threshold=mouth_threshold,
                    nail_counter=nail_counter,
                    nail_threshold=nail_threshold,
                    fps=self.fps,
                )
                small = cv2.resize(annotated, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                self.latest_preview_frame = small

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._release_camera()
