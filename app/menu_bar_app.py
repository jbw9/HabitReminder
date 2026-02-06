#!/usr/bin/env python3
"""
Menu Bar Application
rumps-based menu bar app with an inline camera preview at the top
of the dropdown (rendered via an NSImageView in a custom NSMenuItem).
"""

import queue
import rumps
import objc

from AppKit import NSImageView, NSView, NSMakeRect
from Foundation import NSRunLoop, NSTimer, NSRunLoopCommonModes, NSObject

from app.detector_manager import DetectorManager
from app.camera_thread import CameraThread
from app.alert_system import AlertSystem
from app.preview_window import frame_to_nsimage

# Inline preview dimensions
PREVIEW_W = 320
PREVIEW_H = 180


# ── Timer target (NSObject subclass for NSTimer callback) ─────────────────
class _PreviewTimerTarget(NSObject):
    """NSObject subclass that receives NSTimer fire events."""

    def initWithCallback_(self, callback):
        self = objc.super(_PreviewTimerTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def timerFired_(self, timer):
        if self._callback:
            self._callback()


class HealthMonitorApp(rumps.App):
    """Mac menu bar application for health habit monitoring."""

    MENU_TO_DETECTOR = {
        'Mouth Breathing': 'mouth_breathing',
        'Blink Rate': 'blink_rate',
        'Eye Rubbing': 'eye_rubbing',
        'Face Touching': 'face_touching',
        'Hydration': 'hydration',
    }

    def __init__(self):
        super().__init__("HT", quit_button=None)

        # Core components
        self.detector_manager = DetectorManager()
        self.alert_queue = queue.Queue()
        self.alert_system = AlertSystem()
        self.camera_thread = CameraThread(self.detector_manager, self.alert_queue)

        self.detector_manager.initialize_mediapipe()
        self.detector_manager.initialize_detectors()

        # ── Inline preview ─────────────────────────────────────────────
        # We create a rumps MenuItem, then replace the underlying
        # NSMenuItem's view with an NSImageView so the live camera feed
        # appears at the top of the dropdown menu.
        self._preview_menu_item = rumps.MenuItem('')
        self._preview_enabled = False

        padding = 8
        container = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, PREVIEW_W + 2 * padding, PREVIEW_H + 2 * padding)
        )

        self._preview_image_view = NSImageView.alloc().initWithFrame_(
            NSMakeRect(padding, padding, PREVIEW_W, PREVIEW_H)
        )
        self._preview_image_view.setImageScaling_(2)  # proportionally scale

        # Rounded corners
        self._preview_image_view.setWantsLayer_(True)
        self._preview_image_view.layer().setCornerRadius_(8)
        self._preview_image_view.layer().setMasksToBounds_(True)

        container.addSubview_(self._preview_image_view)

        # Attach view to the underlying NSMenuItem (rumps reuses it as-is)
        self._preview_menu_item._menuitem.setView_(container)
        self._preview_menu_item._menuitem.setHidden_(True)

        # ── Menu ───────────────────────────────────────────────────────
        self.menu = [
            self._preview_menu_item,
            None,
            rumps.MenuItem('Mouth Breathing', callback=self._toggle_detector),
            rumps.MenuItem('Blink Rate', callback=self._toggle_detector),
            rumps.MenuItem('Eye Rubbing', callback=self._toggle_detector),
            rumps.MenuItem('Face Touching', callback=self._toggle_detector),
            rumps.MenuItem('Hydration', callback=self._toggle_detector),
            None,
            rumps.MenuItem('Show Camera Preview', callback=self._toggle_preview),
            rumps.MenuItem('Reset Hydration Timer', callback=self._reset_hydration),
            None,
            rumps.MenuItem('Quit', callback=self._quit),
        ]

        # ── Timers ─────────────────────────────────────────────────────
        self._alert_timer = rumps.Timer(self._check_alerts, 0.5)
        self._alert_timer.start()

        # Preview updater — runs at ~15 fps using a raw NSTimer scheduled
        # in NSRunLoopCommonModes so it fires even while the menu is open.
        self._preview_timer_target = _PreviewTimerTarget.alloc().initWithCallback_(
            self._update_preview_tick
        )
        self._preview_ns_timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 15,
            self._preview_timer_target,
            objc.selector(self._preview_timer_target.timerFired_, signature=b'v@:@'),
            None,
            True,
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._preview_ns_timer,
            NSRunLoopCommonModes,
        )

    # ── Detector toggles ───────────────────────────────────────────────

    def _toggle_detector(self, sender):
        detector_key = self.MENU_TO_DETECTOR.get(sender.title)
        if detector_key is None:
            return

        if sender.state:
            self.detector_manager.disable_detector(detector_key)
            sender.state = False
        else:
            self.detector_manager.enable_detector(detector_key)
            sender.state = True

        self._update_camera_state()

    def _update_camera_state(self):
        if self.detector_manager.any_enabled():
            if not self.camera_thread.running:
                self.camera_thread.start()
        else:
            if self.camera_thread.running:
                self.camera_thread.stop()
            if self._preview_enabled:
                self._set_preview(False)
                self.menu['Show Camera Preview'].state = False

    # ── Inline preview ─────────────────────────────────────────────────

    def _toggle_preview(self, sender):
        if sender.state:
            self._set_preview(False)
            sender.state = False
        else:
            if not self.camera_thread.running:
                rumps.notification(
                    "Habit Tracker", "Camera Not Running",
                    "Enable at least one detector first.",
                )
                return
            self._set_preview(True)
            sender.state = True

    def _set_preview(self, enabled):
        self._preview_enabled = enabled
        self.camera_thread.set_preview(enabled)
        self._preview_menu_item._menuitem.setHidden_(not enabled)

    def _update_preview_tick(self):
        """~15 Hz on the main thread — converts latest frame to NSImage.

        Called by raw NSTimer (no sender argument).
        """
        if not self._preview_enabled:
            return
        frame = self.camera_thread.latest_preview_frame
        if frame is None:
            return
        ns_image = frame_to_nsimage(frame)
        if ns_image is not None:
            self._preview_image_view.setImage_(ns_image)
            # Force the view to redraw immediately
            self._preview_image_view.setNeedsDisplay_(True)

    # ── Alerts ─────────────────────────────────────────────────────────

    def _check_alerts(self, _):
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                self.alert_system.send_alert(alert)
            except queue.Empty:
                break

    # ── Misc ───────────────────────────────────────────────────────────

    def _reset_hydration(self, _):
        self.detector_manager.reset_hydration_timer()
        rumps.notification(
            "Habit Tracker", "Hydration Timer Reset",
            "Timer has been reset. Stay hydrated!",
        )

    def _quit(self, _):
        self.camera_thread.stop()
        self.detector_manager.cleanup()
        rumps.quit_application()
