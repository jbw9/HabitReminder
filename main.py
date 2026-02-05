#!/usr/bin/env python3
"""
Habit Tracker - Mac Menu Bar App
Entry point for the application.
"""

from app.menu_bar_app import HealthMonitorApp


def main():
    app = HealthMonitorApp()
    app.run()


if __name__ == '__main__':
    main()
