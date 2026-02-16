#!/bin/bash
# build.sh — Build, sign, and package HabitTracker.app into a DMG.
#
# ROOT CAUSE WHY CAMERA DIDN'T WORK:
#   PyInstaller's BUNDLE step re-signs the .app without --options runtime,
#   dropping the hardened-runtime flag. macOS 12+ (and especially Sequoia)
#   silently refuses AVFoundation camera access for apps that lack hardened
#   runtime, even when TCC has "Allow" on record. The post-build codesign
#   below re-applies the signature with the correct flags and entitlements.
#
# Usage: bash build.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="HabitTracker"
APP_BUNDLE="dist/${APP_NAME}.app"
DMG_PATH="dist/${APP_NAME}.dmg"
ENTITLEMENTS="entitlements.plist"

# ── 1. Build ──────────────────────────────────────────────────────────────────
echo "==> Building ${APP_NAME} with PyInstaller..."
pyinstaller "${APP_NAME}.spec" --noconfirm

# ── 2. Re-sign the entire .app with hardened runtime + entitlements ───────────
# PyInstaller's BUNDLE step signs without --options runtime, which breaks
# camera (and mic) access on macOS 12+. We fix that here.
echo "==> Re-signing ${APP_BUNDLE} with hardened runtime..."
codesign \
    --force \
    --deep \
    --sign - \
    --options runtime \
    --entitlements "${ENTITLEMENTS}" \
    "${APP_BUNDLE}"

# ── 3. Verify ─────────────────────────────────────────────────────────────────
echo "==> Verifying signature..."
codesign --verify --deep --strict "${APP_BUNDLE}" && echo "    Signature OK"
echo "    Entitlements:"
codesign -d --entitlements :- "${APP_BUNDLE}" 2>&1 | grep -A 20 '<dict>'

# ── 4. Package into a DMG ─────────────────────────────────────────────────────
echo "==> Creating ${DMG_PATH}..."
rm -f "${DMG_PATH}"
hdiutil create \
    -volname "${APP_NAME}" \
    -srcfolder "${APP_BUNDLE}" \
    -ov \
    -format UDZO \
    "${DMG_PATH}"

echo ""
echo "Done!  →  ${DMG_PATH}"
