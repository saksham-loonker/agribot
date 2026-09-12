#!/usr/bin/env bash
#
# Agribot Setup Wizard
#
# Guides farmers and field technicians through setting up Agribot for daily use.
# Checks prerequisites, builds the Android app, prepares model bundles, and
# starts the local dashboard.
#
# Usage:
#   ./21_setup_wizard.sh
#
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[ERROR]${NC} $1"; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANDROID_DIR="$REPO_ROOT/agribot_android_app/android"

echo ""
echo "============================================"
echo "  Agribot Setup Wizard"
echo "  Field Deployment & Daily Use Setup"
echo "============================================"
echo ""

# Step 1: Check prerequisites
echo ""
info "Step 1: Checking prerequisites..."

# Check JDK
if command -v java &>/dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -1 | awk -F '"' '{print $2}')
    ok "JDK found: $JAVA_VERSION"
else
    fail "JDK not found. Install JDK 17 or newer."
    exit 1
fi

# Check Python
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    ok "Python found: $PYTHON_VERSION"
else
    fail "Python 3 not found. Install Python 3.8+."
    exit 1
fi

# Check Android SDK
if [ -n "${ANDROID_HOME:-}" ]; then
    ok "ANDROID_HOME: $ANDROID_HOME"
else
    warn "ANDROID_HOME not set. Attempting to set default..."
    if [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_HOME="$HOME/Android/Sdk"
        ok "Set ANDROID_HOME to $ANDROID_HOME"
    else
        fail "Android SDK not found. Set ANDROID_HOME or install Android SDK."
        exit 1
    fi
fi

# Check gradlew
if [ -f "$ANDROID_DIR/gradlew" ]; then
    ok "Gradle wrapper found"
else
    fail "Gradle wrapper not found at $ANDROID_DIR/gradlew"
    exit 1
fi

# Step 2: Prepare model bundle
echo ""
info "Step 2: Preparing model bundle..."

cd "$REPO_ROOT"
if [ -f "46_prepare_android_model_bundle.py" ]; then
    python3 46_prepare_android_model_bundle.py --json 2>/dev/null || {
        warn "Model bundle preparation had issues. Continuing..."
    }
    ok "Model bundle prepared"
else
    warn "Model bundle preparation script not found. Skipping."
fi

# Step 3: Build Android APK
echo ""
info "Step 3: Building Android APK..."

cd "$ANDROID_DIR"
./gradlew :app:assembleDebug --no-daemon --max-workers=1 -q 2>/dev/null || {
    fail "APK build failed. Check the build output above."
    exit 1
}
APK_PATH="$ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk"
if [ -f "$APK_PATH" ]; then
    ok "APK built successfully: $APK_PATH"
else
    fail "APK not found at expected path."
    exit 1
fi

# Step 4: Install APK (if device connected)
echo ""
info "Step 4: Checking for connected Android device..."

if command -v adb &>/dev/null; then
    DEVICE_COUNT=$(adb devices | grep -c "device$" || true)
    if [ "$DEVICE_COUNT" -gt 0 ]; then
        ok "Android device detected"
        info "Installing APK to device..."
        adb install -r "$APK_PATH" 2>/dev/null && ok "APK installed to device" || warn "APK install failed. Install manually."
    else
        warn "No Android device connected. APK is ready at: $APK_PATH"
        info "To install manually: adb install -r $APK_PATH"
    fi
else
    warn "adb not found. APK is ready at: $APK_PATH"
fi

# Step 5: Start local dashboard
echo ""
info "Step 5: Starting local dashboard..."

cd "$REPO_ROOT"
if [ -f "agribot_platform/run_platform.py" ]; then
    info "Dashboard available at: http://localhost:8080"
    info "Starting dashboard in background..."
    python3 agribot_platform/run_platform.py &
    DASHBOARD_PID=$!
    ok "Dashboard started (PID: $DASHBOARD_PID)"
    warn "Dashboard is running in the background. Close this terminal or kill PID $DASHBOARD_PID to stop."
else
    warn "Dashboard script not found. Skipping."
fi

# Step 6: Summary
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Open the Agribot app on your phone"
echo "  2. Grant camera permission when prompted"
echo "  3. Follow the onboarding flow"
echo "  4. Configure your field layout (rows, plants per row)"
echo "  5. Start scanning with 'Walk & Scan' mode"
echo ""
echo "Dashboard: http://localhost:8080"
echo "APK: $APK_PATH"
echo ""
echo "For troubleshooting, see:"
echo "  - agribot_android_app/README_APK.md"
echo "  - deployment/17_archive_field_cleanup.sh (for field data management)"
echo ""
