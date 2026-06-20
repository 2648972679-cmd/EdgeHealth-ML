# EdgeHealth iOS App

Real-time human activity recognition on iPhone using ONNX Runtime + CoreMotion sensors.

## How It Works

```
iPhone Sensors (50Hz)          ONNX Model (398 KB)            UI Display
┌──────────────────┐           ┌──────────────────┐      ┌──────────────┐
│ Accelerometer x3  │──┐        │  STFT (built-in)  │      │  🚶 Walking  │
│ Gyroscope x3      │  │ 6×128  │  → VisionEncoder  │      │  95.3%       │
└──────────────────┘  │ ────→  │  → IMUEncoder     │ ──→  │  ████████░░  │
                       │        │  → Fusion → 6 cls │      └──────────────┘
└─ 2.56s window ─────┘        └──────────────────┘
```

- **Zero preprocessing**: STFT spectrogram is built into the ONNX model. Raw sensor data goes in, activity label comes out.
- **Hidden WebView engine**: ONNX Runtime Web runs inference in a zero-pixel WebView. No native modules needed.
- **50Hz sampling**: Matches UCI HAR dataset, 128 samples per window (2.56 seconds).

## Build & Install (No Mac Required)

### Prerequisites

1. **Node.js** (v18+) — [download](https://nodejs.org)
2. **Expo account** — sign up at [expo.dev](https://expo.dev) (free)
3. **Apple Developer account** — $99/year at [developer.apple.com](https://developer.apple.com) (needed for TestFlight)

### Step 1: Install dependencies

```bash
cd ios-app
npm install
```

### Step 2: Login to Expo

```bash
npx expo login
```

### Step 3: Build iOS app in the cloud (EAS Build)

```bash
npx eas build --platform ios --profile production
```

This uploads your code to Expo's Mac servers. They compile it into an `.ipa` file (~15-20 minutes).

### Step 4: Install on iPhone

**Option A — TestFlight (recommended):**
```bash
npx eas submit --platform ios
```
Your app appears in TestFlight. Add testers via App Store Connect.

**Option B — Direct install:**
Download the `.ipa` from the Expo dashboard. Install via:
- **AltStore** (Windows) — free sideloading
- **Apple Configurator** (Mac)
- **Xcode** (Mac)

### Step 5 (Optional): Test on iPhone immediately

For quick testing without a full build:

1. Install **Expo Go** from App Store on your iPhone
2. Run `npx expo start` on your PC
3. Scan the QR code with your iPhone camera
4. App opens in Expo Go (⚠️ WebView may have limited ONNX support in Go — use EAS Build for full functionality)

## Project Structure

```
ios-app/
├── App.js              # Main app — sensor collection + UI + WebView bridge
├── app.json            # Expo configuration (permissions, bundle ID)
├── package.json        # Dependencies
├── metro.config.js     # Metro bundler config (allows .onnx files)
├── babel.config.js     # Babel config
├── eas.json            # EAS Build config (cloud compilation)
├── assets/
│   └── ios_model.onnx  # Trained model (398 KB)
└── README.md
```

## Model Details

| Property | Value |
|----------|-------|
| Input | `(batch, 6, 128)` — 3 accel + 3 gyro, 128 time steps |
| Output | `(batch, 6)` — logits for 6 activities |
| Architecture | STFT → MobileNetV3-Small + 3-layer 1D-CNN → Fusion MLP |
| Size | 398 KB (ONNX) |
| Accuracy | 87.6% (UCI HAR test set) |

## Customization

### Change sampling window

In `App.js`, modify `WINDOW_SIZE`:
- Smaller (e.g., 64) = faster response, lower accuracy
- Larger (e.g., 256) = slower response, higher accuracy

### Add new activities

1. Retrain the model with new classes
2. Re-export `ios_model.onnx`
3. Update `ACTIVITIES` array in `App.js`
