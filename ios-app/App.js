import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet, Text, View, SafeAreaView, StatusBar, ScrollView,
} from 'react-native';
import { WebView } from 'react-native-webview';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import * as FileSystem from 'expo-file-system';
import { Asset } from 'expo-asset';

// ─── Constants ────────────────────────────────────────────────
const ACTIVITIES = [
  { name: 'Walking',              emoji: '🚶', color: '#4CAF50' },
  { name: 'Walking Upstairs',     emoji: '🏃', color: '#FF9800' },
  { name: 'Walking Downstairs',   emoji: '🏃', color: '#F44336' },
  { name: 'Sitting',              emoji: '🪑', color: '#2196F3' },
  { name: 'Standing',             emoji: '🧍', color: '#9C27B0' },
  { name: 'Laying',               emoji: '🛌', color: '#607D8B' },
];

const SAMPLE_RATE = 50;          // Hz — matches UCI HAR dataset
const WINDOW_SIZE = 128;         // samples per inference window
const SAMPLE_INTERVAL = 1000 / SAMPLE_RATE; // 20ms between samples

// ─── ONNX Inference HTML (runs in hidden WebView) ─────────────
const INFERENCE_HTML = `
<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js">
</script>
<script>
  let session = null;
  let modelBase64 = null;

  // Listen for messages from React Native
  window.addEventListener('message', async (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'load') {
      // Load model from base64
      try {
        const bytes = Uint8Array.from(atob(msg.model), c => c.charCodeAt(0));
        session = await ort.InferenceSession.create(bytes.buffer);
        window.ReactNativeWebView.postMessage(JSON.stringify({
          type: 'loaded', success: true
        }));
      } catch (e) {
        window.ReactNativeWebView.postMessage(JSON.stringify({
          type: 'loaded', success: false, error: e.message
        }));
      }
    }

    if (msg.type === 'infer' && session) {
      // msg.data: flat array of 768 floats [6 channels * 128 steps]
      const input = new Float32Array(msg.data);
      const tensor = new ort.Tensor('float32', input, [1, 6, 128]);
      const outputs = await session.run({ imu_signal: tensor });
      const logits = Array.from(outputs.logits.data);

      // Softmax to get probabilities
      const maxVal = Math.max(...logits);
      const expSum = logits.reduce((s, v) => s + Math.exp(v - maxVal), 0);
      const probs = logits.map(v => Math.exp(v - maxVal) / expSum);
      const argmax = probs.indexOf(Math.max(...probs));

      window.ReactNativeWebView.postMessage(JSON.stringify({
        type: 'result',
        activity: argmax,
        confidence: probs[argmax],
        probabilities: probs,
      }));
    }
  });

  // Signal ready
  window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'ready' }));
</script>
</body></html>`;

// ─── Main App ──────────────────────────────────────────────────
export default function App() {
  // State
  const [activity, setActivity] = useState(-1);
  const [confidence, setConfidence] = useState(0);
  const [probs, setProbs] = useState([0, 0, 0, 0, 0, 0]);
  const [sensorReady, setSensorReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [recording, setRecording] = useState(false);

  // Refs
  const webViewRef = useRef(null);
  const bufferRef = useRef([]);         // 6 * 128 = 768 floats
  const collectingRef = useRef(false);

  // ── Load model into WebView ────────────────────────────────
  useEffect(() => {
    async function loadModel() {
      try {
        // Load .onnx model via expo-asset (handles Metro bundling)
        const asset = Asset.fromModule(require('./assets/ios_model.onnx'));
        await asset.downloadAsync();  // copies to local filesystem
        const base64 = await FileSystem.readAsStringAsync(
          asset.localUri,
          { encoding: FileSystem.EncodingType.Base64 }
        );
        webViewRef.current?.postMessage(JSON.stringify({
          type: 'load', model: base64,
        }));
      } catch (e) {
        console.error('Model load error:', e.message);
      }
    }
    // Small delay to let WebView mount
    setTimeout(loadModel, 1000);
  }, []);

  // ── Handle WebView messages ────────────────────────────────
  const handleMessage = useCallback((event) => {
    const msg = JSON.parse(event.nativeEvent.data);
    switch (msg.type) {
      case 'ready':
        // WebView is mounted and ready to receive model
        break;
      case 'loaded':
        setModelReady(msg.success);
        if (!msg.success) console.error('ONNX load failed:', msg.error);
        break;
      case 'result':
        setActivity(msg.activity);
        setConfidence(msg.confidence);
        setProbs(msg.probabilities);
        break;
    }
  }, []);

  // ── Start / stop sensors ────────────────────────────────────
  useEffect(() => {
    let accSub, gyrSub;

    async function start() {
      // Request permissions & set update interval
      await Accelerometer.setUpdateInterval(SAMPLE_INTERVAL);
      await Gyroscope.setUpdateInterval(SAMPLE_INTERVAL);

      // We need a 50Hz sample rate. expo-sensors may not support exact 50Hz.
      // Best effort: set to fastest available (game mode ≈ 100Hz on iOS)
      Accelerometer.setUpdateInterval(16); // ~60Hz
      Gyroscope.setUpdateInterval(16);

      setSensorReady(true);
    }
    start();

    return () => {
      accSub?.remove();
      gyrSub?.remove();
    };
  }, [modelReady]);

  // ── Start collecting when model is ready ────────────────────
  useEffect(() => {
    if (!modelReady || !sensorReady) return;

    setRecording(true);
    bufferRef.current = [];
    collectingRef.current = true;

    let accBuffer = [[], [], []];  // [x, y, z] arrays
    let gyrBuffer = [[], [], []];
    let count = 0;

    const accSub = Accelerometer.addListener(({ x, y, z }) => {
      if (!collectingRef.current) return;
      accBuffer[0].push(x);
      accBuffer[1].push(y);
      accBuffer[2].push(z);
      checkBuffer();
    });

    const gyrSub = Gyroscope.addListener(({ x, y, z }) => {
      if (!collectingRef.current) return;
      gyrBuffer[0].push(x);
      gyrBuffer[1].push(y);
      gyrBuffer[2].push(z);
      checkBuffer();
    });

    function checkBuffer() {
      // Wait until we have >= 128 samples in each channel
      const minLen = Math.min(
        accBuffer[0].length, accBuffer[1].length, accBuffer[2].length,
        gyrBuffer[0].length, gyrBuffer[1].length, gyrBuffer[2].length
      );
      if (minLen < WINDOW_SIZE) return;

      // Build flat array: [acc_x*128, acc_y*128, acc_z*128, gyr_x*128, gyr_y*128, gyr_z*128]
      const flat = [
        ...accBuffer[0].slice(0, WINDOW_SIZE),
        ...accBuffer[1].slice(0, WINDOW_SIZE),
        ...accBuffer[2].slice(0, WINDOW_SIZE),
        ...gyrBuffer[0].slice(0, WINDOW_SIZE),
        ...gyrBuffer[1].slice(0, WINDOW_SIZE),
        ...gyrBuffer[2].slice(0, WINDOW_SIZE),
      ];

      // Send to ONNX WebView for inference
      webViewRef.current?.postMessage(JSON.stringify({
        type: 'infer', data: flat,
      }));

      // Slide window: keep the last 16 samples (≈ 25% overlap)
      const keep = 16;
      accBuffer = accBuffer.map(a => a.slice(-keep));
      gyrBuffer = gyrBuffer.map(g => g.slice(-keep));
    }

    return () => {
      collectingRef.current = false;
      accSub.remove();
      gyrSub.remove();
    };
  }, [modelReady, sensorReady]);

  // ── Render ───────────────────────────────────────────────────
  const current = ACTIVITIES[activity] || { name: 'Initializing...', emoji: '⏳', color: '#888' };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* ── Hidden WebView for ONNX inference ── */}
      <WebView
        ref={webViewRef}
        source={{ html: INFERENCE_HTML }}
        style={styles.hiddenWebView}
        onMessage={handleMessage}
        javaScriptEnabled
        originWhitelist={['*']}
      />

      {/* ── Header ── */}
      <View style={styles.header}>
        <Text style={styles.title}>EdgeHealth</Text>
        <Text style={styles.subtitle}>Real-time Activity Recognition</Text>
      </View>

      {/* ── Main activity display ── */}
      <View style={[styles.activityCard, { borderColor: current.color }]}>
        <Text style={styles.emoji}>{current.emoji}</Text>
        <Text style={[styles.activityName, { color: current.color }]}>
          {current.name}
        </Text>
        {activity >= 0 && (
          <Text style={styles.confidence}>
            {(confidence * 100).toFixed(1)}% confidence
          </Text>
        )}
      </View>

      {/* ── All activities confidence bars ── */}
      <ScrollView style={styles.barContainer}>
        {ACTIVITIES.map((act, i) => (
          <View key={i} style={styles.barRow}>
            <Text style={styles.barLabel}>{act.emoji} {act.name}</Text>
            <View style={styles.barTrack}>
              <View style={[
                styles.barFill,
                {
                  width: `${(probs[i] * 100).toFixed(0)}%`,
                  backgroundColor: act.color,
                },
              ]} />
            </View>
            <Text style={styles.barPercent}>
              {(probs[i] * 100).toFixed(0)}%
            </Text>
          </View>
        ))}
      </ScrollView>

      {/* ── Status bar ── */}
      <View style={styles.statusBar}>
        <View style={[
          styles.statusDot,
          { backgroundColor: recording ? '#4CAF50' : '#FF5722' },
        ]} />
        <Text style={styles.statusText}>
          {!modelReady ? 'Loading model...' :
           !sensorReady ? 'Starting sensors...' :
           recording ? 'Live — detecting activity' :
           'Stopped'}
        </Text>
      </View>
    </SafeAreaView>
  );
}

// ─── Styles ────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D1117',
    paddingHorizontal: 20,
  },
  hiddenWebView: {
    height: 0,
    width: 0,
    position: 'absolute',
    opacity: 0,
  },
  header: {
    marginTop: 20,
    marginBottom: 10,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  subtitle: {
    fontSize: 14,
    color: '#8B949E',
    marginTop: 4,
  },
  activityCard: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
    marginVertical: 16,
    borderRadius: 24,
    borderWidth: 3,
    backgroundColor: '#161B22',
  },
  emoji: {
    fontSize: 72,
    marginBottom: 12,
  },
  activityName: {
    fontSize: 32,
    fontWeight: '800',
  },
  confidence: {
    fontSize: 16,
    color: '#8B949E',
    marginTop: 8,
  },
  barContainer: {
    flex: 1,
    marginTop: 8,
  },
  barRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  barLabel: {
    width: 100,
    fontSize: 11,
    color: '#C9D1D9',
  },
  barTrack: {
    flex: 1,
    height: 16,
    backgroundColor: '#21262D',
    borderRadius: 8,
    overflow: 'hidden',
    marginHorizontal: 8,
  },
  barFill: {
    height: '100%',
    borderRadius: 8,
  },
  barPercent: {
    width: 36,
    fontSize: 12,
    color: '#8B949E',
    textAlign: 'right',
  },
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    fontSize: 13,
    color: '#8B949E',
  },
});
