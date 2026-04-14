"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { apiClient, ModelInfo, VerificationResult } from "@/lib/api";
import { getSettings, saveSettings } from "@/lib/settings";

export default function VerifyPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("Facenet");
  const [threshold, setThreshold] = useState(0.7);
  const [preview, setPreview] = useState<string>("");
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const [mode, setMode] = useState("start"); // start, camera, verify
  
  const videoEl = useRef<HTMLVideoElement>(null);
  const mediaStream = useRef<MediaStream | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const s = getSettings();
    setSelectedModel(s.selectedModel);
    setThreshold(s.threshold);
    apiClient.getModels().then(setModels).catch(() => setMsg("Load error"));
  }, []);

  const onStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      mediaStream.current = stream;
      setMode("camera");
    } catch {
      setMsg("No camera");
    }
  };

  useEffect(() => {
    if (mode === "camera" && videoEl.current && mediaStream.current) {
      videoEl.current.srcObject = mediaStream.current;
      videoEl.current.play();
    }
  }, [mode]);

  const doCapture = () => {
    if (!videoEl.current) return;
    const c = document.createElement("canvas");
    c.width = videoEl.current.videoWidth || 640;
    c.height = videoEl.current.videoHeight || 480;
    c.getContext("2d")?.drawImage(videoEl.current, 0, 0);
    setPreview(c.toDataURL("image/jpeg"));
    mediaStream.current?.getTracks().forEach(t => t.stop());
    setMode("verify");
  };

  const onVerify = async () => {
    if (!preview) return;
    setLoading(true);
    try {
      const b = await fetch(preview).then(r => r.blob());
      const f = new File([b], "v.jpg", { type: "image/jpeg" });
      const r = await apiClient.verifyFace(f, selectedModel, threshold);
      setResult(r);
    } catch (e: any) {
      setMsg(e.response?.data?.detail || "Error");
    }
    setLoading(false);
  };

  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => { setPreview(r.result as string); setMode("verify"); };
    r.readAsDataURL(f);
  };

  return (
    <div style={{ background: "var(--background)", minHeight: "calc(100vh - 57px)" }}>
      <div className="max-w-md mx-auto px-4 py-8">
        <h1 style={{ color: "var(--text-primary)", fontSize: "24px", fontWeight: "bold", marginBottom: "20px" }}>Recognize</h1>

        {result ? (
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "80px", filter: result.is_match ? "none" : "grayscale(100%)" }}>
              {result.is_match ? "✓" : "✗"}
            </div>
            <h2 style={{ color: result.is_match ? "var(--success)" : "var(--text-secondary)", fontSize: "28px", fontWeight: "700" }}>
              {result.is_match ? result.name : "Unknown"}
            </h2>
            <p style={{ color: "var(--text-secondary)", marginTop: "8px" }}>{(result.confidence * 100).toFixed(1)}%</p>
            <button onClick={() => { setResult(null); setPreview(""); setMode("start"); }} style={{ marginTop: 24, background: "white", color: "black", padding: "12px 32px", borderRadius: 24, fontWeight: 600, border: "none", cursor: "pointer" }}>
              Another
            </button>
          </div>
        ) : (
          <>
            <div style={{ marginBottom: 16, borderRadius: 16, overflow: "hidden", border: "1px solid var(--border)", background: "#111" }}>
              {mode === "camera" && (
                <video ref={videoEl} autoPlay playsInline muted style={{ width: "100%", display: "block", transform: "scaleX(-1)" }} />
              )}
              {mode === "verify" && preview && (
                <img src={preview} alt="" style={{ width: "100%", display: "block" }} />
              )}
              {mode === "start" && (
                <button onClick={onStart} style={{ width: "100%", aspectRatio: 1, background: "var(--surface)", border: "none", color: "var(--text-secondary)", cursor: "pointer" }}>
                  Tap for camera
                </button>
              )}
            </div>

            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              {mode === "camera" && (
                <>
                  <button onClick={doCapture} style={{ flex: 1, background: "white", color: "black", padding: 14, borderRadius: 999, fontWeight: 600, border: "none", cursor: "pointer" }}>
                    Capture
                  </button>
                  <button onClick={() => { mediaStream.current?.getTracks().forEach(t => t.stop()); setMode("start"); }} style={{ background: "var(--surface)", color: "white", padding: "14px 20px", borderRadius: 999, border: "1px solid var(--border)" }}>
                    Cancel
                  </button>
                </>
              )}
              {mode === "Recognition" && (
                <>
                  <button onClick={onVerify} disabled={loading} style={{ flex: 1, background: loading ? "var(--surface)" : "white", color: loading ? "gray" : "black", padding: 14, borderRadius: 999, fontWeight: 600, border: "none", cursor: loading ? "not-allowed" : "pointer" }}>
                    {loading ? "Recognizing..." : "recognize"}
                  </button>
                  <button onClick={() => { setPreview(""); onStart(); }} style={{ background: "var(--surface)", color: "white", padding: "14px 20px", borderRadius: 999, border: "1px solid var(--border)" }}>
                    Retake
                  </button>
                </>
              )}
            </div>

            <input ref={fileRef} type="file" accept="image/*" onChange={onUpload} hidden />
            <button onClick={() => fileRef.current?.click()} style={{ width: "100%", background: "var(--surface)", color: "var(--text-primary)", padding: 14, borderRadius: 12, border: "1px solid var(--border)", marginBottom: 16 }}>
              Upload
            </button>

            <div style={{ background: "var(--surface)", borderRadius: 16, padding: 16, marginBottom: 16 }}>
              <div style={{ marginBottom: 12 }}>
                <label style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 6, display: "block" }}>Model</label>
                <select value={selectedModel} onChange={(e) => { setSelectedModel(e.target.value); saveSettings({ selectedModel: e.target.value }); }} style={{ width: "100%", background: "var(--background)", border: "1px solid var(--border)", borderRadius: 8, padding: 10, color: "var(--text-primary)" }}>
                  {models.map(m => <option key={m.name} value={m.name} disabled={!m.available}>{m.display_name}</option>)}
                </select>
              </div>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary)", fontSize: 14, marginBottom: 6 }}>
                  <span>Threshold</span>
                  <span>{(threshold * 100).toFixed(0)}%</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => { const v = +e.target.value; setThreshold(v); saveSettings({ threshold: v }); }} style={{ width: "100%" }} />
              </div>
            </div>

            {msg && <div style={{ padding: 12, borderRadius: 12, background: "rgba(244,33,46,0.1)", color: "var(--error)", fontSize: 14 }}>{msg}</div>}
          </>
        )}
      </div>
    </div>
  );
}