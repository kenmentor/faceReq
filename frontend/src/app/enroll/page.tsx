"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { apiClient } from "../../../lib/api";

export default function EnrollPage() {
  const [name, setName] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string; timing?: any } | null>(null);
  const [enrolledCount, setEnrolledCount] = useState<number | null>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [enrollTiming, setEnrollTiming] = useState<any>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const autoCaptureRef = useRef<ReturnType<typeof setInterval> | null>(null);
console.log("enroll page loaded")
  useEffect(() => {
    loadCount();
    return () => {
      stopCamera();
    };
  }, []);

  const loadCount = async () => {
    try {
      const users = await apiClient.getUsers();
      setEnrolledCount(users.length);
    } catch {}
  };

  const startCamera = useCallback(async () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false
      });
      
      streamRef.current = stream;
      setShowCamera(true);
    } catch (err) {
      console.error('Camera error:', err);
      setMessage({ type: "error", text: "Camera access denied" });
    }
  }, []);

  useEffect(() => {
    if (streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().catch(console.error);
      };
    }
  }, [showCamera]);

  const stopCamera = useCallback(() => {
    if (autoCaptureRef.current) {
      clearInterval(autoCaptureRef.current);
      autoCaptureRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setShowCamera(false);
  }, []);

  const captureImage = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    if (images.length >= 5) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    setImages(prev => [...prev, dataUrl]);
  }, [images.length]);

  const startAutoCapture = useCallback(() => {
    if (autoCaptureRef.current) return;
    autoCaptureRef.current = setInterval(captureImage, 1500);
  }, [captureImage]);

  const stopAutoCapture = useCallback(() => {
    if (autoCaptureRef.current) {
      clearInterval(autoCaptureRef.current);
      autoCaptureRef.current = null;
    }
  }, []);

  const handleUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      Array.from(files).forEach(file => {
        if (images.length >= 5) return;
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            setImages(prev => [...prev, event.target!.result as string]);
          }
        };
        reader.readAsDataURL(file);
      });
    }
  }, [images.length]);

  const removeImage = (index: number) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  const dataURLtoFile = (dataurl: string, filename: string): File => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new File([u8arr], filename, { type: mime });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setMessage({ type: "error", text: "Please enter a name" });
      return;
    }
    if (images.length < 3) {
      setMessage({ type: "error", text: "Please add at least 3 images" });
      return;
    }

    setIsLoading(true);
    setMessage(null);

    try {
      const files = images.map((img, i) => dataURLtoFile(img, `img_${i}.jpg`));
      const result = await apiClient.enrollUser(name, files);
      setEnrollTiming(result.timing);
      setMessage({ type: "success", text: `Enrolled ${name} successfully!`, timing: result.timing });
      setName("");
      setImages([]);
      loadCount();
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.detail || "Failed to enroll" });
      setEnrollTiming(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ background: 'var(--background)', minHeight: 'calc(100vh - 57px)', padding: '16px' }}>
      <h1 style={{ color: 'white', fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }}>Enroll New User</h1>

      <div style={{ marginBottom: '16px' }}>
        {showCamera ? (
          <div style={{ position: 'relative' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                borderRadius: '12px',
                border: '1px solid var(--border)',
                transform: 'scaleX(-1)',
                background: '#111'
              }}
            />
            {autoCaptureRef.current && (
              <div style={{
                position: 'absolute',
                top: '8px',
                right: '8px',
                background: 'red',
                color: 'white',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px'
              }}>
                REC
              </div>
            )}
            <div style={{ marginTop: '8px', color: 'var(--text-secondary)', fontSize: '14px' }}>
              {images.length}/5 images captured
            </div>
          </div>
        ) : (
          <button
            onClick={startCamera}
            style={{
              width: '100%',
              padding: '40px',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '12px',
              color: 'white',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Open Camera
          </button>
        )}
      </div>

      {showCamera && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
          {!autoCaptureRef.current ? (
            <button
              onClick={startAutoCapture}
              disabled={images.length >= 5}
              style={{
                padding: '10px 16px',
                background: images.length >= 5 ? 'var(--surface)' : 'var(--accent)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: images.length >= 5 ? 'not-allowed' : 'pointer'
              }}
            >
              Auto Capture
            </button>
          ) : (
            <button
              onClick={stopAutoCapture}
              style={{
                padding: '10px 16px',
                background: 'var(--error)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
            >
              Stop Auto
            </button>
          )}
          <button
            onClick={captureImage}
            disabled={images.length >= 5}
            style={{
              padding: '10px 16px',
              background: images.length >= 5 ? 'var(--surface)' : 'var(--text-primary)',
              color: images.length >= 5 ? 'gray' : 'black',
              border: 'none',
              borderRadius: '8px',
              cursor: images.length >= 5 ? 'not-allowed' : 'pointer'
            }}
          >
            Snap ({images.length}/5)
          </button>
          <button
            onClick={stopCamera}
            style={{
              padding: '10px 16px',
              background: 'var(--surface)',
              color: 'white',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
          >
            Close
          </button>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleUpload}
        style={{ display: 'none' }}
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={images.length >= 5}
        style={{
          width: '100%',
          padding: '10px',
          background: 'transparent',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          color: 'var(--accent)',
          cursor: images.length >= 5 ? 'not-allowed' : 'pointer',
          marginBottom: '16px'
        }}
      >
        Upload Images ({images.length}/5)
      </button>

      {images.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
          {images.map((img, i) => (
            <div key={i} style={{ position: 'relative' }}>
              <img src={img} alt="" style={{ width: '60px', height: '60px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--border)' }} />
              <button
                onClick={() => removeImage(i)}
                style={{
                  position: 'absolute',
                  top: '-6px',
                  right: '-6px',
                  width: '20px',
                  height: '20px',
                  background: 'var(--error)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '50%',
                  cursor: 'pointer',
                  fontSize: '14px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter user name"
          style={{
            width: '100%',
            padding: '12px',
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'white',
            marginBottom: '12px'
          }}
        />
        
        {message && (
          <div style={{ marginBottom: '12px' }}>
            <div style={{
              padding: '10px',
              background: message.type === 'success' ? 'rgba(0,186,124,0.2)' : 'rgba(244,33,46,0.2)',
              color: message.type === 'success' ? 'var(--success)' : 'var(--error)',
              borderRadius: '8px',
              marginBottom: message.timing ? '8px' : '0'
            }}>
              {message.text}
            </div>
            {message.timing && (
              <div style={{
                padding: '12px',
                background: 'var(--surface)',
                borderRadius: '8px',
                fontSize: '14px'
              }}>
                <div style={{ color: 'var(--text-primary)', fontWeight: '600', marginBottom: '8px' }}>Performance</div>
                <div style={{ display: 'grid', gap: '6px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Total Time</span>
                    <span style={{ color: 'var(--accent)', fontWeight: '600' }}>{(message.timing.total_ms / 1000).toFixed(3)}s</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Face Detection</span>
                    <span style={{ color: 'var(--text-primary)' }}>{(message.timing.face_detection_total_ms / 1000).toFixed(3)}s</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Embedding</span>
                    <span style={{ color: 'var(--text-primary)' }}>{(message.timing.embedding_extraction_total_ms / 1000).toFixed(3)}s</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Avg per Image</span>
                    <span style={{ color: 'var(--text-primary)' }}>{(message.timing.avg_per_image_ms / 1000).toFixed(3)}s</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || images.length < 3}
          style={{
            width: '100%',
            padding: '14px',
            background: isLoading || images.length < 3 ? 'var(--surface)' : 'var(--text-primary)',
            color: isLoading || images.length < 3 ? 'gray' : 'black',
            border: 'none',
            borderRadius: '8px',
            cursor: isLoading || images.length < 3 ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          {isLoading ? 'Enrolling...' : 'Enroll User'}
        </button>
      </form>

      {enrolledCount !== null && (
        <p style={{ color: 'var(--text-secondary)', textAlign: 'center', marginTop: '16px' }}>
          {enrolledCount} user(s) enrolled
        </p>
      )}
    </div>
  );
}
