"use client";

import { useEffect, useRef } from "react";

interface VideoFeedProps {
  onStream: (stream: MediaStream) => void;
  className?: string;
}

export default function VideoFeed({ onStream, className }: VideoFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: 640, height: 480 },
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
        
        if (stream) {
          onStream(stream);
        }
      } catch (err) {
        console.error('Camera error:', err);
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [onStream]);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      style={{
        width: '100%',
        height: 'auto',
        objectFit: 'cover',
        transform: 'scaleX(-1)',
        backgroundColor: '#111'
      }}
      className={className}
    />
  );
}
