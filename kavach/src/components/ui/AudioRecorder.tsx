import { useState, useRef, useEffect } from 'react';
import { cn } from '../layout/AppLayout';
import { Square, Play, RotateCcw, Check } from 'lucide-react';

interface AudioRecorderProps {
  onRecordingComplete?: (blob: Blob, durationMs: number) => void;
  onAccept?: (blob: Blob, durationMs: number) => void;
}

export function AudioRecorder({ onRecordingComplete, onAccept }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [durationMs, setDurationMs] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const dataArray = useRef<Uint8Array | null>(null);
  const requestRef = useRef<number>(0);
  const startTime = useRef<number>(0);
  const chunks = useRef<BlobPart[]>([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const levelRef = useRef<HTMLDivElement>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext.current = new AudioContext();
      const source = audioContext.current.createMediaStreamSource(stream);
      analyser.current = audioContext.current.createAnalyser();
      analyser.current.fftSize = 256;
      source.connect(analyser.current);
      
      const bufferLength = analyser.current.frequencyBinCount;
      dataArray.current = new Uint8Array(bufferLength);

      mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      chunks.current = [];
      
      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };
      
      mediaRecorder.current.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        setRecordedBlob(blob);
        if (onRecordingComplete) onRecordingComplete(blob, durationMs);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setError(null);
      startTime.current = performance.now();
      drawWaveform();

    } catch (err) {
      setError('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      cancelAnimationFrame(requestRef.current);
      if (audioContext.current) audioContext.current.close();
    }
  };

  const drawWaveform = () => {
    if (!analyser.current || !dataArray.current || !canvasRef.current || !levelRef.current) return;
    
    analyser.current.getByteFrequencyData(dataArray.current);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate RMS for level meter
    let sum = 0;
    for (let i = 0; i < dataArray.current.length; i++) {
      sum += dataArray.current[i] * dataArray.current[i];
    }
    const rms = Math.sqrt(sum / dataArray.current.length);
    const volume = Math.min(1, rms / 128); // 0 to 1
    
    levelRef.current.style.width = `${volume * 100}%`;
    if (volume > 0.95) levelRef.current.classList.add('bg-app-reject');
    else levelRef.current.classList.remove('bg-app-reject');

    // Draw bars
    const barWidth = 2;
    const gap = 1;
    const barCount = Math.floor(canvas.width / (barWidth + gap));
    const step = Math.floor(dataArray.current.length / barCount);

    ctx.fillStyle = 'var(--app-text-muted)';
    
    for (let i = 0; i < barCount; i++) {
      let val = dataArray.current[i * step] || 0;
      const percent = val / 255;
      const height = Math.max(1, percent * canvas.height);
      const y = (canvas.height - height) / 2;
      ctx.fillRect(i * (barWidth + gap), y, barWidth, height);
    }

    setDurationMs(performance.now() - startTime.current);
    requestRef.current = requestAnimationFrame(drawWaveform);
  };

  const handleReset = () => {
    setRecordedBlob(null);
    setDurationMs(0);
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    if (levelRef.current) levelRef.current.style.width = '0%';
  };

  const handleAccept = () => {
    if (recordedBlob && onAccept) {
      onAccept(recordedBlob, durationMs);
    }
  };
  
  const formatTime = (ms: number) => {
    const totalSec = Math.floor(ms / 1000);
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    const ms1 = Math.floor((ms % 1000) / 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms1}`;
  };

  if (error) {
    return (
      <div className="h-12 border border-app-reject bg-app-reject-subtle text-app-reject flex items-center px-4 text-[13px]">
        {error}
      </div>
    );
  }

  return (
    <div className="h-14 border border-app-border bg-app-surface flex items-center px-4 gap-4">
      {/* Controls */}
      <div className="flex-shrink-0">
        {!recordedBlob ? (
          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            className={cn(
              "w-8 h-8 flex items-center justify-center border border-app-border rounded-sm transition-colors duration-120",
              isRecording ? "bg-app-bg hover:bg-app-bg" : "hover:bg-app-bg"
            )}
            title={isRecording ? "Stop Recording" : "Start Recording"}
          >
            {isRecording ? (
              <Square className="w-3.5 h-3.5 fill-app-reject text-app-reject" />
            ) : (
              <div className="w-3 h-3 rounded-full bg-app-accent" />
            )}
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleReset}
              className="px-2 h-8 flex items-center gap-1.5 border border-app-border rounded-sm hover:bg-app-bg transition-colors duration-120 text-[11px] uppercase tracking-wider text-app-text-muted"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Re-record
            </button>
            <button
              type="button"
              onClick={handleAccept}
              className="px-2 h-8 flex items-center gap-1.5 border border-app-accent bg-app-accent text-white rounded-sm hover:bg-app-accent-hover transition-colors duration-120 text-[11px] uppercase tracking-wider"
            >
              <Check className="w-3.5 h-3.5" /> Accept
            </button>
          </div>
        )}
      </div>

      {/* Timer */}
      <div className="mono text-[13px] w-16 text-right">
        {formatTime(durationMs)}
      </div>

      {/* Waveform & Meter */}
      <div className="flex-1 flex flex-col justify-center h-full gap-1 relative">
         <canvas 
           ref={canvasRef} 
           width={300} 
           height={24} 
           className={cn("w-full h-6 block", recordedBlob && "opacity-50 grayscale")} 
         />
         {!recordedBlob && (
           <div className="w-full h-[2px] bg-app-bg">
             <div ref={levelRef} className="h-full bg-app-accept w-0 transition-all duration-75" />
           </div>
         )}
      </div>
    </div>
  );
}
