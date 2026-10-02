import { useState, useRef, useEffect } from 'react';
import { cn, Button } from './kit';
import { Mic, Square, RotateCcw, Check, Upload, FileAudio } from 'lucide-react';

interface AudioRecorderProps {
  onRecordingComplete?: (blob: Blob, durationMs: number) => void;
  onAccept?: (blob: Blob, durationMs: number, filename?: string) => void;
  /** Allow submitting an audio file instead of the microphone -- how replayed
   *  and cloned clips get into the system during a demo. */
  allowUpload?: boolean;
  acceptLabel?: string;
  busy?: boolean;
}

/**
 * Canvas cannot resolve CSS custom properties, so `fillStyle = 'var(--x)'` is
 * silently ignored and the waveform used to draw in black on every theme.
 * Resolve the token to a concrete colour first.
 */
function cssColor(name: string, fallback: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export function AudioRecorder({ onRecordingComplete, onAccept, allowUpload, acceptLabel = 'Submit', busy }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [durationMs, setDurationMs] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [filename, setFilename] = useState<string | undefined>();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const dataArray = useRef<Uint8Array | null>(null);
  const requestRef = useRef<number>(0);
  const startTime = useRef<number>(0);
  const durationRef = useRef(0);
  const chunks = useRef<BlobPart[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const setResult = (blob: Blob, name?: string) => {
    setRecordedBlob(blob);
    setFilename(name);
    setPreviewUrl(URL.createObjectURL(blob));
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext.current = new AudioContext();
      const source = audioContext.current.createMediaStreamSource(stream);
      analyser.current = audioContext.current.createAnalyser();
      analyser.current.fftSize = 512;
      source.connect(analyser.current);
      dataArray.current = new Uint8Array(analyser.current.fftSize);

      mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      chunks.current = [];
      mediaRecorder.current.ondataavailable = e => { if (e.data.size > 0) chunks.current.push(e.data); };
      mediaRecorder.current.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        setResult(blob);
        onRecordingComplete?.(blob, durationRef.current);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setError(null);
      startTime.current = performance.now();
      draw();
    } catch {
      setError('Microphone access was denied or no microphone is available. You can still upload a file.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      cancelAnimationFrame(requestRef.current);
      audioContext.current?.close();
    }
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!analyser.current || !dataArray.current || !canvas) return;
    analyser.current.getByteTimeDomainData(dataArray.current);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const bars = Math.floor(w / 4);
    const step = Math.max(1, Math.floor(dataArray.current.length / bars));
    ctx.fillStyle = cssColor('--app-accent', '#2B4C7E');
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      for (let j = 0; j < step; j++) peak = Math.max(peak, Math.abs((dataArray.current[i * step + j] ?? 128) - 128));
      const bh = Math.max(2, (peak / 128) * h * 1.6);
      ctx.fillRect(i * 4, (h - Math.min(h, bh)) / 2, 2, Math.min(h, bh));
    }

    durationRef.current = performance.now() - startTime.current;
    setDurationMs(durationRef.current);
    requestRef.current = requestAnimationFrame(draw);
  };

  const handleReset = () => {
    setRecordedBlob(null);
    setFilename(undefined);
    setPreviewUrl(null);
    setDurationMs(0);
    durationRef.current = 0;
    const c = canvasRef.current;
    c?.getContext('2d')?.clearRect(0, 0, c.width, c.height);
  };

  const onFile = (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setResult(file, file.name);
    const probe = new Audio(URL.createObjectURL(file));
    probe.onloadedmetadata = () => {
      if (Number.isFinite(probe.duration)) { durationRef.current = probe.duration * 1000; setDurationMs(durationRef.current); }
    };
  };

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  };

  return (
    <div className="rounded-lg border border-app-border bg-app-surface">
      <div className="flex items-center gap-4 p-4">
        {!recordedBlob ? (
          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            className={cn(
              'w-12 h-12 rounded-full flex items-center justify-center shrink-0 transition-all',
              isRecording ? 'bg-app-reject text-white ring-4 ring-app-reject/20' : 'bg-app-accent text-app-on-accent hover:bg-app-accent-hover',
            )}
            title={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isRecording ? <Square className="w-4 h-4 fill-current" /> : <Mic className="w-5 h-5" />}
          </button>
        ) : (
          <div className="w-12 h-12 rounded-full bg-app-accept-soft text-app-accept flex items-center justify-center shrink-0">
            {filename ? <FileAudio className="w-5 h-5" /> : <Check className="w-5 h-5" />}
          </div>
        )}

        <div className="flex-1 min-w-0">
          {recordedBlob && previewUrl ? (
            <div className="flex flex-col gap-1.5">
              <div className="text-[12.5px] text-app-text-muted truncate">
                {filename ? <>File: <span className="text-app-text font-medium">{filename}</span></> : 'Recording ready'} · <span className="tnum">{formatTime(durationMs)}</span>
              </div>
              <audio src={previewUrl} controls className="w-full" />
            </div>
          ) : (
            <div className="relative flex items-center gap-4">
              <canvas ref={canvasRef} className={cn('flex-1 h-10 min-w-0', !isRecording && 'opacity-0')} />
              {!isRecording && (
                <div className="absolute text-[13px] text-app-text-muted pointer-events-none">
                  Press the microphone and answer in your own words.
                </div>
              )}
              <span className={cn('tnum text-[14px] font-medium w-12 text-right', isRecording ? 'text-app-reject' : 'text-app-text-subtle')}>{formatTime(durationMs)}</span>
            </div>
          )}
        </div>
      </div>

      {(recordedBlob || allowUpload || error) && (
        <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-app-border bg-app-surface-muted/50 rounded-b-lg">
          <div className="text-[12px] text-app-reject min-w-0">{error}</div>
          <div className="flex items-center gap-2 shrink-0">
            {allowUpload && !recordedBlob && !isRecording && (
              <>
                <input ref={fileRef} type="file" accept="audio/*,video/mp4,.m4a,.ogg,.wav,.webm" className="hidden" onChange={e => onFile(e.target.files?.[0])} />
                <Button size="sm" variant="ghost" icon={<Upload className="w-3.5 h-3.5" />} onClick={() => fileRef.current?.click()}>
                  Upload audio file
                </Button>
              </>
            )}
            {recordedBlob && (
              <>
                <Button size="sm" variant="ghost" icon={<RotateCcw className="w-3.5 h-3.5" />} onClick={handleReset} disabled={busy}>Redo</Button>
                <Button size="sm" variant="primary" icon={<Check className="w-3.5 h-3.5" />} loading={busy} onClick={() => onAccept?.(recordedBlob, durationMs, filename)}>
                  {acceptLabel}
                </Button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
