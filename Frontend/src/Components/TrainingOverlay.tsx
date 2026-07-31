import { useRef, useState, useEffect } from 'react';
import { sendMessageToBackend } from '../Utils/MessageUtils';
import type { TrainingEventDefinition } from '../Models/types';

interface Box { x: number; y: number; w: number; h: number; }
interface LabeledBox { id: number; box: Box; eventId: number | null; eventName: string; }

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isPaused: boolean;
  enabled: boolean;
  events: TrainingEventDefinition[];
  gameId: string;
  onSampleAdded: () => void;
}

export default function TrainingOverlay({ videoRef, isPaused, enabled, events, gameId, onSampleAdded }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [boxes, setBoxes] = useState<LabeledBox[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [picker, setPicker] = useState<{ idx: number } | null>(null);
  const [newName, setNewName] = useState('');
  const [showNew, setShowNew] = useState(false);
  const [saving, setSaving] = useState(false);

  const isDown = useRef(false);
  const sx = useRef(0); const sy = useRef(0);
  const ex = useRef(0); const ey = useRef(0);
  const moving = useRef(false);
  const msx = useRef(0); const msy = useRef(0);
  const bs = useRef<Box>({ x: 0, y: 0, w: 0, h: 0 });
  const rh = useRef<string | null>(null);
  const nid = useRef(1);
  const videoEl = useRef<HTMLVideoElement | null>(null);
  videoEl.current = videoRef.current;
  const pickerRef = useRef<{ idx: number } | null>(null);
  pickerRef.current = picker;

  const getVR = () => videoEl.current?.getBoundingClientRect();

  function draw() {
    const c = canvasRef.current;
    const vr = getVR();
    if (!c || !vr || vr.width === 0) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.style.left = vr.left + 'px';
    c.style.top = vr.top + 'px';
    c.style.width = vr.width + 'px';
    c.style.height = vr.height + 'px';
    c.width = vr.width;
    c.height = vr.height;
    ctx.clearRect(0, 0, vr.width, vr.height);

    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i].box;
      const s = i === sel;
      ctx.strokeStyle = s ? '#3b82f6' : '#22c55e';
      ctx.lineWidth = s ? 2.5 : 2;
      ctx.fillStyle = s ? 'rgba(59,130,246,0.12)' : 'rgba(34,197,94,0.10)';
      ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      if (boxes[i].eventName) {
        ctx.font = '11px sans-serif';
        const tw = ctx.measureText(boxes[i].eventName).width;
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.fillRect(b.x - 2, b.y - 17, tw + 6, 16);
        ctx.fillStyle = '#fff';
        ctx.fillText(boxes[i].eventName, b.x, b.y - 5);
      }
      if (s) {
        ctx.fillStyle = '#fff'; ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5;
        for (const [px, py] of [[b.x, b.y], [b.x + b.w / 2, b.y], [b.x + b.w, b.y],
          [b.x + b.w, b.y + b.h / 2], [b.x + b.w, b.y + b.h],
          [b.x + b.w / 2, b.y + b.h], [b.x, b.y + b.h], [b.x, b.y + b.h / 2]]) {
          ctx.fillRect(px - 4, py - 4, 8, 8); ctx.strokeRect(px - 4, py - 4, 8, 8);
        }
      }
    }
    if (isDown.current && !moving.current && !rh.current) {
      const x = Math.min(sx.current, ex.current), y = Math.min(sy.current, ey.current);
      const w = Math.abs(ex.current - sx.current), h = Math.abs(ey.current - sy.current);
      if (w > 0 || h > 0) {
        ctx.strokeStyle = 'rgba(255,50,50,0.9)'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
        ctx.fillStyle = 'rgba(255,50,50,0.06)'; ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
      }
    }
  }

  useEffect(() => {
    if (!enabled || !isPaused) return;
    let running = true;
    const loop = () => { if (!running) return; draw(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    return () => { running = false; };
  }, [enabled, isPaused, boxes, sel]);

  useEffect(() => {
    if (!enabled || !isPaused) { setBoxes([]); setSel(null); setPicker(null); }
  }, [enabled, isPaused]);

  useEffect(() => {
    if (!enabled || !isPaused) return;

    const ondown = (e: MouseEvent) => {
      e.preventDefault();
      if (pickerRef.current) return;
      const r = getVR(); if (!r) return;
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const h = (sel !== null) ? (() => {
        const b = boxes[sel].box;
        for (const [hx, hy, hh] of [[b.x, b.y, 'nw'], [b.x + b.w / 2, b.y, 'n'], [b.x + b.w, b.y, 'ne'],
          [b.x + b.w, b.y + b.h / 2, 'e'], [b.x + b.w, b.y + b.h, 'se'],
          [b.x + b.w / 2, b.y + b.h, 's'], [b.x, b.y + b.h, 'sw'], [b.x, b.y + b.h / 2, 'w']] as const) {
          if (Math.abs(px - hx) <= 10 && Math.abs(py - hy) <= 10) return hh;
        } return null;
      })() : null;
      if (h && sel !== null) { rh.current = h; bs.current = { ...boxes[sel].box }; msx.current = px; msy.current = py; return; }
      const hit = (() => { for (let i = boxes.length - 1; i >= 0; i--) { const b = boxes[i].box; if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return i; } return -1; })();
      if (hit >= 0) { setSel(hit); moving.current = true; msx.current = px; msy.current = py; bs.current = { ...boxes[hit].box }; return; }
      setSel(null); isDown.current = true; sx.current = px; sy.current = py; ex.current = px; ey.current = py;
    };
    const onmove = (e: MouseEvent) => {
      const r = getVR(); if (!r) return;
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (rh.current && sel !== null) {
        const h = rh.current, s = bs.current;
        const dx = px - msx.current, dy = py - msy.current;
        let x = s.x, y = s.y, w = s.w, bh = s.h;
        if (h.includes('w')) { x = s.x + dx; w = s.w - dx; }
        if (h.includes('e')) w = s.w + dx;
        if (h.includes('n')) { y = s.y + dy; bh = s.h - dy; }
        if (h.includes('s')) bh = s.h + dy;
        if (w < 10) { w = 10; if (h.includes('w')) x = s.x + s.w - 10; }
        if (bh < 10) { bh = 10; if (h.includes('n')) y = s.y + s.h - 10; }
        setBoxes(prev => { const n = [...prev]; n[sel] = { ...n[sel], box: { x, y, w, h: bh } }; return n; });
        return;
      }
      if (moving.current && sel !== null) {
        const dx = px - msx.current, dy = py - msy.current;
        setBoxes(prev => { const n = [...prev]; n[sel] = { ...n[sel], box: { x: bs.current.x + dx, y: bs.current.y + dy, w: bs.current.w, h: bs.current.h } }; return n; });
        return;
      }
      if (isDown.current) { ex.current = px; ey.current = py; }
    };
    const onup = () => {
      if (rh.current) { rh.current = null; return; }
      if (moving.current) { moving.current = false; return; }
      if (!isDown.current) return;
      isDown.current = false;
      const w = Math.abs(ex.current - sx.current), h = Math.abs(ey.current - sy.current);
      if (w < 5 || h < 5) return;
      const x = Math.min(sx.current, ex.current), y = Math.min(sy.current, ey.current);
      const id = nid.current++;
      setBoxes(prev => { const idx = prev.length; setTimeout(() => setPicker({ idx }), 0); return [...prev, { id, box: { x, y, w, h }, eventId: null, eventName: '' }]; });
      setSel(boxes.length);
    };
    window.addEventListener('mousedown', ondown);
    window.addEventListener('mousemove', onmove);
    window.addEventListener('mouseup', onup);
    return () => { window.removeEventListener('mousedown', ondown); window.removeEventListener('mousemove', onmove); window.removeEventListener('mouseup', onup); };
  }, [enabled, isPaused, boxes, sel]);

  function saveAll() {
    const labeled = boxes.filter(b => b.eventId !== null);
    if (!labeled.length) return;
    setSaving(true);
    const vEl = videoEl.current;
    const vr = getVR();
    if (!vEl || !vr) return;
    const vw = vEl.videoWidth, vh = vEl.videoHeight;
    if (!vw || !vh) return;

    // Render full frame once
    const o = document.createElement('canvas');
    o.width = vw; o.height = vh;
    const octx = o.getContext('2d'); if (!octx) return;
    octx.drawImage(vEl, 0, 0, vw, vh);
    const imageData = o.toDataURL('image/png').split(',')[1];

    const s = Math.min(vr.width / vw, vr.height / vh);
    const ox = (vr.width - vw * s) / 2, oy = (vr.height - vh * s) / 2;

    for (const ab of labeled) {
      const nb = {
        x: Math.max(0, (ab.box.x - ox) / s),
        y: Math.max(0, (ab.box.y - oy) / s),
        w: Math.min(ab.box.w / s, vw),
        h: Math.min(ab.box.h / s, vh),
      };

      sendMessageToBackend('AddTrainingSample', {
        gameId,
        eventId: ab.eventId,
        imageData,
        boxX: Math.round(((nb.x + nb.w / 2) / vw) * 10000) / 10000,
        boxY: Math.round(((nb.y + nb.h / 2) / vh) * 10000) / 10000,
        boxW: Math.round((nb.w / vw) * 10000) / 10000,
        boxH: Math.round((nb.h / vh) * 10000) / 10000,
      });
    }
    setBoxes([]); setSel(null); setPicker(null);
    setSaving(false);
    onSampleAdded();
  }

  function assignEvent(eventId: number) {
    if (!picker) return;
    const ev = events.find(e => e.id === eventId);
    setBoxes(prev => { const n = [...prev]; n[picker.idx] = { ...n[picker.idx], eventId, eventName: ev?.name ?? '' }; return n; });
    setPicker(null);
  }

  function createAndAssign() {
    if (!newName.trim() || !picker) return;
    const ev: TrainingEventDefinition = { id: (Date.now() % 2000000000) + Math.floor(Math.random() * 1000), name: newName.trim(), type: 'Trigger', classId: 0 };
    sendMessageToBackend('SaveTrainingEvent', { gameId, event: ev });
    setBoxes(prev => { const n = [...prev]; n[picker.idx] = { ...n[picker.idx], eventId: ev.id, eventName: ev.name }; return n; });
    setNewName(''); setShowNew(false); setPicker(null);
  }

  if (!enabled || !isPaused) return null;

  return (
    <>
      <canvas ref={canvasRef} className="fixed z-10" style={{ pointerEvents: 'none' }} />
      {boxes.filter(b => b.eventName).map((ab, i) => (
        <div key={ab.id} className="fixed z-20 px-1 text-[10px] leading-none rounded-sm pointer-events-none"
          style={{ left: (getVR()?.left ?? 0) + ab.box.x, top: (getVR()?.top ?? 0) + ab.box.y - 16, backgroundColor: i === sel ? '#3b82f6' : '#22c55e', color: '#fff', display: picker?.idx === i ? 'none' : 'block' }}>
          {ab.eventName}
        </div>
      ))}
      {boxes.filter(b => b.eventId !== null).length > 0 && (
        <div className="fixed z-20 flex gap-2" style={{ left: (getVR()?.left ?? 0) + (getVR()?.width ?? 0) / 2 - 60, top: (getVR()?.top ?? 0) + (getVR()?.height ?? 0) - 40 }}>
          <button className="btn btn-primary btn-xs" onClick={saveAll} disabled={saving}>
            {saving ? 'Saving...' : `Save ${boxes.filter(b => b.eventId !== null).length}`}
          </button>
          <button className="btn btn-ghost btn-xs" onClick={() => { setBoxes([]); setSel(null); setPicker(null); }}>Clear</button>
        </div>
      )}
      {sel !== null && boxes[sel] && (
        <div className="fixed z-30 flex gap-1" style={{ left: (getVR()?.left ?? 0) + boxes[sel].box.x + boxes[sel].box.w - 20, top: (getVR()?.top ?? 0) + boxes[sel].box.y - 20 }}>
          <button className="btn btn-square btn-xs text-red-400 hover:text-red-300" onClick={() => {
            setBoxes(prev => { const n = [...prev]; n.splice(sel, 1); return n; });
            setSel(null);
          }} title="Delete box">✕</button>
          {boxes[sel].eventName && (
            <button className="btn btn-xs" onClick={() => setPicker({ idx: sel })} title="Change event">Edit</button>
          )}
        </div>
      )}

      {picker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => {
          setBoxes(prev => { const n = [...prev]; if (n[picker.idx] && !n[picker.idx].eventId) n.splice(picker.idx, 1); return n; });
          setSel(null);
          setPicker(null);
        }}>
          <div className="bg-base-200 border border-base-400 rounded-lg shadow-xl p-3 w-64" onClick={e => e.stopPropagation()}>
            {!showNew ? (
              <>
                <p className="text-xs font-semibold mb-2">Label this box</p>
                {events.length > 0 ? (
                  <div className="space-y-1 max-h-40 overflow-y-auto mb-2">
                    {events.map(ev => (
                      <button key={ev.id} className="block w-full text-left text-xs px-2 py-1 rounded hover:bg-base-300" onClick={() => assignEvent(ev.id)}>
                        <span className="badge badge-xs mr-1">{ev.type === 'Trigger' ? 'T' : 'E'}</span> {ev.name}
                      </button>
                    ))}
                  </div>
                ) : <p className="text-xs opacity-60 mb-2">No event definitions yet — create one below.</p>}
                <button className="btn btn-ghost btn-xs w-full mt-1" onClick={() => setShowNew(true)}>+ New Event</button>
              </>
            ) : (
              <>
                <p className="text-xs font-semibold mb-2">New event name</p>
                <input className="input input-bordered input-xs w-full mb-2" value={newName} onChange={e => setNewName(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') createAndAssign(); }} autoFocus />
                <div className="flex gap-1">
                  <button className="btn btn-ghost btn-xs" onClick={() => { setShowNew(false); setNewName(''); }}>Back</button>
                  <button className="btn btn-primary btn-xs" onClick={createAndAssign} disabled={!newName.trim()}>Create</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}