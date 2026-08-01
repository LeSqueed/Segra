import { useState, useEffect, useRef, useCallback } from 'react';
import { sendMessageToBackend } from '../Utils/MessageUtils';
import { useAppState } from '../Context/AppStateContext';
import { BookmarkType, type TrainingEventDefinition } from '../Models/types';

interface TrainingSample {
  id: number;
  eventName: string;
  eventId: number;
  eventNames?: string[];
  labelCount?: number;
}

interface DatasetExport {
  success: boolean;
  datasetPath: string;
  gameId: string;
}

interface ModelStatus {
  hasModel: boolean;
  sampleCount: number;
  gameId: string;
  lastTrained?: string;
}

export default function TrainingPage() {
  const appState = useAppState();

  const games = [...new Set(appState.content.map((c) => c.game).filter(Boolean))];
  const [selectedGame, setSelectedGame] = useState(games[0] || '');
  const [events, setEvents] = useState<TrainingEventDefinition[]>([]);
  const [sampleCounts, setSampleCounts] = useState<Record<number, number>>({});
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [samples, setSamples] = useState<TrainingSample[]>([]);
  const [exportResult, setExportResult] = useState<DatasetExport | null>(null);
  const [trainResult, setTrainResult] = useState<{ status: string; message: string } | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainLog, setTrainLog] = useState<{ status: string; message: string }[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);
  const [loadingData, setLoadingData] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [trainMetrics, setTrainMetrics] = useState<{
    epoch: number; totalEpochs: number;
    boxLoss: number; clsLoss: number; dflLoss: number;
    precision: number; recall: number; mAP50: number; mAP5095: number;
    bestMAP50: number;
  } | null>(null);

  const [editingEvent, setEditingEvent] = useState<TrainingEventDefinition | null>(null);
  const [showEventModal, setShowEventModal] = useState(false);
  const [eventForm, setEventForm] = useState({
    name: '',
    type: 'Trigger' as 'Trigger' | 'Exclusion',
    bookmarkType: undefined as BookmarkType | undefined,
  });

  const [viewSample, setViewSample] = useState<{ sampleId: number; imageData: string; label: string; eventName: string } | null>(null);

  const [regionEvent, setRegionEvent] = useState<TrainingEventDefinition | null>(null);
  const regionCanvasRef = useRef<HTMLCanvasElement>(null);
  const [regionBox, setRegionBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const regionDrawing = useRef(false);
  const rsx = useRef(0); const rsy = useRef(0);
  const rex = useRef(0); const rey = useRef(0);
  const [regionBg, setRegionBg] = useState<string | null>(null);
  const regionBgRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exportCmdRef = useRef<HTMLInputElement>(null);

  const fetchTrainingData = useCallback(() => {
    const game = selectedGame;
    if (!game) return;
    setLoadingData(true);
    setTrainLog([]);
    setTrainResult(null);
    sendMessageToBackend('GetTrainingEvents', { gameId: game });
    sendMessageToBackend('GetTrainingModelStatus', { gameId: game });
  }, [selectedGame]);

  useEffect(() => {
    fetchTrainingData();
  }, [fetchTrainingData]);

  useEffect(() => {
    const handler = (e: CustomEvent) => {
      const data = e.detail;
      if (data.method === 'TrainingEvents') {
        setEvents(data.content.events || []);
        setSampleCounts(data.content.sampleCounts || {});
        setSamples(data.content.samples || []);
        setLoadingData(false);
      } else if (data.method === 'TrainingEventSaved') {
        if (data.content.success) fetchTrainingData();
        setShowEventModal(false);
        setEditingEvent(null);
      } else if (data.method === 'TrainingEventDeleted') {
        if (data.content.success) fetchTrainingData();
      } else if (data.method === 'TrainingSampleAdded') {
        if (data.content.success) fetchTrainingData();
      } else if (data.method === 'TrainingDatasetExported') {
        setExportResult(data.content);
      } else if (data.method === 'TrainingModelStatus') {
        setModelStatus(data.content);
      } else if (data.method === 'TrainingSampleImage') {
        if (data.content.success) setViewSample(data.content);
      } else if (data.method === 'TrainingModelLoaded') {
        if (data.content.success) fetchTrainingData();
      } else if (data.method === 'TrainingProgress') {
        setTrainLog(prev => [...prev, { status: data.content.status, message: data.content.message }]);
        setTrainResult({ status: data.content.status, message: data.content.message });

        // Parse YOLO metric lines (strip ANSI codes first)
        const msg = (data.content.message as string).replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '').replace(/\r/g, '').trim();
        // Epoch line: "1/100         0G      2.512      9.956       1.52         27        640"
        const epochMatch = msg.match(/^(\d+)\/(\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/);
        // Metric line: "all          6          6    0.00115        0.5      0.497      0.249"
        const metricMatch = msg.match(/^all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/);
        if (metricMatch) {
          setTrainMetrics(prev => {
            const mAP50 = parseFloat(metricMatch[5]);
            return {
              epoch: prev?.epoch ?? 0,
              totalEpochs: prev?.totalEpochs ?? 100,
              boxLoss: prev?.boxLoss ?? 0,
              clsLoss: prev?.clsLoss ?? 0,
              dflLoss: prev?.dflLoss ?? 0,
              precision: parseFloat(metricMatch[3]),
              recall: parseFloat(metricMatch[4]),
              mAP50,
              mAP5095: parseFloat(metricMatch[6]),
              bestMAP50: Math.max(prev?.bestMAP50 ?? 0, mAP50),
            };
          });
        }
        if (epochMatch) {
          setTrainMetrics(prev => ({
            epoch: parseInt(epochMatch[1]),
            totalEpochs: parseInt(epochMatch[2]),
            boxLoss: parseFloat(epochMatch[3]),
            clsLoss: parseFloat(epochMatch[4]),
            dflLoss: parseFloat(epochMatch[5]),
            precision: prev?.precision ?? 0,
            recall: prev?.recall ?? 0,
            mAP50: prev?.mAP50 ?? 0,
            mAP5095: prev?.mAP5095 ?? 0,
            bestMAP50: prev?.bestMAP50 ?? 0,
          }));
        }
        if (data.content.status === 'completed' || data.content.status === 'error') {
          fetchTrainingData();
        }
      }
    };

    window.addEventListener('websocket-message', handler as EventListener);
    return () => window.removeEventListener('websocket-message', handler as EventListener);
  }, [selectedGame]);

const openAddEvent = () => {
    setEditingEvent(null);
    setEventForm({ name: '', type: 'Trigger', bookmarkType: BookmarkType.Kill });
    setShowEventModal(true);
};

  const openEditEvent = (ev: TrainingEventDefinition) => {
    setEditingEvent(ev);
    setEventForm({
      name: ev.name,
      type: ev.type,
      bookmarkType: ev.bookmarkType,
    });
    setShowEventModal(true);
  };

  const saveEvent = () => {
    if (!eventForm.name.trim()) return;
    const event: TrainingEventDefinition = {
      id: editingEvent?.id ?? ((Date.now() % 2000000000) + Math.floor(Math.random() * 1000)),
      name: eventForm.name,
      type: eventForm.type,
      classId: editingEvent?.classId ?? events.length + 1,
      bookmarkType: eventForm.bookmarkType,
      screenRegionX: editingEvent?.screenRegionX,
      screenRegionY: editingEvent?.screenRegionY,
      screenRegionW: editingEvent?.screenRegionW,
      screenRegionH: editingEvent?.screenRegionH,
    };
    sendMessageToBackend('SaveTrainingEvent', { gameId: selectedGame, event });
  };

  const deleteEvent = (eventId: number) => {
    sendMessageToBackend('DeleteTrainingEvent', { gameId: selectedGame, eventId });
  };

  const deleteSample = (sampleId: number) => {
    sendMessageToBackend('DeleteTrainingSample', { gameId: selectedGame, sampleId });
  };

  const exportDataset = () => {
    setExportResult(null);
    setExporting(true);
    setExportProgress(0);
    sendMessageToBackend('ExportTrainingDataset', { gameId: selectedGame });
    // Simulate progress steps for UX
    let p = 0;
    const iv = setInterval(() => { p += 1; setExportProgress(Math.min(p, 95)); if (p >= 95) clearInterval(iv); }, 300);
    const handler2 = (e: CustomEvent) => {
      const d = e.detail;
      if (d.method === 'TrainingDatasetExported' || d.method === 'TrainingProgress') {
        clearInterval(iv);
        setExportProgress(100);
        setTimeout(() => setExporting(false), 500);
        window.removeEventListener('websocket-message', handler2 as EventListener);
      }
    };
    window.addEventListener('websocket-message', handler2 as EventListener);
  };

  const loadModel = () => {
    sendMessageToBackend('LoadTrainingModel', { gameId: selectedGame });
  };

  const trainModel = () => {
    if (totalSamples === 0) return;
    setIsTraining(true);
    setTrainLog([]);
    setTrainResult({ status: 'starting', message: 'Starting training... (this may take a while)' });
    sendMessageToBackend('TrainModel', { gameId: selectedGame });
  };

  useEffect(() => {
    if (trainResult?.status === 'completed' || trainResult?.status === 'error')
      setIsTraining(false);
  }, [trainResult]);

  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = logContainerRef.current;
    if (!el) return;
    // Only auto-scroll if already near the bottom
    const threshold = 50;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    if (isNearBottom) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainLog]);

  useEffect(() => {
    const c = regionCanvasRef.current;
    if (!c || !regionEvent) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w; c.height = h;
    ctx.clearRect(0, 0, w, h);

    // Background screenshot
    const img = regionBgRef.current;
    if (img) {
      ctx.drawImage(img, 0, 0, w, h);
    } else {
      // Dark background with grid
      ctx.fillStyle = '#1a1a2e';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
      for (let x = 0; x <= w; x += w / 3) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y <= h; y += h / 3) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    }

    const box = regionBox;
    if (box) {
      const bx = box.x * w, by = box.y * h, bw = box.w * w, bh = box.h * h;
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(245,158,11,0.1)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeRect(bx, by, bw, bh);
    }
  }, [regionEvent, regionBox, regionBg]);

  const copyExportCmd = () => {
    exportCmdRef.current?.select();
    navigator.clipboard?.writeText(exportCmdRef.current?.value ?? '');
  };

  const totalSamples = Object.values(sampleCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="min-h-full bg-base-200 dark:bg-base-300 p-5 space-y-6">
      {/* Game Selector */}
      <div className="card bg-base-200">
        <div className="card-body p-4">
          <div className="flex items-center gap-4">
            <label className="text-sm font-semibold text-base-content">Game</label>
            <select
              className="select select-bordered w-64"
              value={selectedGame}
              onChange={(e) => setSelectedGame(e.target.value)}
            >
              {games.length === 0 && <option value="">No games with content</option>}
              {games.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Event List */}
      <div className="card bg-base-200">
        <div className="card-body p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Training Events</h2>
            <button className="btn btn-primary btn-sm" onClick={openAddEvent}>
              Add Event
            </button>
          </div>

          {events.length === 0 ? (
            <p className="text-base-content/60 text-sm">No events defined. Click "Add Event" to create one.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="table table-zebra w-full">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Samples</th>
                    <th>Pre-time (ms)</th>
                    <th>Region</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => (
                    <tr key={ev.id}>
                      <td className="font-medium">{ev.name}</td>
                      <td>
                        <span className={`badge badge-sm ${ev.type === 'Trigger' ? 'badge-primary' : 'badge-warning'}`}>
                          {ev.type}
                        </span>
                      </td>
                      <td>{sampleCounts[ev.id] ?? 0}</td>
                      <td>
                        <span className={`badge badge-xs ${ev.screenRegionW ? 'badge-success' : 'badge-ghost'}`}>
                          {ev.screenRegionW ? 'Set' : 'None'}
                        </span>
                      </td>
                      <td className="flex gap-2">
                        <button className="btn btn-ghost btn-xs" onClick={() => {
                          setRegionEvent(ev);
                          setRegionBox(ev.screenRegionW ? { x: ev.screenRegionX!, y: ev.screenRegionY!, w: ev.screenRegionW!, h: ev.screenRegionH! } : null);
                          setRegionBg(null); regionBgRef.current = null;
                        }}>
                          Region
                        </button>
                        <button className="btn btn-ghost btn-xs" onClick={() => openEditEvent(ev)}>
                          Edit
                        </button>
                        <button className="btn btn-ghost btn-xs text-error" onClick={() => deleteEvent(ev.id)}>
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Model Status */}
      <div className="card bg-base-200">
        <div className="card-body p-4">
          <h2 className="text-lg font-semibold mb-4">Model Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-base-300 rounded-lg p-3">
              <div className="text-xs text-base-content/60 mb-1">Trained Model</div>
              <div className="flex items-center gap-2">
                <span className={`badge badge-sm ${modelStatus?.hasModel ? 'badge-success' : 'badge-ghost'}`}>
                  {modelStatus?.hasModel ? 'Available' : 'None'}
                </span>
              </div>
            </div>
            <div className="bg-base-300 rounded-lg p-3">
              <div className="text-xs text-base-content/60 mb-1">Total Samples</div>
              <div className="text-lg font-bold">{totalSamples}</div>
            </div>
            <div className="bg-base-300 rounded-lg p-3">
              <div className="text-xs text-base-content/60 mb-1">Last Trained</div>
              <div className="text-sm">
                {modelStatus?.lastTrained ? new Date(modelStatus.lastTrained).toLocaleDateString() : 'Never'}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <button className="btn btn-primary btn-sm" onClick={exportDataset} disabled={totalSamples === 0}>
              Export Dataset
            </button>
            <button className="btn btn-secondary btn-sm" onClick={trainModel}
               disabled={totalSamples === 0 || isTraining}>
              {isTraining ? 'Training...' : 'Train Model'}
            </button>
            <button className="btn btn-outline btn-sm" onClick={loadModel}>
              Load Model
            </button>
          </div>

          {trainMetrics && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-base-300 rounded-lg p-3">
                <div className="text-xs text-base-content/60 mb-1">Epoch</div>
                <div className="text-lg font-bold">{trainMetrics.epoch}/{trainMetrics.totalEpochs}</div>
              </div>
              <div className="bg-base-300 rounded-lg p-3">
                <div className="text-xs text-base-content/60 mb-1">mAP50</div>
                <div className="text-lg font-bold">{trainMetrics.mAP50.toFixed(3)}</div>
                <div className="text-xs text-base-content/40">Best: {trainMetrics.bestMAP50.toFixed(3)}</div>
              </div>
              <div className="bg-base-300 rounded-lg p-3">
                <div className="text-xs text-base-content/60 mb-1">Precision</div>
                <div className="text-lg font-bold">{trainMetrics.precision.toFixed(4)}</div>
              </div>
              <div className="bg-base-300 rounded-lg p-3">
                <div className="text-xs text-base-content/60 mb-1">Recall</div>
                <div className="text-lg font-bold">{trainMetrics.recall.toFixed(3)}</div>
              </div>
              <div className="bg-base-300 rounded-lg p-3">
                <div className="text-xs text-base-content/60 mb-1">Box Loss</div>
                <div className="text-sm font-mono">{trainMetrics.boxLoss.toFixed(3)}</div>
              </div>
              <div className="bg-base-300 rounded-lg p-3">
                <div className="text-xs text-base-content/60 mb-1">Class Loss</div>
                <div className="text-sm font-mono">{trainMetrics.clsLoss.toFixed(3)}</div>
              </div>
            </div>
          )}
          {trainResult && (trainResult.status === 'completed' || trainResult.status === 'error') && (
            <div className={`mt-4 p-3 rounded-lg ${trainResult.status === 'error' ? 'bg-error/20 text-error' : 'bg-success/20 text-success'}`}>
              <p className="text-sm whitespace-pre-wrap">{trainResult.message}</p>
            </div>
          )}

          {exportResult && (
            <div className="mt-4 p-3 bg-base-300 rounded-lg">
              <p className="text-sm text-base-content/60 mb-2">
                Dataset exported. Run this command to train:
              </p>
              <div className="flex items-center gap-2">
                <input
                  ref={exportCmdRef}
                  className="input input-bordered input-sm flex-1 font-mono text-xs"
                  value={`python train_yolo.py data/training/${exportResult.gameId}/`}
                  readOnly
                />
                <button className="btn btn-ghost btn-xs" onClick={copyExportCmd}>
                  Copy
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sample List */}
      <div className="card bg-base-200">
        <div className="card-body p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Samples ({samples.length})</h2>
          </div>
          {samples.length === 0 ? (
            <p className="text-base-content/60 text-sm">No samples available. Add training samples from recordings.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="table table-zebra w-full">
                <thead>
                  <tr>
                    <th>Labels</th>
                    <th>Events</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((s) => (
                    <tr key={s.id}>
                      <td className="text-xs opacity-60">{(s.labelCount ?? 1)}</td>
                      <td>
                        <div className="flex flex-wrap gap-1">
                          {(s.eventNames ?? [s.eventName]).map((name, i) => (
                            <span key={i} className="badge badge-sm badge-primary">{name}</span>
                          ))}
                        </div>
                      </td>
                      <td className="flex gap-2">
                        <button className="btn btn-ghost btn-xs" onClick={() => sendMessageToBackend('GetTrainingSampleImage', { gameId: selectedGame, sampleId: s.id })}>
                          Edit
                        </button>
                        <button className="btn btn-ghost btn-xs text-error" onClick={() => deleteSample(s.id)}>
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Event Modal */}
      {showEventModal && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-4">
              {editingEvent ? 'Edit Event' : 'Add Event'}
            </h3>
            <div className="space-y-4">
              <div className="form-control">
                <label className="label">
                  <span className="label-text">Event Name</span>
                </label>
                <input
                  type="text"
                  className="input input-bordered"
                  value={eventForm.name}
                  onChange={(e) => setEventForm({ ...eventForm, name: e.target.value })}
                  placeholder="e.g. Kill, Goal, Death"
                />
              </div>
              <div className="form-control">
                <label className="label">
                  <span className="label-text">Type</span>
                </label>
                <select
                  className="select select-bordered"
                  value={eventForm.type}
                  onChange={(e) => {
                    const newType = e.target.value as 'Trigger' | 'Exclusion';
                    setEventForm({
                      ...eventForm,
                      type: newType,
                      bookmarkType: newType === 'Exclusion' ? undefined : eventForm.bookmarkType ?? BookmarkType.Kill,
                    });
                  }}
                >
                  <option value="Trigger">Trigger</option>
                  <option value="Exclusion">Exclusion</option>
                </select>
              </div>
              <div className="form-control">
                <label className="label">
                  <span className="label-text">Bookmark Type</span>
                </label>
                <select
                  className="select select-bordered"
                  value={eventForm.bookmarkType ?? ''}
                  disabled={eventForm.type === 'Exclusion'}
                  onChange={(e) => setEventForm({ ...eventForm, bookmarkType: (e.target.value || undefined) as BookmarkType | undefined })}
                >
                  <option value="">None</option>
                  <option value="Kill">Kill</option>
                  <option value="Goal">Goal</option>
                  <option value="Assist">Assist</option>
                  <option value="Death">Death</option>
                </select>
              </div>

            </div>
            <div className="modal-action">
              <button className="btn btn-ghost" onClick={() => { setShowEventModal(false); setEditingEvent(null); }}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={saveEvent}>
                {editingEvent ? 'Save Changes' : 'Create Event'}
              </button>
            </div>
          </div>
          <div className="modal-backdrop" onClick={() => { setShowEventModal(false); setEditingEvent(null); }} />
        </div>
      )}

      {/* Loading Overlay */}
      {loadingData && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40">
          <div className="bg-base-200 rounded-lg p-6 flex items-center gap-4">
            <span className="loading loading-spinner loading-lg text-primary" />
            <div>
              <p className="font-semibold">Loading training data...</p>
              <p className="text-xs opacity-60">Fetching events, samples, and model status</p>
            </div>
          </div>
        </div>
      )}

      {/* Export Progress Modal */}
      {exporting && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-4">Exporting Dataset</h3>
            <div className="flex items-center gap-3 mb-2">
              <span className="loading loading-spinner text-primary" />
              <span className="text-sm">Cropping samples to screen regions & building dataset...</span>
            </div>
            <progress className="progress progress-primary w-full" value={exportProgress} max="100" />
            <p className="text-xs text-right mt-1 opacity-60">{exportProgress}%</p>
          </div>
        </div>
      )}

      {/* Sample Viewer Modal */}
      {viewSample && (() => {
        // Parse labels into editable array: { eventId, boxX, boxY, boxW, boxH }
        const initialLabels: { eventId: number; boxX: number; boxY: number; boxW: number; boxH: number }[] = [];
        if (viewSample.label) {
          for (const line of viewSample.label.split('\n')) {
            const parts = line.trim().split(' ');
            if (parts.length >= 5) {
              const [eid, bx, by, bw, bh] = parts.map(Number);
              if (!isNaN(eid)) initialLabels.push({ eventId: eid, boxX: bx, boxY: by, boxW: bw, boxH: bh });
            }
          }
        }

        const Editor = () => {
          const canvasRef = useRef<HTMLCanvasElement>(null);
          const [labels, setLabels] = useState(initialLabels);
          const [sel, setSel] = useState<number | null>(null);
          const [imgLoaded, setImgLoaded] = useState(false);
          const imgRef = useRef<HTMLImageElement | null>(null);

          // Drawing state
          const isDown = useRef(false);
          const sx = useRef(0); const sy = useRef(0);
          const ex = useRef(0); const ey = useRef(0);
          const moving = useRef(false);
          const msx = useRef(0); const msy = useRef(0);
          const bs = useRef({ x: 0, y: 0, w: 0, h: 0 });
          const rh = useRef<string | null>(null);

          const draw = useCallback(() => {
            const c = canvasRef.current;
            const img = imgRef.current;
            if (!c || !img) return;
            const r = c.getBoundingClientRect();
            c.width = r.width * (window.devicePixelRatio || 1);
            c.height = r.height * (window.devicePixelRatio || 1);
            const ctx = c.getContext('2d');
            if (!ctx) return;
            ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
            ctx.drawImage(img, 0, 0, r.width, r.height);

            for (let i = 0; i < labels.length; i++) {
              const l = labels[i];
              const x = (l.boxX - l.boxW / 2) * r.width;
              const y = (l.boxY - l.boxH / 2) * r.height;
              const w = l.boxW * r.width;
              const h = l.boxH * r.height;
              const s = i === sel;
              ctx.strokeStyle = s ? '#3b82f6' : '#22c55e';
              ctx.lineWidth = s ? 2.5 : 2;
              ctx.fillStyle = s ? 'rgba(59,130,246,0.12)' : 'rgba(34,197,94,0.10)';
              ctx.fillRect(x, y, w, h);
              ctx.strokeRect(x, y, w, h);
              const ev = events.find(e => e.id === l.eventId);
              const label = ev?.name ?? `Event ${l.eventId}`;
              ctx.font = '11px sans-serif';
              const tw = ctx.measureText(label).width;
              ctx.fillStyle = 'rgba(0,0,0,0.75)';
              ctx.fillRect(x - 2, y - 17, tw + 6, 16);
              ctx.fillStyle = '#fff';
              ctx.fillText(label, x, y - 5);
              if (s) {
                ctx.fillStyle = '#fff'; ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5;
                for (const [px, py] of [[x, y], [x + w / 2, y], [x + w, y],
                  [x + w, y + h / 2], [x + w, y + h],
                  [x + w / 2, y + h], [x, y + h], [x, y + h / 2]]) {
                  ctx.fillRect(px - 4, py - 4, 8, 8); ctx.strokeRect(px - 4, py - 4, 8, 8);
                }
              }
            }
          }, [labels, sel, events]);

          useEffect(() => {
            if (!imgLoaded) return;
            draw();
            let running = true;
            const loop = () => { if (!running) return; draw(); requestAnimationFrame(loop); };
            requestAnimationFrame(loop);
            return () => { running = false; };
          }, [draw, imgLoaded]);

          useEffect(() => {
            if (!imgLoaded) return;
            const c = canvasRef.current;
            if (!c) return;

            const ondown = (e: MouseEvent) => {
              const r = c.getBoundingClientRect();
              const px = (e.clientX - r.left) / r.width;
              const py = (e.clientY - r.top) / r.height;

              // Check handle hit
              if (sel !== null) {
                const l = labels[sel];
                const bx = (l.boxX - l.boxW / 2) * r.width;
                const by = (l.boxY - l.boxH / 2) * r.height;
                const bw = l.boxW * r.width;
                const bh = l.boxH * r.height;
                const epx = e.clientX - r.left;
                const epy = e.clientY - r.top;
                for (const [hx, hy, hh] of [[bx, by, 'nw'], [bx + bw / 2, by, 'n'], [bx + bw, by, 'ne'],
                  [bx + bw, by + bh / 2, 'e'], [bx + bw, by + bh, 'se'],
                  [bx + bw / 2, by + bh, 's'], [bx, by + bh, 'sw'], [bx, by + bh / 2, 'w']] as const) {
                  if (Math.abs(epx - hx) <= 10 && Math.abs(epy - hy) <= 10) {
                    rh.current = hh; bs.current = { x: l.boxX, y: l.boxY, w: l.boxW, h: l.boxH };
                    return;
                  }
                }
              }

              // Check box body hit
              for (let i = labels.length - 1; i >= 0; i--) {
                const l = labels[i];
                const lx = l.boxX - l.boxW / 2;
                const ly = l.boxY - l.boxH / 2;
                if (px >= lx && px <= lx + l.boxW && py >= ly && py <= ly + l.boxH) {
                  setSel(i); moving.current = true; msx.current = px; msy.current = py;
                  bs.current = { x: l.boxX, y: l.boxY, w: l.boxW, h: l.boxH };
                  return;
                }
              }

              // Start new draw
              setSel(null);
              isDown.current = true;
              sx.current = px; sy.current = py; ex.current = px; ey.current = py;
            };

            const onmove = (e: MouseEvent) => {
              const r = c.getBoundingClientRect();
              const px = (e.clientX - r.left) / r.width;
              const py = (e.clientY - r.top) / r.height;

              if (rh.current && sel !== null) {
                const h = rh.current, s = bs.current;
                const dx = px - msx.current, dy = py - msy.current;
                let x = s.x, y = s.y, bw = s.w, bh = s.h;
                if (h.includes('w')) { x = s.x + dx; bw = s.w - dx; }
                if (h.includes('e')) bw = s.w + dx;
                if (h.includes('n')) { y = s.y + dy; bh = s.h - dy; }
                if (h.includes('s')) bh = s.h + dy;
                if (bw < 0.005) { bw = 0.005; if (h.includes('w')) x = s.x + s.w - 0.005; }
                if (bh < 0.005) { bh = 0.005; if (h.includes('n')) y = s.y + s.h - 0.005; }
                setLabels(prev => { const n = [...prev]; n[sel] = { ...n[sel], boxX: x, boxY: y, boxW: bw, boxH: bh }; return n; });
                return;
              }

              if (moving.current && sel !== null) {
                const dx = px - msx.current, dy = py - msy.current;
                setLabels(prev => { const n = [...prev]; n[sel] = { ...n[sel], boxX: bs.current.x + dx, boxY: bs.current.y + dy, boxW: bs.current.w, boxH: bs.current.h }; return n; });
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
              if (w < 0.005 || h < 0.005) return;
              const x = Math.min(sx.current, ex.current), y = Math.min(sy.current, ey.current);
              const newLabel = { eventId: 0, boxX: x + w / 2, boxY: y + h / 2, boxW: w, boxH: h };
              setLabels(prev => { setSel(prev.length); return [...prev, newLabel]; });
            };

            window.addEventListener('mousedown', ondown);
            window.addEventListener('mousemove', onmove);
            window.addEventListener('mouseup', onup);
            return () => { window.removeEventListener('mousedown', ondown); window.removeEventListener('mousemove', onmove); window.removeEventListener('mouseup', onup); };
          }, [labels, sel, imgLoaded]);

          return (
            <div className="space-y-3">
              <div className="bg-base-300 rounded-lg overflow-hidden relative" style={{ maxHeight: '65vh' }}>
                <img src={viewSample.imageData} className="hidden" alt=""
                  onLoad={(e) => { imgRef.current = e.currentTarget; setImgLoaded(true); }} />
                <canvas ref={canvasRef} className="w-full cursor-crosshair" style={{ display: imgLoaded ? 'block' : 'none' }} />
                {!imgLoaded && <div className="flex items-center justify-center h-64"><span className="loading loading-spinner" /></div>}
              </div>
              {sel !== null && labels[sel] && (
                <div className="flex gap-2 items-center">
                  <select className="select select-bordered select-xs flex-1"
                    value={labels[sel].eventId}
                    onChange={(e) => setLabels(prev => { const n = [...prev]; n[sel] = { ...n[sel], eventId: Number(e.target.value) }; return n; })}>
                    <option value={0}>Select event...</option>
                    {events.map(ev => (
                      <option key={ev.id} value={ev.id}>{ev.name}</option>
                    ))}
                  </select>
                  <button className="btn btn-ghost btn-xs text-error" onClick={() => {
                    setLabels(prev => { const n = [...prev]; n.splice(sel, 1); return n; });
                    setSel(null);
                  }}>Delete</button>
                </div>
              )}
              <div className="flex gap-2">
                <button className="btn btn-primary btn-sm flex-1" onClick={() => {
                  sendMessageToBackend('UpdateTrainingSampleLabels', {
                    gameId: selectedGame,
                    sampleId: viewSample.sampleId,
                    labels: labels.map(l => ({
                      eventId: l.eventId,
                      boxX: Math.round(l.boxX * 10000) / 10000,
                      boxY: Math.round(l.boxY * 10000) / 10000,
                      boxW: Math.round(l.boxW * 10000) / 10000,
                      boxH: Math.round(l.boxH * 10000) / 10000,
                    })),
                  });
                  setViewSample(null);
                }}>Save Changes</button>
                <button className="btn btn-ghost btn-sm" onClick={() => setViewSample(null)}>Cancel</button>
              </div>
            </div>
          );
        };

        return (
          <div className="modal modal-open">
            <div className="modal-box max-w-4xl">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-lg">Edit Sample</h3>
                <span className="text-xs opacity-60">{initialLabels.length} original objects</span>
              </div>
              <Editor />
            </div>
            <div className="modal-backdrop" onClick={() => setViewSample(null)} />
          </div>
        );
      })()}

      {/* Region Editor Modal */}
      {regionEvent && (
        <div className="modal modal-open">
          <div className="modal-box max-w-3xl">
            <h3 className="font-bold text-lg mb-2">
              Screen Region: {regionEvent.name}
            </h3>
            <p className="text-xs opacity-60 mb-4">
              Draw where on screen this element typically appears. During live detection, only detections within this region will be considered.
            </p>
            <div className="flex gap-2 mb-2 flex-wrap">
              <button className="btn btn-ghost btn-xs" onClick={() => fileInputRef.current?.click()}>
                Load Screenshot
              </button>
              {regionBg && (
                <button className="btn btn-ghost btn-xs" onClick={() => { setRegionBg(null); regionBgRef.current = null; }}>
                  Clear Screenshot
                </button>
              )}
              <input ref={fileInputRef} type="file" accept="image/png,image/jpeg" className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const reader = new FileReader();
                  reader.onload = () => {
                    const img = new Image();
                    img.onload = () => { regionBgRef.current = img; setRegionBg(reader.result as string); };
                    img.src = reader.result as string;
                  };
                  reader.readAsDataURL(file);
                }} />
              {events.filter(e => e.screenRegionW).length > 0 && (
                <div className="dropdown dropdown-bottom">
                  <button tabIndex={0} className="btn btn-ghost btn-xs">
                    Copy From Existing
                  </button>
                  <ul tabIndex={0} className="dropdown-content z-10 menu p-2 shadow bg-base-200 rounded-box w-52">
                    {events.filter(e => e.screenRegionW && e.id !== regionEvent?.id).map(e => (
                      <li key={e.id}>
                        <button onClick={() => setRegionBox({ x: e.screenRegionX!, y: e.screenRegionY!, w: e.screenRegionW!, h: e.screenRegionH! })}>
                          {e.name}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="bg-base-300 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
              <canvas
                ref={regionCanvasRef}
                className="w-full h-full cursor-crosshair"
                style={{ display: 'block' }}
                onMouseDown={(e) => {
                  const rect = regionCanvasRef.current!.getBoundingClientRect();
                  rsx.current = (e.clientX - rect.left) / rect.width;
                  rsy.current = (e.clientY - rect.top) / rect.height;
                  rex.current = rsx.current; rey.current = rsy.current;
                  regionDrawing.current = true;
                }}
                onMouseMove={(e) => {
                  if (!regionDrawing.current) return;
                  const rect = regionCanvasRef.current!.getBoundingClientRect();
                  rex.current = (e.clientX - rect.left) / rect.width;
                  rey.current = (e.clientY - rect.top) / rect.height;
                  setRegionBox({
                    x: Math.min(rsx.current, rex.current),
                    y: Math.min(rsy.current, rey.current),
                    w: Math.abs(rex.current - rsx.current),
                    h: Math.abs(rey.current - rsy.current),
                  });
                }}
                onMouseUp={() => { regionDrawing.current = false; }}
              />
            </div>
            {regionBox && (
              <div className="mt-2 text-xs opacity-60">
                Region: x={regionBox.x.toFixed(3)} y={regionBox.y.toFixed(3)} w={regionBox.w.toFixed(3)} h={regionBox.h.toFixed(3)}
              </div>
            )}
            <div className="modal-action">
              <button className="btn btn-ghost" onClick={() => { setRegionEvent(null); setRegionBox(null); }}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={() => {
                if (!regionEvent) return;
                const updated = { ...regionEvent };
                if (regionBox) {
                  updated.screenRegionX = Math.round(regionBox.x * 10000) / 10000;
                  updated.screenRegionY = Math.round(regionBox.y * 10000) / 10000;
                  updated.screenRegionW = Math.round(regionBox.w * 10000) / 10000;
                  updated.screenRegionH = Math.round(regionBox.h * 10000) / 10000;
                } else {
                  updated.screenRegionX = undefined;
                  updated.screenRegionY = undefined;
                  updated.screenRegionW = undefined;
                  updated.screenRegionH = undefined;
                }
                sendMessageToBackend('SaveTrainingEvent', { gameId: selectedGame, event: updated });
                setRegionEvent(null);
                setRegionBox(null);
              }}>
                Save Region
              </button>
              {regionBox && (
                <button className="btn btn-ghost btn-xs" onClick={() => setRegionBox(null)}>
                  Clear
                </button>
              )}
            </div>
          </div>
          <div className="modal-backdrop" onClick={() => { setRegionEvent(null); setRegionBox(null); }} />
        </div>
      )}
    </div>
  );
}
