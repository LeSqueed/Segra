# Training Events — Implementation Plan

## Overview

Allow users to train a YOLO nano model to detect on-screen UI elements
("events") during recordings. Events fall into two categories:

- **Trigger events** — auto-create bookmarks when detected (e.g. elimination
  icon, victory screen).
- **Exclusion events** — suppress trigger bookmarks while visible (e.g. kill
  cam overlay, respawn timer).

Detection runs live during recording via ONNX Runtime + `Obs.SubscribeRawVideo()`.
Training is external: the app exports a YOLO-format dataset and the user runs a
Python script to produce an ONNX model.

Feature-gated behind `#if ENABLE_TRAINING_EVENTS` (backend) and a matching
build-time flag (frontend).

---

## Task Breakdown

### T1 — Feature flag + build setup

**Goal**: Toggle the entire feature with a single flag.

- Add `<DefineConstants>$(DefineConstants);ENABLE_TRAINING_EVENTS</DefineConstants>`
  to `Segra.csproj` (both TFM conditionals), but only on the `dev-build` /
  `feat/training-events` branch — **do not commit to LeSqueed branches**.
- Add `Microsoft.ML.OnnxRuntime` NuGet package (CPU, ~12 MB) to the
  cross-platform `ItemGroup` in `Segra.csproj`.
- Frontend: add a `__ENABLE_TRAINING_EVENTS__` define in `vite.config.ts`
  (set to `true` on this branch).
- Add `Training` to the `MenuItemId` type in `Frontend/src/Models/types.ts`,
  conditionally compiled via the flag.
- Create stub files so the build passes:
  - `Backend/Training/TrainingEventService.cs` — empty class
  - `Frontend/src/Pages/training.tsx` — placeholder component

**Verification**: `dotnet build -c Release` succeeds, `npm run build` succeeds,
and a "Training" menu item appears in the sidebar.

---

### T2 — Backend: TrainingEventService skeleton

**Goal**: Core service that manages per-game training data and loads models.

`Backend/Training/TrainingEventService.cs`:

```
TrainingEventService (static)
├── Initialize()
├── EventDefinitions: Dictionary<int, TrainingEventDefinition>
├── LoadEventDefinitions(gameId)
├── SaveEventDefinitions(gameId)
├── Samples: List<TrainingSample>
├── AddSample(frame, boundingBox, eventId, gameId)
│   ├── saves crop PNG to data/training/{gameId}/samples/
│   ├── appends label to YOLO .txt
│   └── updates events.json
├── ExportDataset(gameId) → writes dataset.yaml
└── LoadModel(gameId) → loads model.onnx via ONNX Runtime session
```

Supporting types in `Backend/Training/Models.cs`:

```
TrainingEventDefinition
├── Id: int
├── Name: string
├── Type: EventType (Trigger | Exclusion)
├── ClassId: int
├── CooldownMs: int
├── PreTimeMs: int
└── PostTimeMs: int

TrainingSample
├── ImagePath: string
├── LabelPath: string
├── EventId: int
├── Timestamp: TimeSpan
└── RecordingFile: string

TrainingEventDetectionResult
├── EventId: int
├── Confidence: float
├── BoundingBox: (x, y, w, h)
└── Timestamp: DateTime
```

**Verification**: Unit tests that create/load event definitions, add samples,
and export a valid YOLO dataset structure.

---

### T3 — Backend: TrainingEventDetector (real-time inference)

**Goal**: Subscribe to OBS raw video frames, run ONNX inference at a
configurable rate, and produce detection results.

`Backend/Training/TrainingEventDetector.cs`:

```
TrainingEventDetector
├── Start(gameId, outputWidth, outputHeight)
│   ├── LoadModel(gameId)
│   ├── Obs.SubscribeRawVideo(BGRA, targetW, targetH, OnFrame, divisor)
│   └── Start detection timer (configurable ms, default 1000)
├── Stop()
│   ├── Unsubscribe from raw video
│   └── Dispose model session
├── OnFrame(RawVideoFrame frame)
│   ├── Copy pixel data to managed buffer (non-blocking on video thread)
│   └── Enqueue for inference on background thread
├── DetectionLoop()
│   ├── Dequeue frame buffer at configured interval
│   ├── Resize to model input size (YOLO nano: 640×640 or 320×320)
│   ├── Run ONNX inference → List<DetectionResult>
│   └── Return results to caller
└── DetectAsync(byte[] bgraData, int width, int height) → List<DetectionResult>
```

- Use `Task.Run` to offload inference from the OBS video thread.
- Keep a ring buffer of the last N frames so we don't skip detections if
  inference takes longer than the interval.
- YOLO nano model input: typically 640×640, RGB, normalized to [0,1].

**Verification**: Unit test with a known ONNX model and synthetic frame data
returns expected detection results.

---

### T4 — Backend: Event lifecycle manager

**Goal**: Track active events, handle trigger/exclusion logic, create bookmarks.

`Backend/Training/TrainingEventLifecycle.cs`:

```
TrainingEventLifecycle
├── ActiveTriggers: Dictionary<int, ActiveEvent>  // eventId → first/last seen
├── ActiveExclusions: DateTime?  // when the current exclusion started
├── ExclusionCooldownMs: int
│
├── ProcessDetections(detections: List<DetectionResult>, now: DateTime)
│   ├── Update ActiveExclusions based on detected exclusion events
│   ├── For each detected trigger:
│   │   ├── If exclusion active → log but skip bookmark
│   │   ├── If not in ActiveTriggers → start new event (firstSeen=now)
│   │   ├── If already active → extend lastSeen=now
│   ├── For each active trigger NOT in this tick's detections:
│   │   ├── Mark as ended → create bookmark at firstSeen
│   │   └── Remove from ActiveTriggers
│   └── Exclusions: if not detected this tick → clear exclusion

ActiveEvent
├── EventDefinition
├── FirstSeen: DateTime
├── LastSeen: DateTime
└── BookmarkCreated: bool

// Bookmark is created with:
// - Type = BookmarkType.TrainingEvent
// - Time = firstSeen - recording.StartTime
// - TrainingEventName stored for frontend rendering
```

- Add `BookmarkType.TrainingEvent` to the backend `BookmarkType` enum.
- Add a `TrainingEventName` property to the `Bookmark` class (nullable string).
- The bookmark's `Time` is the `firstSeen` moment. Pre/post times are stored
  on the event definition and applied during clip extraction.

**Verification**: Unit test simulating a sequence of detections over time:
t=1 trigger A detected, t=2 trigger A detected (extended), t=3 no detection
(ended), t=5 trigger A detected (new event). Verify exactly 2 bookmarks created.

---

### T5 — Backend: Wire into GameIntegrationService

**Goal**: Start/stop the training event detector alongside game-specific
integrations.

In `GameIntegrationService.Start()`:

```
if (ENABLE_TRAINING_EVENTS)
{
    var gameId = resolvedGameId;
    if (TrainingEventService.HasModelForGame(gameId))
    {
        _trainingDetector = new TrainingEventDetector();
        _trainingDetector.Start(gameId, outputWidth, outputHeight);
        _trainingLifecycle = new TrainingEventLifecycle(eventDefs);
    }
}
```

In `GameIntegrationService.Shutdown()`:

```
_trainingDetector?.Stop();
_trainingLifecycle = null;
```

The detector callback calls `_trainingLifecycle.ProcessDetections()` which
calls `recording.AddBookmark()`.

- `GameIntegrationService` already starts in `CompleteRecordingStart`.
- OBSService already calls `GameIntegrationService.Start()` with `igdbId`,
  `gameName`, `exePath`. We need `outputWidth`/`outputHeight` too — either
  pass them or read from `_currentBaseWidth`/`_currentBaseHeight`.
- The detector needs access to the OBS raw video subscription, which is
  managed by `TrainingEventDetector` internally.

**Verification**: Integration test that starts a recording, verifies the
detector subscribes to raw video, and stops cleanly.

---

### T6 — Backend: Sample acquisition endpoint

**Goal**: Backend API to receive training samples from the frontend.

In `Backend/Training/TrainingController.cs` (or add to an existing controller):

```
POST /api/training/sample
├── Body: { gameId, eventId, timestamp, recordingFile, boundingBox: {x,y,w,h}, frameBase64 }
├── TrainingEventService.AddSample(...)
└── Returns: { success, sampleCount }

GET /api/training/events?gameId=...
└── Returns: List<TrainingEventDefinition>

POST /api/training/events
├── Body: { gameId, eventDefinition }
├── TrainingEventService.SaveEventDefinitions(gameId)
└── Returns: { success }

POST /api/training/export?gameId=...
├── TrainingEventService.ExportDataset(gameId)
├── Returns: { datasetPath, sampleCount, command: "python train.py ..." }

GET /api/training/model-status?gameId=...
├── Checks for model.onnx existence + metadata
└── Returns: { hasModel, sampleCount, trainedAt? }

POST /api/training/load-model?gameId=...
├── TrainingEventService.LoadModel(gameId)
└── Returns: { success }
```

These are simple endpoints called from the frontend. The frame is sent as a
base64-encoded PNG (taken from the video player's `<canvas>`).

**Verification**: Call each endpoint manually via curl/inspector and verify
files are created on disk.

---

### T7 — Frontend: Training tab page

**Goal**: New page at `/training` route accessible from the sidebar.

`Frontend/src/Pages/training.tsx`:

```
TrainingPage
├── GameSelectionHeader
│   └── Dropdown of all games that have recordings
├── EventList
│   ├── Each event: name, type badge (trigger/exclusion), sample count
│   ├── Add Event button → opens EventEditorModal
│   ├── Edit/Delete buttons per event
│   └── Pre/post time inputs (seconds)
├── ModelStatusCard
│   ├── Shows: has model ✓/✗, sample count per event, last trained date
│   └── Train button → calls export endpoint, shows CLI command
├── SampleGallery
│   ├── Grid of cropped training samples with event label overlay
│   └── Delete sample button
└── NoTrainingEventsPlaceholder
    └── "Open a recording, pause on a UI element, and draw a box to get started"
```

- Add `Training` to `MenuItemId` type (conditional on feature flag).
- Add to `DEFAULT_MENU_ITEMS` and `MENU_ICONS` in `menu.tsx`.
- Add `Training` case to `renderContent()` in `App.tsx`.
- Icon: `BrainCircuit` from lucide-react (or `ScanEye`).

**Verification**: Navigate to the Training tab, see the empty state UI, add an
event, see it appear in the list.

---

### T8 — Frontend: Canvas overlay for box drawing

**Goal**: When viewing a video, toggle "training mode" that adds a canvas
overlay for drawing bounding boxes.

`Frontend/src/Components/TrainingOverlay.tsx`:

```
TrainingOverlay
├── Props: videoRef, currentTime, isPaused, onBoxDrawn
├── Canvas element positioned absolutely over the <video>
├── State: isDrawing, startPoint, currentBox, boxes[]
├── On pause + toggle:
│   ├── Show canvas overlay overlay
│   ├── mousedown → startPoint = (x, y)
│   ├── mousemove → draw resizing rectangle
│   └── mouseup → finalize box → open EventPickerModal
├── EventPickerModal
│   ├── Select which event this box belongs to
│   ├── Or create new event inline
│   └── Confirm → capture frame + crop → send to backend
└── Keyboard shortcut: T to toggle training mode
```

Integration in `video.tsx`:

```
// Near the video player element
{enableTrainingEvents && trainingMode && isPaused && (
  <TrainingOverlay
    videoRef={videoRef}
    currentTime={currentTime}
    isPaused={isPaused}
    onBoxDrawn={handleTrainingSample}
  />
)}
```

- Add a "Training Mode" toggle button in the video player controls (next to
  fullscreen, etc.), gated by `__ENABLE_TRAINING_EVENTS__`.
- When a box is drawn, extract the cropped region from the video element
  using `canvas.drawImage(video, cropX, cropY, cropW, cropH)`.
- Send the crop + metadata to the backend via `sendMessageToBackend` or a
  direct HTTP POST (check existing pattern — Segra uses WebSocket for most
  frontend→backend communication).

**Verification**: Pause a video, toggle training mode, draw a box, assign it
to an event, verify the sample appears in the backend filesystem.

---

### T9 — Frontend: Training workflow

**Goal**: Complete the user flow from drawing → collecting → training → using.

- **Sample collection**: User draws boxes across multiple recordings for the
  same game. The Training tab shows sample count per event.
- **Train**: When ready, user clicks "Train". Frontend calls
  `POST /api/training/export`, receives the dataset path + command.
- **CLI command display**: Frontend shows a modal with the command to run
  (e.g., `python train.py data/training/{gameId}/`) and a "Copy to clipboard"
  button.
- **Model detection**: After training, app auto-loads the new model on next
  recording (or user clicks "Load Model").
- **Real-time detection status**: During recording, show detected events
  in the recording status area (e.g., "Detected: Elimination (2), Kill Cam
  active").

**Verification**: Full E2E walkthrough: draw 3 samples → export → export
directory has correct YOLO structure → train → model.onnx appears → next
recording logs detections.

---

### T10 — Python training script

**Goal**: A standalone Python script that trains YOLO nano on the exported
dataset and produces an ONNX model.

`scripts/train_yolo.py`:

```
#!/usr/bin/env python3
# Usage: python train_yolo.py <dataset_dir>

1. Read dataset.yaml
2. Install ultralytics if not present
3. Train YOLO11n (or YOLOv8n) with:
   - imgsz=320 (smaller = faster for UI element detection)
   - epochs=100
   - batch=16
   - patience=20
4. Export to ONNX: model.export(format="onnx", imgsz=320)
5. Copy model.onnx to dataset_dir/
6. Print "Training complete: {dataset_dir}/model.onnx"
```

- Keep it simple: single script, no external dependencies beyond ultralytics
  (which bundles PyTorch).
- Progress output piped to stdout so the user sees training progress.
- Include a `requirements.txt` with `ultralytics>=8.0.0`.

**Verification**: Run on a test dataset with 10+ samples, confirm ONNX model
is produced and loads successfully in C# with ONNX Runtime.

---

### T11 — Integration: Real-time detection during recording

**Goal**: Full integration test — during a recording, detect trained events
and create bookmarks.

- When recording starts and a model exists for the game, `TrainingEventDetector`
  begins running.
- Every N ms, it runs inference on the current frame.
- When a trigger event is detected:
  - If no exclusion is active → enters active tracking → creates bookmark
    when the event disappears.
  - Bookmark has type `TrainingEvent` and `TrainingEventName` set.
- When an exclusion event is detected:
  - Starts exclusion mode → suppresses trigger bookmarks.
  - Exclusion automatically ends when the exclusion visual is no longer detected.
- Bookmarks appear in the timeline with a dedicated icon/color.
- Frontend shows "Detected events: X" in the recording card.

**Verification**: Manual test with a game where the user has trained a model.
Start recording, trigger the UI element, stop recording, verify bookmarks exist.

---

### T12 — Frontend: Event bookmark display in timeline

**Goal**: Training event bookmarks are distinguishable in the video player
timeline.

- Add a new icon for `BookmarkType.TrainingEvent` (e.g., `Crosshair` or a
  custom dot with the event color).
- Hovering shows: event name, time, confidence.
- The bookmark list (right panel in video.tsx) shows training events with
  their event name.
- Exclusions are NOT bookmarked — they only suppress triggers.

**Verification**: Open a recording with training event bookmarks, see them
in the timeline with proper icon and tooltip.

---

### T13 — Polish: Edge cases, error handling, UX

- **No model file**: Graceful fallback — log warning, don't crash recording.
- **ONNX Runtime init failure**: Catch exception, log, disable detection.
- **Disk full during sample saving**: Show error, don't lose the recording.
- **Empty dataset**: Disable "Train" button if fewer than N samples per class.
- **Model loading failure**: Show error in Training tab, keep old model.
- **Training in progress**: Disable "Train" button, show spinner.
- **Frame too dark/blurry for training**: Suggest user chooses clearer frames.
- **Per-game event import/export**: Share trained events between users (future).

---

## Directory Structure Summary

```
Segra/
├── Backend/
│   └── Training/
│       ├── TrainingEventService.cs     # Dataset management + model loading
│       ├── TrainingEventDetector.cs    # Frame subscription + ONNX inference
│       ├── TrainingEventLifecycle.cs   # Event state machine + bookmark creation
│       ├── Models.cs                   # Event definition, sample, detection types
│       └── TrainingController.cs       # HTTP/WebSocket API for frontend
├── Frontend/
│   └── src/
│       ├── Pages/
│       │   └── training.tsx            # Training management page
│       └── Components/
│           └── TrainingOverlay.tsx      # Canvas box-drawing overlay
├── scripts/
│   ├── train_yolo.py                   # YOLO training script
│   └── requirements.txt                # ultralytics
└── data/
    └── training/
        └── {gameId}/
            ├── events.json
            ├── dataset.yaml
            ├── images/train/
            ├── labels/train/
            └── model.onnx
```

## Execution Order

Each task feeds into the next. Implement in order, review with sub-agent
before marking complete.

1. T1  — Feature flag + build setup
2. T2  — TrainingEventService skeleton
3. T6  — Backend API endpoints (needed before frontend can send data)
4. T7  — Frontend: Training tab page
5. T8  — Frontend: Canvas overlay (box drawing)
6. T9  — Frontend: Training workflow (collect → export → train)
7. T10 — Python training script
8. T3  — TrainingEventDetector (real-time inference)
9. T4  — Event lifecycle manager
10. T5  — Wire into GameIntegrationService
11. T11 — Integration testing
12. T12 — Frontend: Event bookmark display in timeline
13. T13 — Polish & edge cases
