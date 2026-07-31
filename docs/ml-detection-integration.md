# ML Detection Integration — Architecture

## Overview

Run an ONNX model on OBS video frames to detect visual UI elements and create Bookmarks, identical to how game integrations (CS2, Dota2) create Kill/Death bookmarks from log data.

The training pipeline (dataset export, YOLO training, sample collection) is a separate dev-only toolchain and not part of this runtime path.

---

## Naming

Current names carry "training" baggage that doesn't reflect the runtime purpose:

| Current | Proposed | Rationale |
|---|---|---|
| `TrainingEventDetector` | `VisualEventDetector` | Detects visual events in video; "training" is misleading at runtime |
| `TrainingEventService` | `ModelService` | Loads/caches ONNX models; single responsibility |
| `TrainingEventDefinition` | `EventDefinition` | Defines what to detect (region, class, type) |
| `TrainingEventDetectionResult` | `DetectionResult` | Output from the model: class, confidence, bounding box |
| `TrainingEventLifecycle` | *(remove entirely)* | Replaced by simple cooldown dedup matching game integration pattern |
| `TrainingEventDetector.cs` | `VisualEventDetector.cs` | |
| `TrainingEventService.cs` | `ModelService.cs` | |
| `TrainingEventLifecycle.cs` | *(delete)* | |
| `Models.cs` | `DetectionModels.cs` | Non-training model types |
| `_trainingDetector` | `_visualDetector` | Field in GameIntegrationService |
| `ENABLE_TRAINING_EVENTS` | `ENABLE_ML_DETECTION` | Feature flag |

The code lives under `Backend/Detection/` instead of `Backend/Training/`.

This mirrors `Backend/Games/CounterStrike2/`, `Backend/Games/Dota2/` — each game integration is a self-contained module that detects events and creates bookmarks. ML detection is the same pattern, just with a model instead of a log file.

---

## Architecture

```
OBS frame (1920x1080 BGRA)
  → VisualEventDetector.DetectionLoop (1fps)
    → BgraToGray (BT.601 luma)
    → For each EventDefinition region:
        → Crop + bilinear resize to 640x640
        → ModelService.RunInference (ONNX session)
        → ParseYoloOutput (class, confidence, box)
    → Filter: confidence >= 0.7, non-exclusion types
    → For each qualifying detection:
        → Dedup by classId + cooldownMs (same as CS2's kill-count check)
        → recording.AddBookmark(new Bookmark {
            Type = BookmarkType.MLDetected,  // new enum value
            Time = DateTime.Now - recording.StartTime,
            Subtype = eventDefinition.Name  // e.g. "Elimination"
          })

  [recording ends]
    → HighlightService checks IncludeInHighlight
    → MLDetected has [IncludeInHighlight]
    → FFmpeg extract [Time - 4s, Time + 4s]
    → Highlight video created
```

### Comparison to CS2 Integration

| Step | CS2 Integration | ML Detection |
|---|---|---|
| Data source | Game state JSON via HTTP | OBS video frames via SubscribeRawVideo |
| Detection | Kill count increments → event | Model output > threshold → event |
| Dedup | `_lastValidKills != currentKills` | `lastDetectedAt[classId] + cooldownMs > now` |
| Bookmark | `AddBookmark(BookmarkType.Kill)` | `recording.AddBookmark(MLDetected, name)` |
| Highlight | `[IncludeInHighlight] Kill` + global 4s padding | `[IncludeInHighlight] MLDetected` + global 4s padding |
| Model file | N/A | `data/{gameId}/model.onnx` (ONNX, 640×640 input) |

---

## File Layout

```
Backend/
  Detection/
    VisualEventDetector.cs    # OBS subscription + detection loop
    ModelService.cs           # ONNX session load/cache/inference
    DetectionModels.cs        # EventDefinition, DetectionResult, RegionGroup
    CooldownTracker.cs        # Simple time-based dedup per class
    ParseYoloOutput.cs        # (inline in VisualEventDetector or standalone)

  Games/
    GameIntegrationService.cs # Starts/stops detector per game; wires events → bookmark

  Core/
    Models/
      Bookmark.cs             # Add MLDetected enum value

Frontend/
  src/
    Models/
      types.ts               # EventDefinition (no pre/post/cooldown — only name, type, region)
    Pages/
      detection.tsx           # (rename from training.tsx or split into dev-only page)
```

Each game that wants its own model gets its data directory:
```
data/
  overwatch/
    model.onnx
    events.json
  marvel-rivals/
    model.onnx
    events.json
```

The game ID from the process detector maps to the directory, same as how CS2 integration maps to `CounterStrike2Integration`.

---

## Key Design Decisions

### 1. No TrainingEventLifecycle State Machine

Current `TrainingEventLifecycle` has FirstSeen/LastSeen/exclusion tracking — unnecessary complexity. Replace with:

```
if (!CooldownTracker.CanDetect(classId, definition.CooldownMs))
    return;
CooldownTracker.Record(classId);
AddBookmark(definition);
```

Matches how CS2 checks `if (currentKills > _lastValidKills)` — no state beyond "last time I saw this."

Exclusion types are handled by filtering before bookmark creation (if an exclusion event is detected in the same frame, skip triggers for that frame). No cross-frame state needed.

### 2. Bookmark Created Immediately

Like all game integrations, create the bookmark the moment the event is detected. Do not wait for the event to end (the UI element disappearing from screen).

### 3. Global Padding Only

Training event bookmarks use the same global `HighlightPaddingBefore`/`HighlightPaddingAfter` as Kill/Goal bookmarks. Per-event pre/post times are a training-only concept.

### 4. NCHW Tensor Layout

The ONNX model was exported with channel-first layout. Tensor data must be written as:
```
for pixel i:
  inputTensor[i] = val                // channel 0
  inputTensor[i + pixels] = val        // channel 1
  inputTensor[i + 2 * pixels] = val    // channel 2
```

Not interleaved. This was the root cause of the model producing garbage (29% instead of 96%).

### 5. Game-Specific Models are Just Files

A game gets ML detection if `data/{gameId}/model.onnx` and `data/{gameId}/events.json` exist. The detector loads them dynamically — no new C# class needed per game. Adding a new game's model is a data change, not a code change.

---

## Cleanup Opportunities

1. **Rename files** — `Backend/Training/` → `Backend/Detection/`, rename all classes
2. **Remove TrainingEventLifecycle** — replace with `CooldownTracker` (~30 lines)
3. **Add `BookmarkType.MLDetected`** — new enum value with `[IncludeInHighlight]`
4. **Frontend rename** — `training.tsx` → `detection.tsx`, or split into a dev-only dev-tools page
5. **Remove preTimeMs/postTimeMs/cooldownMs** from EventDefinition — already done
6. **Remove classId** from EventDefinition — class is just the sorted index of the definition list
7. **Feature flag** — `ENABLE_TRAINING_EVENTS` → `ENABLE_ML_DETECTION`

---

## Future Expansion

- **Per-game models**: Drop `model.onnx` + `events.json` in `data/{gameId}/` — the detector auto-discovers it
- **Multiple models per game**: Extend `EventDefinition` with a `modelId` field; load multiple ONNX sessions
- **Different model architectures**: As long as the input is 640×640 float32 NCHW and output is `[4+N, 8400]`, any YOLO variant works
- **Different input sizes**: Parameterize `ModelInputSize` per model
- **Confidence thresholds per event**: Add `minConfidence` field to `EventDefinition`
