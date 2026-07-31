# Training Event Integration Design

## Goal

Make training event detection (ML model on OBS frames) create Bookmarks and Highlights identically to how game integrations (Dota2, CS2, etc.) do. No special-casing, no per-event padding, no lifecycle state machine — detect event → create bookmark → highlight auto-generates on recording end.

---

## Current Architecture

```
OBS frame (1920x1080 BGRA, 1fps)
  → BgraToGray (BT.601 luma, full-frame)
  → CropAndResizeGray (crop region, bilinear 640x640)
  → ONNX inference (NCHW float32 tensor)
  → ParseYoloOutput (confidence threshold 0.7)
  → TrainingEventDetector raises DetectionsAvailable
  → GameIntegrationService.ProcessDetections
  → TrainingEventLifecycle.ProcessDetections
      → CreateBookmark()
  → recording.AddBookmark(bookmark)
  → [recording ends] → HighlightService checks IncludeInHighlight
  → FFmpeg extract padding → highlight video
```

---

## Key Differences vs Game Integrations

### 1. Detection Interval vs Event-Driven

| Game Integrations | Training Events |
|---|---|
| Log file watcher / state poller detects events immediately when they happen | Polls OBS at 1fps; event may appear/disappear between polls |
| One bookmark per kill/death/assist | Multiple consecutive frames detect the same UI element — must deduplicate |

### 2. Bookmark Timing

| Game Integrations | Training Events |
|---|---|
| `Time = DateTime.Now - recording.StartTime` (immediate) | `Time = FirstSeen - recording.StartTime` (first frame the event was seen) |
| Accurate to within ~100ms of the actual event | Up to 1s delayed (1fps polling) — acceptable for ~2-3s UI elements |

### 3. State Machine (TrainingEventLifecycle)

Game integrations don't need a lifecycle — each detection is an atomic event (one kill = one bookmark). Training events need tracking because the same UI element persists across frames:

- **Trigger start**: First frame detects event → create bookmark
- **Trigger extend**: Subsequent frames still detect it → update `LastSeen`, no new bookmark
- **Trigger end**: Event no longer visible → finalize (bookmark already created at start)
- **Exclusion**: Special event type that suppresses triggers while visible (e.g. "Death Spectating" overrides "Elimination")

---

## Implementation That Matches Game Integrations

### Model Loading (`TrainingEventService.LoadModel`)

- Load ONNX session once, cache by gameId
- SessionOptions with graph optimization (default `ORT_ENABLE_ALL`)
- Unload + reload on model file replacement

### Detection Loop (`TrainingEventDetector`)

```
Every 1s:
  1. Dequeue latest OBS frame (1920x1080 BGRA)
  2. BgraToGray (full-frame BT.601)
  3. For each region group:
     a. Crop rect from grayscale
     b. Bilinear resize to 640x640
     c. Build float32 NCHW tensor (channel-first!)
     d. session.Run() with RunOptions
     e. ParseYoloOutput (>0.7 threshold)
  4. Invoke DetectionsAvailable(allResults)
```

**Critical detail**: Tensor must be NCHW (not NHWC). ONNX model was trained with channel-first layout. Writing `inputTensor[i] = val; inputTensor[i+pixels] = val; inputTensor[i+2*pixels] = val` places all channel-0 pixels first, then channel-1, then channel-2.

### Bookmark Creation (replaces TrainingEventLifecycle)

The lifecycle should be replaced with a pattern matching game integrations:

```
On DetectionsAvailable:
  for each detection where type != Exclusion:
    if classId hasn't been seen within cooldown period:
      recording.AddBookmark(new Bookmark {
        Type = BookmarkType.TrainingEvent,
        TrainingEventName = definition.Name,
        Time = DateTime.Now - recording.StartTime
      })
```

This removes the FirstSeen/LastSeen/bookmark-on-end logic entirely. Each detection within a cooldown window creates one bookmark — same as a kill log line.

### Highlight Generation (no changes needed)

`BookmarkType.TrainingEvent` has `[IncludeInHighlight]`. When a recording ends:

1. `HighlightService` filters bookmarks to `IncludeInHighlight()` types (includes TrainingEvent)
2. For each bookmark: segment = `[Time - HighlightPaddingBefore, Time + HighlightPaddingAfter]`
3. Overlapping segments merged
4. FFmpeg stream-copy extracts highlight

This already works — no special-casing needed. Per-event pre/post times are not used, matching how Kill/Goal bookmarks all use the global 4s padding.

---

## Remaining Work

1. Replace `TrainingEventLifecycle.ProcessDetections` with simple cooldown-based dedup (like `_lastDetectionTime` per class)
2. Remove exclusion logic (or keep as detection-only filter with no state machine)
3. Verify bookmark creation during recording (not after recording ends)
4. Test that highlights auto-generate on recording end for recordings with training event bookmarks

The backend types (`TrainingEventDefinition`, `TrainingEventDetectionResult`) and frontend UI (event editor, sample list, training controls) stay as-is — only the lifecycle/bookmark creation path changes.
