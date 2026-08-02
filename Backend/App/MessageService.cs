using Serilog;
using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Segra.Backend.Auth;
using Segra.Backend.Core;
using System.Diagnostics;
using Segra.Backend.Games;
using Segra.Backend.Media;
using Segra.Backend.Shared;
using Segra.Backend.Platform;
using System.Net.WebSockets;
using Segra.Backend.Recorder;
using Segra.Backend.Core.Models;
using Segra.Backend.Windows.Storage;
#if ENABLE_TRAINING
using Segra.Backend.Detection;
using Segra.Backend.Training;
#endif

namespace Segra.Backend.App
{
    public static class MessageService
    {
        private static WebSocket? activeWebSocket;
        private static readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);
        private static readonly JsonSerializerOptions jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public static async Task HandleMessage(string message)
        {
            if (string.IsNullOrEmpty(message))
            {
                Log.Information("Received empty message.");
                return;
            }

            // Handle heartbeat ping (plain string, not JSON)
            if (message == "ping")
            {
                await SendFrontendMessage("pong", new { });
                return;
            }

            Log.Information("Websocket message received: " + GeneralUtils.RedactSensitiveInfo(message));

            try
            {
                var jsonDoc = JsonDocument.Parse(message);
                var root = jsonDoc.RootElement;

                if (root.TryGetProperty("Method", out JsonElement methodElement))
                {
                    string? method = methodElement.GetString();

                    if (method == null)
                    {
                        Log.Warning("Received message with null method.");
                        return;
                    }

                    switch (method)
                    {
                        case "ToggleFullscreen":
                            if (root.TryGetProperty("Parameters", out var fsParams) &&
                                fsParams.TryGetProperty("enabled", out var enabledEl))
                            {
                                bool enabled = enabledEl.GetBoolean();
                                try
                                {
                                    Program.SetFullscreen(enabled);
                                }
                                catch (Exception ex)
                                {
                                    Log.Error(ex, "Failed to toggle fullscreen");
                                }
                            }
                            break;
                        case "Login":
                            root.TryGetProperty("Parameters", out JsonElement loginParameterElement);
                            string accessToken = loginParameterElement.GetProperty("accessToken").GetString()!;
                            string refreshToken = loginParameterElement.GetProperty("refreshToken").GetString()!;
                            _ = Task.Run(() => AuthService.Login(accessToken, refreshToken));
                            break;
                        case "Logout":
                            _ = Task.Run(AuthService.Logout);
                            break;
                        case "LoginWithDiscord":
                            _ = Task.Run(DiscordLoginService.Begin);
                            break;
                        case "CancelDiscordLogin":
                            DiscordLoginService.Cancel();
                            break;
                        case "CancelClip":
                            if (root.TryGetProperty("Parameters", out var cancelClipParams) &&
                                cancelClipParams.TryGetProperty("id", out var clipId))
                            {
                                ClipService.CancelClip(clipId.GetInt32());
                            }
                            break;
                        case "CreateClip":
                            root.TryGetProperty("Parameters", out JsonElement clipParameterElement);
                            _ = Task.Run(() => HandleCreateClip(clipParameterElement));
                            break;
                        case "CreateAiClip":
                            root.TryGetProperty("Parameters", out JsonElement aiClipParameterElement);
                            _ = Task.Run(() => HandleCreateAiClip(aiClipParameterElement));
                            break;
                        case "CompressVideo":
                            root.TryGetProperty("Parameters", out JsonElement compressParameterElement);
                            _ = Task.Run(() => HandleCompressVideo(compressParameterElement));
                            break;
                        case "ApplyUpdate":
                            UpdateService.ApplyUpdate();
                            break;
                        case "CheckForUpdates":
                            Log.Information("CheckForUpdates command received.");
                            _ = Task.Run(() => UpdateService.UpdateAppIfNecessary(forceCheck: true));
                            break;
                        case "DeleteContent":
                            root.TryGetProperty("Parameters", out JsonElement deleteContentParameterElement);
                            _ = Task.Run(() => HandleDeleteContent(deleteContentParameterElement));
                            break;
                        case "DeleteMultipleContent":
                            root.TryGetProperty("Parameters", out JsonElement deleteMultipleContentParameterElement);
                            _ = Task.Run(() => HandleDeleteMultipleContent(deleteMultipleContentParameterElement));
                            break;
                        case "UploadContent":
                            root.TryGetProperty("Parameters", out JsonElement uploadContentParameterElement);
                            _ = Task.Run(() => UploadService.HandleUploadContent(uploadContentParameterElement));
                            break;
                        case "CancelUpload":
                            if (root.TryGetProperty("Parameters", out var cancelUploadParams) &&
                                cancelUploadParams.TryGetProperty("fileName", out var uploadFileName))
                            {
                                UploadService.CancelUpload(uploadFileName.GetString()!);
                            }
                            break;
                        case "OpenFileLocation":
                            if (root.TryGetProperty("Parameters", out JsonElement openFileLocationParameterElement) &&
                                openFileLocationParameterElement.TryGetProperty("FilePath", out JsonElement filePathElement) &&
                                filePathElement.ValueKind == JsonValueKind.String)
                            {
                                PlatformServices.Dialogs.OpenFileLocation(filePathElement.GetString()!);
                            }
                            else
                            {
                                Log.Warning("FilePath parameter not found in OpenFileLocation message");
                            }
                            break;
                        case "CopyFileToClipboard":
                            root.TryGetProperty("Parameters", out JsonElement copyFileParams);
                            if (copyFileParams.TryGetProperty("FilePath", out JsonElement copyFilePath))
                            {
                                string clipboardFilePath = copyFilePath.GetString()!;
                                if (File.Exists(clipboardFilePath))
                                {
                                    PlatformServices.Dialogs.CopyFileToClipboard(clipboardFilePath);
                                }
                                else
                                {
                                    Log.Warning($"File not found for clipboard copy: {clipboardFilePath}");
                                }
                            }
                            break;
                        case "OpenInBrowser":
                            root.TryGetProperty("Parameters", out JsonElement openInBrowserParameterElement);
                            if (openInBrowserParameterElement.TryGetProperty("Url", out JsonElement urlElement))
                            {
                                string url = urlElement.GetString()!;
                                Log.Information($"Opening URL in browser: {url}");
                                PlatformServices.Dialogs.OpenUrl(url);
                            }
                            else
                            {
                                Log.Error("URL parameter not found in OpenInBrowser message");
                            }
                            break;
                        case "OpenLogsLocation":
                            _ = Task.Run(() => DiagnosticsService.LogSnapshot());
                            string logDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Segra");
                            string? logFilePath = Directory.GetFiles(logDir, "*.log").FirstOrDefault();
                            if (!string.IsNullOrEmpty(logFilePath))
                            {
                                PlatformServices.Dialogs.OpenFileLocation(logFilePath);
                            }
                            else
                            {
                                Log.Warning("No log files found in the Segra directory");
                            }
                            break;
                        case "SelectGameExecutable":
                            await HandleSelectGameExecutable();
                            break;
                        case "StartRecording":
                            if (AppState.Instance.Recording != null || AppState.Instance.PreRecording != null)
                            {
                                Log.Information("Recording already in progress. Skipping...");
                                return;
                            }

                            _ = Task.Run(() => OBSService.StartRecording(startManually: true));
                            break;
                        case "StopRecording":
                            _ = Task.Run(OBSService.StopRecording);
                            break;
                        case "RefreshStorageStats":
                            StorageService.UpdateRecordingDriveSpaceInState();
                            break;
                        case "NewConnection":
                            Log.Information("NewConnection command received.");
                            await SendSettingsToFrontend("New connection");
                            await SendStateToFrontend("New connection");

                            await SendGameList();

                            // canSelfUpdate: false on Linux/Flatpak, where the package manager owns updates.
                            // Informational version, not GetName().Version: it keeps the -beta.N suffix,
                            // which the frontend's What's New check needs on Flatpak (no Velopack metadata).
                            string appVersion = UpdateService.UpdateManager.CurrentVersion?.ToString()
                                ?? Assembly.GetExecutingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
                                ?? "0.0.0";

                            await SendFrontendMessage("AppVersion", new
                            {
                                version = appVersion,
#if WINDOWS
                                canSelfUpdate = true,
#else
                                canSelfUpdate = false,
#endif
                            });

                            await UpdateService.SendCurrentUpdateProgressToFrontend();
                            _ = Task.Run(() => UpdateService.GetReleaseNotes());
                            break;
                        case "SetVideoLocation":
                            await SetVideoLocationAsync();
                            Log.Information("SetVideoLocation command received.");
                            break;
                        case "SetCacheLocation":
                            await SetCacheLocationAsync();
                            Log.Information("SetCacheLocation command received.");
                            break;
                        case "UpdateSettings":
                            root.TryGetProperty("Parameters", out JsonElement settingsParameterElement);
                            Log.Information("UpdateSettings command received.");
                            await SettingsService.HandleUpdateSettings(settingsParameterElement);
                            break;
                        case "AddBookmark":
                            root.TryGetProperty("Parameters", out JsonElement bookmarkParameterElement);
                            await ContentService.HandleAddBookmark(bookmarkParameterElement);
                            Log.Information("AddBookmark command received.");
                            break;
                        case "DeleteBookmark":
                            root.TryGetProperty("Parameters", out JsonElement deleteBookmarkParameterElement);
                            await ContentService.HandleDeleteBookmark(deleteBookmarkParameterElement);
                            Log.Information("DeleteBookmark command received.");
                            break;
                        case "RenameContent":
                            root.TryGetProperty("Parameters", out JsonElement renameContentParameterElement);
                            await ContentService.HandleRenameContent(renameContentParameterElement);
                            Log.Information("RenameContent command received.");
                            break;
                        case "ImportFile":
                            root.TryGetProperty("Parameters", out JsonElement importParameterElement);
                            _ = Task.Run(() => ImportService.HandleImportFile(importParameterElement));
                            Log.Information("ImportFile command received.");
                            break;
                        case "MigrateContent":
                            _ = Task.Run(ContentMigrationService.HandleMigrateContent);
                            Log.Information("MigrateContent command received.");
                            break;
                        case "StorageWarningConfirm":
                            root.TryGetProperty("Parameters", out JsonElement storageWarningParameterElement);
                            _ = Task.Run(() => StorageWarningService.HandleStorageWarningConfirm(storageWarningParameterElement));
                            Log.Information("StorageWarningConfirm command received.");
                            break;
                        case "RecoveryConfirm":
                            root.TryGetProperty("Parameters", out JsonElement recoveryParameterElement);
                            _ = Task.Run(() => RecoveryService.HandleRecoveryConfirm(recoveryParameterElement));
                            Log.Information("RecoveryConfirm command received.");
                            break;
                        case "ApplyVideoPreset":
                            if (root.TryGetProperty("Parameters", out var videoPresetParams) &&
                                videoPresetParams.TryGetProperty("preset", out var videoPresetEl))
                            {
                                string? videoPreset = videoPresetEl.GetString();
                                if (!string.IsNullOrEmpty(videoPreset))
                                {
                                    await PresetsService.ApplyVideoPreset(videoPreset);
                                }
                            }
                            break;
                        case "ApplyClipPreset":
                            if (root.TryGetProperty("Parameters", out var clipPresetParams) &&
                                clipPresetParams.TryGetProperty("preset", out var clipPresetEl))
                            {
                                string? clipPreset = clipPresetEl.GetString();
                                if (!string.IsNullOrEmpty(clipPreset))
                                {
                                    await PresetsService.ApplyClipPreset(clipPreset);
                                }
                            }
                            break;
#if ENABLE_TRAINING
                        case "GetTrainingEvents":
                            root.TryGetProperty("Parameters", out var getEventsParams);
                            HandleGetTrainingEvents(getEventsParams);
                            break;
                        case "SaveTrainingEvent":
                            root.TryGetProperty("Parameters", out var saveEventParams);
                            HandleSaveTrainingEvent(saveEventParams);
                            break;
                        case "DeleteTrainingEvent":
                            root.TryGetProperty("Parameters", out var deleteEventParams);
                            HandleDeleteTrainingEvent(deleteEventParams);
                            break;
                        case "GetTrainingModelStatus":
                            root.TryGetProperty("Parameters", out var modelStatusParams);
                            HandleGetTrainingModelStatus(modelStatusParams);
                            break;
                        case "LoadTrainingModel":
                            root.TryGetProperty("Parameters", out var loadModelParams);
                            HandleLoadTrainingModel(loadModelParams);
                            break;
                        case "AddTrainingSample":
                            root.TryGetProperty("Parameters", out var addSampleParams);
                            HandleAddTrainingSample(addSampleParams);
                            break;
                        case "ExportTrainingDataset":
                            root.TryGetProperty("Parameters", out var exportParams);
                            HandleExportTrainingDataset(exportParams);
                            break;
                        case "TrainModel":
                            root.TryGetProperty("Parameters", out var trainModelParams);
                            HandleTrainModel(trainModelParams);
                            break;
                        case "DeleteTrainingSample":
                            root.TryGetProperty("Parameters", out var deleteSampleParams);
                            HandleDeleteTrainingSample(deleteSampleParams);
                            break;
                        case "GetTrainingSampleImage":
                            root.TryGetProperty("Parameters", out var sampleImageParams);
                            HandleGetTrainingSampleImage(sampleImageParams);
                            break;
                        case "UpdateTrainingSampleLabels":
                            root.TryGetProperty("Parameters", out var updateLabelsParams);
                            HandleUpdateTrainingSampleLabels(updateLabelsParams);
                            break;
#endif
                        default:
                            Log.Information($"Unknown method: {method}");
                            break;
                    }
                }
                else
                {
                    Log.Information("Method property not found in message.");
                }
            }
            catch (JsonException ex)
            {
                Log.Error($"Failed to parse message as JSON: {ex.Message}");
            }
            catch (Exception ex)
            {
                Log.Error($"Unhandled exception in message handler: {ex.Message}");
                Log.Error($"Stack trace: {ex.StackTrace}");
            }
        }

        public static async Task HandleDeleteContent(JsonElement message)
        {
            Log.Information($"Handling DeleteContent with message: {message}");

            if (message.TryGetProperty("FileName", out JsonElement fileNameElement) &&
                message.TryGetProperty("ContentType", out JsonElement contentTypeElement))
            {
                string fileName = fileNameElement.GetString()!;
                string contentTypeStr = contentTypeElement.GetString()!;

                if (Enum.TryParse(contentTypeStr, true, out Content.ContentType contentType))
                {
                    Content? content = AppState.Instance.Content.FirstOrDefault(c =>
                        c.FileName == fileName && c.Type == contentType);

                    if (content != null && !string.IsNullOrEmpty(content.FilePath))
                    {
                        await ContentService.DeleteContent(content.FilePath, contentType);
                    }
                    else
                    {
                        Log.Warning($"Content not found in state for deletion: {fileName} ({contentTypeStr})");
                    }
                }
                else
                {
                    Log.Error($"Invalid ContentType provided: {contentTypeStr}");
                }
            }
            else
            {
                Log.Information("FileName or ContentType property not found in DeleteContent message.");
            }
        }

        public static async Task HandleDeleteMultipleContent(JsonElement message)
        {
            Log.Information($"Handling DeleteMultipleContent with message: {message}");

            if (!message.TryGetProperty("Items", out JsonElement itemsElement))
            {
                Log.Information("Items property not found in DeleteMultipleContent message.");
                return;
            }

            // Use bulk update to prevent multiple frontend updates
            Settings.Instance._isBulkUpdating = true;
            try
            {
                foreach (var item in itemsElement.EnumerateArray())
                {
                    if (item.TryGetProperty("FileName", out JsonElement fileNameElement) &&
                        item.TryGetProperty("ContentType", out JsonElement contentTypeElement))
                    {
                        string fileName = fileNameElement.GetString()!;
                        string contentTypeStr = contentTypeElement.GetString()!;

                        if (Enum.TryParse(contentTypeStr, true, out Content.ContentType contentType))
                        {
                            Content? content = AppState.Instance.Content.FirstOrDefault(c =>
                                c.FileName == fileName && c.Type == contentType);

                            if (content != null && !string.IsNullOrEmpty(content.FilePath))
                            {
                                await ContentService.DeleteContent(content.FilePath, contentType, sendToFrontend: false);
                                Log.Information($"Deleted content: {fileName}");
                            }
                            else
                            {
                                Log.Warning($"Content not found in state for deletion: {fileName} ({contentTypeStr})");
                            }
                        }
                        else
                        {
                            Log.Error($"Invalid ContentType provided: {contentTypeStr}");
                        }
                    }
                }
            }
            finally
            {
                Settings.Instance._isBulkUpdating = false;
                // Reload content and send single update to frontend
                await SettingsService.LoadContentFromFolderIntoState(true);
            }
        }

        public static async Task StartWebsocket()
        {
            HttpListener listener = new HttpListener();
            listener.Prefixes.Add("http://localhost:44030/");
            listener.Start();
            Log.Information("WebSocket server started at ws://localhost:44030/");

            try
            {
                while (true)
                {
                    HttpListenerContext context = await listener.GetContextAsync();

                    if (context.Request.IsWebSocketRequest)
                    {
                        Log.Information("Received WebSocket connection request");

                        if (activeWebSocket != null && activeWebSocket.State == WebSocketState.Open)
                        {
                            await activeWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "New connection", CancellationToken.None);
                            Log.Information("Closed previous WebSocket connection.");
                        }

                        HttpListenerWebSocketContext wsContext = await context.AcceptWebSocketAsync(null);
                        activeWebSocket = wsContext.WebSocket;

                        Log.Information("WebSocket connection established");
                        await HandleWebSocketAsync(activeWebSocket);
                    }
                    else
                    {
                        Log.Information("Invalid request: Not a WebSocket request");
                        context.Response.StatusCode = 400;
                        context.Response.Close();
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Information($"Exception in StartWebsocket: {ex.Message}");
                if (ex.StackTrace != null)
                {
                    Log.Information(ex.StackTrace);
                }
            }
        }

        public static async Task SendFrontendMessage(string method, object content)
        {
            await sendLock.WaitAsync();
            try
            {
                // Wait for up to 10 seconds for the websocket to be open
                int maxWaitTimeMs = 10000;
                int waitIntervalMs = 100;
                int elapsedTime = 0;

                while ((activeWebSocket == null || activeWebSocket.State != WebSocketState.Open)
                    && elapsedTime < maxWaitTimeMs)
                {
                    await Task.Delay(waitIntervalMs);
                    elapsedTime += waitIntervalMs;
                }

                if (activeWebSocket?.State == WebSocketState.Open)
                {
                    var message = new { method, content };
                    byte[] buffer = JsonSerializer.SerializeToUtf8Bytes(message, jsonOptions);
                    await activeWebSocket.SendAsync(
                        buffer,
                        WebSocketMessageType.Text,
                        endOfMessage: true,
                        cancellationToken: CancellationToken.None
                    );
                }
            }
            catch (Exception ex)
            {
                Log.Error($"Error sending message: {ex.Message}");
            }
            finally
            {
                sendLock.Release();
            }
        }

        public static async Task ShowModal(string title, string description, string type = "info", string? subtitle = null)
        {
            if (type != "info" && type != "warning" && type != "error")
            {
                Log.Warning($"Invalid modal type '{type}'. Defaulting to 'info'.");
                type = "info";
            }

            var modalContent = new
            {
                title,
                subtitle,
                description,
                type
            };

            await SendFrontendMessage("ShowModal", modalContent);
            Log.Information($"Sent modal to frontend: {title} ({type})");
        }

        public static async Task SendSettingsToFrontend(string cause)
        {
            if (!Program.hasLoadedInitialSettings || Settings.Instance._isBulkUpdating)
                return;

            Log.Information("Sending settings to frontend ({Cause})", cause);
            await SendFrontendMessage("Settings", Settings.Instance);
        }

        public static async Task SendStateToFrontend(string cause)
        {
            if (!Program.hasLoadedInitialSettings || Settings.Instance._isBulkUpdating)
                return;

            Log.Information("Sending state to frontend ({Cause})", cause);
            await SendFrontendMessage("State", AppState.Instance);
        }

        public static async Task SendGameList()
        {
            try
            {
                var gameList = GameUtils.GetGameList();
                await SendFrontendMessage("GameList", gameList);
                Log.Information("Sent game list to frontend ({Count} games)", gameList.Count);
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Error sending game list to frontend");
                await SendFrontendMessage("GameList", new List<object>());
            }
        }

        private static async Task HandleCreateAiClip(JsonElement message)
        {
            Log.Information($"{message}");
            message.TryGetProperty("FileName", out JsonElement fileNameElement);
            await AiService.CreateHighlight(fileNameElement.GetString()!);
        }

        private static async Task HandleCompressVideo(JsonElement message)
        {
            Log.Information($"CompressVideo: {message}");
            message.TryGetProperty("FilePath", out JsonElement filePathElement);
            await CompressionService.CompressVideo(filePathElement.GetString()!);
        }

        private static async Task HandleCreateClip(JsonElement message)
        {
            Log.Information($"{message}");

            if (message.TryGetProperty("Segments", out JsonElement segmentsElement))
            {
                var segments = new List<Segment>();
                foreach (var segmentElement in segmentsElement.EnumerateArray())
                {
                    if (segmentElement.TryGetProperty("id", out JsonElement idElement) &&
                        segmentElement.TryGetProperty("startTime", out JsonElement startTimeElement) &&
                        segmentElement.TryGetProperty("endTime", out JsonElement endTimeElement) &&
                        segmentElement.TryGetProperty("fileName", out JsonElement fileNameElement) &&
                        segmentElement.TryGetProperty("type", out JsonElement videoTypeElement) &&
                        segmentElement.TryGetProperty("game", out JsonElement gameElement) &&
                        segmentElement.TryGetProperty("title", out JsonElement titleElement))
                    {
                        long id = idElement.GetInt64();
                        double startTime = startTimeElement.GetDouble();
                        double endTime = endTimeElement.GetDouble();
                        string fileName = fileNameElement.GetString()!;
                        string type = videoTypeElement.GetString()!;
                        string game = gameElement.GetString()!;
                        string title = titleElement.GetString() ?? string.Empty;
                        int? igdbId = segmentElement.TryGetProperty("igdbId", out JsonElement igdbIdElement) && igdbIdElement.ValueKind == JsonValueKind.Number
                            ? igdbIdElement.GetInt32()
                            : null;
                        string? filePath = segmentElement.TryGetProperty("filePath", out JsonElement filePathElement)
                            ? filePathElement.GetString()
                            : null;
                        List<int>? mutedAudioTracks = null;
                        if (segmentElement.TryGetProperty("mutedAudioTracks", out JsonElement mutedEl)
                            && mutedEl.ValueKind == JsonValueKind.Array)
                        {
                            mutedAudioTracks = mutedEl.EnumerateArray().Select(e => e.GetInt32()).ToList();
                        }
                        Dictionary<int, double>? audioTrackVolumes = null;
                        if (segmentElement.TryGetProperty("audioTrackVolumes", out JsonElement volEl)
                            && volEl.ValueKind == JsonValueKind.Object)
                        {
                            audioTrackVolumes = new Dictionary<int, double>();
                            foreach (var prop in volEl.EnumerateObject())
                            {
                                if (int.TryParse(prop.Name, out int trackIdx) && prop.Value.TryGetDouble(out double vol))
                                    audioTrackVolumes[trackIdx] = vol;
                            }
                        }

                        segments.Add(new Segment
                        {
                            Id = id,
                            Type = type,
                            StartTime = startTime,
                            EndTime = endTime,
                            FileName = fileName,
                            FilePath = filePath,
                            Game = game,
                            Title = title,
                            IgdbId = igdbId,
                            MutedAudioTracks = mutedAudioTracks,
                            AudioTrackVolumes = audioTrackVolumes
                        });
                    }
                }

                bool createSeparateClips = message.TryGetProperty("OutputMode", out JsonElement outputModeElement)
                    && string.Equals(outputModeElement.GetString(), "separate", StringComparison.OrdinalIgnoreCase);

                await ClipService.CreateClips(segments, createSeparateClips);
            }
            else
            {
                Log.Information("Segments property not found in CreateClip message.");
            }
        }

        private static async Task SetVideoLocationAsync()
        {
            string? picked = await PlatformServices.Dialogs.PickFolderAsync("Select a folder to set as the video location.");
            if (picked != null)
            {
                string selectedPath = Shared.PathUtils.Normalize(picked);
                Log.Information($"Selected Folder: {selectedPath}");

                // Check if the new folder would exceed storage limit
                bool shouldProceed = await StorageWarningService.CheckContentFolderChange(selectedPath);
                if (shouldProceed)
                {
                    Settings.Instance.ContentFolder = selectedPath;

                    // Push the updated path to the frontend so the settings UI reflects the change
                    await SendSettingsToFrontend("Content folder changed");
                }
                // If not proceeding, a warning modal was sent to the frontend
            }
            else
            {
                Log.Information("Folder selection was canceled.");
            }
        }

        private static async Task SetCacheLocationAsync()
        {
            string? picked = await PlatformServices.Dialogs.PickFolderAsync("Select a folder for metadata, thumbnails, and waveforms.");
            if (picked != null)
            {
                string selectedPath = Shared.PathUtils.Normalize(picked);
                string oldCacheFolder = Settings.Instance.CacheFolder;
                Log.Information($"Selected Cache Folder: {selectedPath}");

                Settings.Instance.CacheFolder = selectedPath;
                SettingsService.SaveSettings();

                await SettingsService.MigrateCacheFolder(oldCacheFolder, selectedPath);

                await SendSettingsToFrontend("Cache folder changed");
            }
            else
            {
                Log.Information("Cache folder selection was canceled.");
            }
        }

        private static async Task HandleWebSocketAsync(WebSocket webSocket)
        {
            byte[] buffer = new byte[4096];
            try
            {
                while (webSocket.State == WebSocketState.Open)
                {
                    WebSocketReceiveResult result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        Log.Information("Client initiated WebSocket closure.");
                        await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client initiated closure", CancellationToken.None);
                    }
                    else
                    {
                        string receivedMessage = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        await HandleMessage(receivedMessage);
                    }
                }
            }
            catch (WebSocketException wsEx)
            {
                Log.Information($"WebSocketException in HandleWebSocketAsync: {wsEx.Message}");
                Log.Information($"WebSocket state at exception: {webSocket.State}");
                if (wsEx.InnerException != null)
                {
                    Log.Information($"Inner exception: {wsEx.InnerException.Message}");
                }
            }
            catch (Exception ex)
            {
                Log.Information($"General exception in HandleWebSocketAsync: {ex.Message}");
            }
            finally
            {
                if (webSocket.State != WebSocketState.Closed && webSocket.State != WebSocketState.Aborted)
                {
                    await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, "Server-side error", CancellationToken.None);
                }
                Log.Information("WebSocket connection closed.");
            }
        }

        private static async Task HandleSelectGameExecutable()
        {
            try
            {
                string? pickedFile = await PlatformServices.Dialogs.PickFileAsync("Select Game Executable", "Executable Files (*.exe)", "exe");

                if (pickedFile != null)
                {
                    string filePath = Shared.PathUtils.Normalize(pickedFile);
                    string fileName = Path.GetFileNameWithoutExtension(filePath);

                    // If the selected exe is a known catalog game, link it (catalog name + igdb id + CDN
                    // icon) so it behaves exactly like adding from search; otherwise treat it as a custom
                    // game and extract the exe's own icon.
                    string? catalogName = GameUtils.GetGameNameFromExePath(pickedFile);
                    int? igdbId = GameUtils.GetIgdbIdFromExePath(pickedFile);
                    string? catalogIcon = GameUtils.GetIconFromExePath(pickedFile);
                    // Fall back to the exe's own icon whenever the catalog has no icon for it (even for a
                    // known game), so the entry always has the best icon available.
                    string? customIcon = catalogIcon == null
                        ? Shared.IconUtils.ExtractExeIconBase64(pickedFile)
                        : null;

                    var gameObject = new
                    {
                        name = catalogName ?? fileName,
                        paths = new[] { filePath },
                        igdbId,
                        icon = catalogIcon,
                        customIcon
                    };

                    await SendFrontendMessage("SelectedGameExecutable", gameObject);
                    Log.Information($"Selected game executable: {filePath}{(catalogName != null ? $" (matched catalog game '{catalogName}')" : "")}");
                }
            }
            catch (Exception ex)
            {
                Log.Error($"Error selecting game executable: {ex.Message}");
                await ShowModal("Error", $"Failed to select game executable: {ex.Message}", "error");
            }
        }

#if ENABLE_TRAINING
        private static async void HandleGetTrainingEvents(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var events = TrainingEventService.LoadEventDefinitions(gameId);
                var samples = TrainingEventService.GetSamples(gameId);
                var sampleCounts = events
                    .ToDictionary(e => e.Id, e => TrainingEventService.GetSampleCountForEvent(gameId, e.Id));
                var samplesWithImages = samples
                    .OrderByDescending(s => s.Timestamp)
                    .Select(s =>
                {
                    // Collect all event IDs from the multi-line label file
                    var labelEventIds = new HashSet<int> { s.EventId };
                    string? labelText = null;
                    try
                    {
                        if (File.Exists(s.LabelPath))
                        {
                            labelText = File.ReadAllText(s.LabelPath).Trim();
                            foreach (var line in labelText.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                            {
                                var parts = line.Trim().Split(' ');
                                if (parts.Length >= 5 && int.TryParse(parts[0], out var eid))
                                    labelEventIds.Add(eid);
                            }
                        }
                    }
                    catch { }

                    var labelCount = labelText?.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length ?? 1;
                    var eventNames = labelEventIds
                        .Select(id => events.FirstOrDefault(e => e.Id == id)?.Name ?? $"Event {id}")
                        .ToList();

                    return new
                    {
                        id = Math.Abs(s.ImagePath.GetHashCode()),
                        imageData = (string?)null,
                        eventId = s.EventId,
                        eventName = eventNames.FirstOrDefault() ?? $"Event {s.EventId}",
                        eventNames,
                        labelCount
                    };
                }).ToList();
                var result = new
                {
                    events,
                    samples = samplesWithImages,
                    sampleCounts
                };
                await SendFrontendMessage("TrainingEvents", result);
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleGetTrainingEvents failed");
            }
        }

        private static async void HandleSaveTrainingEvent(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var evt = JsonSerializer.Deserialize<EventDefinition>(
                    parameters.GetProperty("event").GetRawText(), jsonOptions);
                if (evt != null)
                {
                    TrainingEventService.AddEventDefinition(gameId, evt);
                    await SendFrontendMessage("TrainingEventSaved", new { success = true });
                }
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleSaveTrainingEvent failed");
                await SendFrontendMessage("TrainingEventSaved",
                    new { success = false, error = ex.Message });
            }
        }

        private static async void HandleDeleteTrainingEvent(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var eventId = parameters.GetProperty("eventId").GetInt32();
                TrainingEventService.RemoveEventDefinition(gameId, eventId);
                await SendFrontendMessage("TrainingEventDeleted", new { success = true });
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleDeleteTrainingEvent failed");
            }
        }

#endif
#if ENABLE_TRAINING
        private static async void HandleAddTrainingSample(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var eventId = parameters.GetProperty("eventId").GetInt32();
                var imageBase64 = parameters.GetProperty("imageData").GetString() ?? "";
                var imageBytes = Convert.FromBase64String(imageBase64);

                var sample = new TrainingSample
                {
                    EventId = eventId,
                    BoxX = parameters.TryGetProperty("boxX", out var bx) ? bx.GetSingle() : 0.5f,
                    BoxY = parameters.TryGetProperty("boxY", out var by) ? by.GetSingle() : 0.5f,
                    BoxW = parameters.TryGetProperty("boxW", out var bw) ? bw.GetSingle() : 1.0f,
                    BoxH = parameters.TryGetProperty("boxH", out var bh) ? bh.GetSingle() : 1.0f,
                    Timestamp = TimeSpan.Zero
                };

                TrainingEventService.AddSample(gameId, sample, imageBytes);
                await SendFrontendMessage("TrainingSampleAdded",
                    new { success = true, sampleCount = TrainingEventService.GetSampleCount(gameId) });
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleAddTrainingSample failed");
                await SendFrontendMessage("TrainingSampleAdded",
                    new { success = false, error = ex.Message });
            }
        }

        private static async void HandleExportTrainingDataset(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                TrainingEventService.ExportDataset(gameId);
                var gamePath = TrainingEventService.GetGamePath(gameId);
                await SendFrontendMessage("TrainingDatasetExported",
                    new { success = true, datasetPath = gamePath, gameId });
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleExportTrainingDataset failed");
                await SendFrontendMessage("TrainingDatasetExported",
                    new { success = false, error = ex.Message });
            }
        }

#endif
#if ENABLE_TRAINING
        private static async void HandleGetTrainingModelStatus(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var hasModel = TrainingEventService.HasModelForGame(gameId);
                var sampleCount = TrainingEventService.GetSampleCount(gameId);
                await SendFrontendMessage("TrainingModelStatus",
                    new { hasModel, sampleCount, gameId });
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleGetTrainingModelStatus failed");
            }
        }

#endif
#if ENABLE_TRAINING
        private static async void HandleDeleteTrainingSample(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                string? pathToDelete = null;
                if (parameters.TryGetProperty("imagePath", out var imgPath))
                    pathToDelete = imgPath.GetString();
                else if (parameters.TryGetProperty("sampleId", out var sid))
                {
                    var sampleId = sid.GetInt32();
                    var samples = TrainingEventService.GetSamples(gameId);
                    var sample = samples.FirstOrDefault(s => s.EventId == sampleId);
                    if (sample != null) pathToDelete = sample.ImagePath;
                }
                if (pathToDelete != null)
                    TrainingEventService.DeleteSample(gameId, pathToDelete);
                await SendFrontendMessage("TrainingSampleDeleted", new { success = true, gameId });
                _ = Task.Run(() => HandleGetTrainingEvents(parameters));
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleDeleteTrainingSample failed");
            }
        }

        private static async void HandleGetTrainingSampleImage(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var imageHashCode = parameters.GetProperty("sampleId").GetInt32();
                var samples = TrainingEventService.GetSamples(gameId);
                var sample = samples.FirstOrDefault(s => Math.Abs(s.ImagePath.GetHashCode()) == imageHashCode);
                if (sample == null || !File.Exists(sample.ImagePath))
                {
                    await SendFrontendMessage("TrainingSampleImage", new { success = false, gameId });
                    return;
                }

                var imageBase64 = Convert.ToBase64String(File.ReadAllBytes(sample.ImagePath));
                string? labelContent = null;
                if (File.Exists(sample.LabelPath))
                    labelContent = File.ReadAllText(sample.LabelPath).Trim().Replace(',', '.');

                var events = TrainingEventService.LoadEventDefinitions(gameId);
                var ev = events.FirstOrDefault(e => e.Id == sample.EventId);

                await SendFrontendMessage("TrainingSampleImage", new
                {
                    success = true,
                    gameId,
                    sampleId = imageHashCode,
                    imageData = "data:image/png;base64," + imageBase64,
                    label = labelContent,
                    eventName = ev?.Name ?? $"Event {sample.EventId}",
                    eventId = sample.EventId
                });
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleGetTrainingSampleImage failed");
                await SendFrontendMessage("TrainingSampleImage", new { success = false, error = ex.Message });
            }
        }

        private static async void HandleUpdateTrainingSampleLabels(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var imageHashCode = parameters.GetProperty("sampleId").GetInt32();
                var labelsArray = parameters.GetProperty("labels").EnumerateArray().ToList();

                var samples = TrainingEventService.GetSamples(gameId);
                var sample = samples.FirstOrDefault(s => Math.Abs(s.ImagePath.GetHashCode()) == imageHashCode);
                if (sample == null)
                {
                    await SendFrontendMessage("TrainingSampleLabelsUpdated", new { success = false, gameId, error = "Sample not found" });
                    return;
                }

                var inv = CultureInfo.InvariantCulture;
                var lines = new List<string>();
                foreach (var lbl in labelsArray)
                {
                    var eventId = lbl.GetProperty("eventId").GetInt32();
                    var bx = lbl.GetProperty("boxX").GetSingle();
                    var by = lbl.GetProperty("boxY").GetSingle();
                    var bw = lbl.GetProperty("boxW").GetSingle();
                    var bh = lbl.GetProperty("boxH").GetSingle();
                    lines.Add($"{eventId} {bx.ToString(inv)} {by.ToString(inv)} {bw.ToString(inv)} {bh.ToString(inv)}");
                }

                File.WriteAllLines(sample.LabelPath, lines);
                await SendFrontendMessage("TrainingSampleLabelsUpdated", new { success = true, gameId });
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleUpdateTrainingSampleLabels failed");
                await SendFrontendMessage("TrainingSampleLabelsUpdated", new { success = false, error = ex.Message });
            }
        }

#endif
#if ENABLE_TRAINING
        private static async void HandleLoadTrainingModel(JsonElement parameters)
        {
            try
            {
                var gameId = parameters.GetProperty("gameId").GetString() ?? "";
                var success = TrainingEventService.HasModelForGame(gameId);
                await SendFrontendMessage("TrainingModelLoaded",
                    new { success, gameId });
                if (success)
                    Log.Information("ML model loaded for {GameId}", gameId);
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleLoadTrainingModel failed");
                await SendFrontendMessage("TrainingModelLoaded",
                    new { success = false, gameId = "", error = ex.Message });
            }
        }

        private static async void HandleTrainModel(JsonElement parameters)
        {
            var gameId = "";
            try
            {
                gameId = parameters.GetProperty("gameId").GetString() ?? "";
                await MessageService.SendFrontendMessage("TrainingProgress",
                    new { gameId, status = "starting", message = "Starting training..." });

                var gamePath = TrainingEventService.GetGamePath(gameId);
                var datasetDir = Path.Combine(gamePath, "dataset");

                await MessageService.SendFrontendMessage("TrainingProgress",
                    new { gameId, status = "starting", message = "Exporting dataset..." });
                TrainingEventService.ExportDataset(gameId);

                _ = Task.Run(() => TrainingRunner.RunTraining(gameId, datasetDir));
            }
            catch (Exception ex)
            {
                Log.Error(ex, "HandleTrainModel failed");
                await SendFrontendMessage("TrainingProgress",
                    new { gameId, status = "error", message = ex.Message });
            }
        }
#endif
    }
}
