#if ENABLE_ML_DETECTION

using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Serilog;

namespace Segra.Backend.Detection;

public static class ModelService
{
    public static readonly string BasePath = Path.Combine(AppContext.BaseDirectory, "data", "training");

    private static readonly ConcurrentDictionary<string, InferenceSession> _models = new();
    private static readonly ConcurrentDictionary<string, List<EventDefinition>> _definitions = new();

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public static List<EventDefinition> LoadEventDefinitions(string gameId)
    {
        if (_definitions.TryGetValue(gameId, out var cached))
            return cached;

        var path = Path.Combine(GetGamePath(gameId), "events.json");

        if (!File.Exists(path))
        {
            var empty = new List<EventDefinition>();
            _definitions[gameId] = empty;
            return empty;
        }

        var json = File.ReadAllText(path);
        var definitions = JsonSerializer.Deserialize<List<EventDefinition>>(json, _jsonOptions) ?? new();
        _definitions[gameId] = definitions;
        Log.Information("Loaded {Count} event definitions for game {GameId}", definitions.Count, gameId);
        return definitions;
    }

    public static void SaveEventDefinitions(string gameId, List<EventDefinition> definitions)
    {
        var path = Path.Combine(GetGamePath(gameId), "events.json");
        var json = JsonSerializer.Serialize(definitions, _jsonOptions);
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, json);
        _definitions[gameId] = definitions;
        Log.Information("Saved {Count} event definitions for game {GameId}", definitions.Count, gameId);
    }

    public static bool HasModelForGame(string gameId)
    {
        return File.Exists(GetModelPath(gameId));
    }

    public static InferenceSession LoadModel(string gameId)
    {
        if (_models.TryGetValue(gameId, out var existing))
            return existing;

        var modelPath = GetModelPath(gameId);

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found for game {gameId}", modelPath);

        var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        var session = new InferenceSession(modelPath, options);
        _models[gameId] = session;

        Log.Information("Loaded ONNX model for game {GameId}", gameId);
        return session;
    }

    public static void UnloadModel(string gameId)
    {
        if (_models.TryRemove(gameId, out var session))
        {
            session.Dispose();
            Log.Information("Unloaded ONNX model for game {GameId}", gameId);
        }

        _definitions.TryRemove(gameId, out _);
    }

    public static string GetGamePath(string gameId)
    {
        return Path.Combine(BasePath, gameId);
    }

    public static string GetModelPath(string gameId)
    {
        return Path.Combine(GetGamePath(gameId), "model.onnx");
    }
}

#endif
