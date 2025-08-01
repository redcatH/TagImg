using System.Drawing;
using System.Drawing.Imaging;
using System.Net.Mime;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TagImg;

public class JoyPredictor
{
    private InferenceSession _session;
    private List<string> _tags;
    private const int ImageSize = 448;
    private const float Threshold = 0.4f;
    private const string MODEL_FILENAME = "model.onnx";
    private const string LABEL_FILENAME = "top_tags.txt";
    private const string MODEL_REPO = "fancyfeast/joytag";

    public event EventHandler<string> LogUpdated;
    public bool IsGpuLoaded { get; private set; }

    private void AddLogEntry(string message)
    {
        string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
        LogUpdated?.Invoke(this, $"JoyPredictor: {logMessage}");
    }

    private bool _isModelLoaded = false;

    public async Task LoadModel(string modelRepo, bool useGpu = true)
    {
        AddLogEntry($"从存储库加载模型: {modelRepo}");
        var (tagsPath, modelPath) = await DownloadModel(modelRepo);

        int retryCount = 0;
        const int maxRetries = 3;

        while (retryCount < maxRetries)
        {
            try
            {
                var sessionOptions = new SessionOptions();
                var gpuDeviceId = 0;

                if (useGpu)
                {
                    AddLogEntry("初始化 ONNX 推理会话（尝试使用 GPU）");

                    try
                    {
                        sessionOptions.AppendExecutionProvider_CUDA(gpuDeviceId);
                        AddLogEntry("使用 GPU");
                        IsGpuLoaded = true;
                    }
                    catch (Exception ex)
                    {
                        AddLogEntry($"GPU初始化失败： {ex.Message}");
                        AddLogEntry("使用 CPU");
                        IsGpuLoaded = false;
                    }
                }
                else
                {
                    AddLogEntry("初始化 ONNX 推理会话（CPU 使用率）");
                    IsGpuLoaded = false;
                }

                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _session = new InferenceSession(modelPath, sessionOptions);
                _tags = File.ReadAllLines(tagsPath).Where(line => !string.IsNullOrWhiteSpace(line)).ToList();
                AddLogEntry("模型已成功加载。");
                _isModelLoaded = true;
                AddLogEntry("模型已完成加载。");
                break; // 成功した場合、ループを抜ける
            }
            catch (Exception ex)
            {
                AddLogEntry($"无法初始化 ONNX 推理会话： {ex.Message}");
                retryCount++;

                if (retryCount >= maxRetries)
                {
                    AddLogEntry("已达到最大重试次数。模型加载失败。");
                    throw;
                }

                AddLogEntry($"尝试重新下载模型。: {retryCount}");
                File.Delete(modelPath);
                AddLogEntry($"删除了现有的模型文件: {modelPath}");
                (tagsPath, modelPath) = await DownloadModel(modelRepo);
            }
        }
    }

    private async Task<(string, string)> DownloadModel(string modelRepo)
    {
        using var client = new HttpClient();
        var modelDir = Path.Combine(Path.GetTempPath(), "tagmane", modelRepo.Split('/').Last());
        Directory.CreateDirectory(modelDir);
        var tagsPath = Path.Combine(modelDir, LABEL_FILENAME);
        var modelPath = Path.Combine(modelDir, MODEL_FILENAME);
        AddLogEntry($"开始下载模型文件: {modelRepo}");
        AddLogEntry($"词典下载路径: {tagsPath}");
        AddLogEntry($"模型下载路径: {modelPath}");

        bool needsDownload = !File.Exists(tagsPath) || !File.Exists(modelPath) || !VerifyFileIntegrity(modelPath);

        if (needsDownload)
        {
            await DownloadFileWithRetry(client, $"https://huggingface.co/{modelRepo}/resolve/main/{LABEL_FILENAME}",
                tagsPath);
            // var progress = new Progress<double>(p => MediaTypeNames.Application.Current.Dispatcher.Invoke(() => ((MainWindow)MediaTypeNames.Application.Current.MainWindow).ProgressBar.Value = p * 100));
            await DownloadFileWithRetry(client, $"https://huggingface.co/{modelRepo}/resolve/main/{MODEL_FILENAME}",
                modelPath);

            if (!VerifyFileIntegrity(modelPath))
            {
                throw new Exception("下载的模型文件已损坏或不完整。");
            }

            AddLogEntry("我已经重新下载了模型文件。");
        }
        else
        {
            AddLogEntry("使用现有的模型文件。");
            // MediaTypeNames.Application.Current.Dispatcher.Invoke(() => ((MainWindow)MediaTypeNames.Application.Current.MainWindow).ProgressBar.Value = 100);
        }

        // MediaTypeNames.Application.Current.Dispatcher.Invoke(() => ((MainWindow)MediaTypeNames.Application.Current.MainWindow).ProgressBar.Value = 0);

        return (tagsPath, modelPath);
    }

    private async Task DownloadFileWithRetry(HttpClient client, string url, string filePath, int maxRetries = 3,
        IProgress<double> progress = null)
    {
        for (int i = 0; i < maxRetries; i++)
        {
            try
            {
                AddLogEntry($"下载文件：{url}");
                using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();
                long? totalBytes = response.Content.Headers.ContentLength;
                using var contentStream = await response.Content.ReadAsStreamAsync();
                using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192,
                    true);

                var totalRead = 0L;
                var buffer = new byte[8192];
                var isMoreToRead = true;

                do
                {
                    var read = await contentStream.ReadAsync(buffer, 0, buffer.Length);
                    if (read == 0)
                    {
                        isMoreToRead = false;
                    }
                    else
                    {
                        await fileStream.WriteAsync(buffer, 0, read);

                        totalRead += read;
                        if (totalBytes.HasValue)
                        {
                            var progressPercentage = (double)totalRead / totalBytes.Value;
                            progress?.Report(progressPercentage);
                        }
                    }
                } while (isMoreToRead);

                AddLogEntry($"文件下载完成: {filePath}");
                return;
            }
            catch (Exception ex)
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                    AddLogEntry($"下载失败，所以我删除了不完整的文件： {filePath}");
                }

                if (i == maxRetries - 1)
                    throw new Exception($"{maxRetries}尝试下载文件后失败: {ex.Message}");
                AddLogEntry($"下载尝试 {{i + 1}} 失败。正在重试...");
            }

            await Task.Delay(1000 * (i + 1));
        }
    }

    private bool VerifyFileIntegrity(string filePath)
    {
        try
        {
            using var stream = File.OpenRead(filePath);
            using var sha256 = SHA256.Create();
            byte[] hash = sha256.ComputeHash(stream);
            AddLogEntry($"文件完整性检查成功: {filePath}");
            return true;
        }
        catch (Exception ex)
        {
            AddLogEntry($"文件完整性检查失败: {ex.Message}");
            return false;
        }
    }

    public DenseTensor<float>? PrepareTensor(Bitmap image)
    {
        if (!_isModelLoaded)
        {
            AddLogEntry("模型尚未加载。请在执行Predict之前调用LoadModel。");
            throw new InvalidOperationException("模型尚未加载。请在执行Predict之前调用LoadModel。");
        }

        DenseTensor<float> inputTensor;
        try
        {
            inputTensor = InnerPrepareTensor(image);
        }
        catch (Exception ex)
        {
            AddLogEntry($"在准备图像时发生错误：{ex.Message}");
            return null;
        }

        return inputTensor;
    }

    public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
        DenseTensor<float> inputTensor,
        float generalThresh)
    {
        AddLogEntry("开始推理");
        AddLogEntry($"generalThresh: {generalThresh}");

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        using (var results = _session.Run(inputs))
        {
            var output = results.First().AsTensor<float>();
            var scores = new Dictionary<string, float>();

            for (int i = 0; i < _tags.Count; i++)
            {
                scores[_tags[i]] = Sigmoid(output[0, i]);

                // scores[System.Text.RegularExpressions.Regex.Replace(_tags[i], @"(?<=\w)_(?=\w)", " ")] = Sigmoid(output[0, i]);
            }

            var filteredScores = scores.Where(kv => kv.Value >= generalThresh)
                .OrderByDescending(kv => kv.Value)
                .ToDictionary(kv => kv.Key, kv => kv.Value);

            var sortedGeneralStrings = string.Join(", ", filteredScores.Keys);

            return (sortedGeneralStrings, new Dictionary<string, float>(), new Dictionary<string, float>(),
                new Dictionary<string, float>());
        }
    }

    private DenseTensor<float> InnerPrepareTensor(Bitmap image)
    {
        AddLogEntry("正在准备画像");

        var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });

        // 获取 Bitmap 的像素数据
        BitmapData bitmapData = image.LockBits(
            new Rectangle(0, 0, image.Width, image.Height),
            ImageLockMode.ReadOnly,
            PixelFormat.Format32bppArgb); // 假设使用 32 位 ARGB 格式

        int stride = bitmapData.Stride;
        byte[] pixels = new byte[stride * image.Height];
        System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, pixels, 0, pixels.Length);
        image.UnlockBits(bitmapData);

        int sourceWidth = image.Width;
        int sourceHeight = image.Height;

        const float rMean = 0.48145466f, gMean = 0.4578275f, bMean = 0.40821073f;
        const float rStd = 0.26862954f, gStd = 0.26130258f, bStd = 0.27577711f;

        float xRatio = (float)sourceWidth / ImageSize;
        float yRatio = (float)sourceHeight / ImageSize;

        for (int y = 0; y < ImageSize; y++)
        {
            int sourceY = (int)(y * yRatio);
            for (int x = 0; x < ImageSize; x++)
            {
                int sourceX = (int)(x * xRatio);
                int sourceIndex = (sourceY * stride) + (sourceX * 4);

                // Bitmap 的像素顺序通常是 BGRA
                tensor[0, 0, y, x] = (pixels[sourceIndex + 2] / 255f - rMean) / rStd; // R
                tensor[0, 1, y, x] = (pixels[sourceIndex + 1] / 255f - gMean) / gStd; // G
                tensor[0, 2, y, x] = (pixels[sourceIndex + 0] / 255f - bMean) / bStd; // B
            }
        }

        AddLogEntry("已将图像转换为张量");
        return tensor;
    }

    private float Sigmoid(float x)
    {
        return 1f / (1f + (float)Math.Exp(-x));
    }

    public void Dispose()
    {
        _session?.Dispose();
        AddLogEntry("JoyPredictor 资源已发布");
    }
}