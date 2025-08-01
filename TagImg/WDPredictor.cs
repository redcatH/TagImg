using System.Collections.ObjectModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Net.Mime;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TagImg;

    public class WDPredictor
    {
        private InferenceSession _model;
        private List<string> _tagNames;
        private List<int> _ratingIndexes;
        private List<int> _generalIndexes;
        private List<int> _characterIndexes;
        private int _modelTargetSize;
        private const int MaxLogEntries = 20;

        private const string MODEL_FILENAME = "model.onnx";
        private const string LABEL_FILENAME = "selected_tags.csv";

        public event EventHandler<string> LogUpdated;
        public bool IsGpuLoaded { get; private set; }

        private void AddLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            LogUpdated?.Invoke(this, $"WDPredictor: {logMessage}");
        }

        private bool _isModelLoaded = false;

        public async Task LoadModel(string modelRepo, bool useGpu = true)
        {
            AddLogEntry($"从存储库中加载模型: {modelRepo}");
            var (csvPath, modelPath) = await DownloadModel(modelRepo);

            _tagNames = new List<string>();
            _ratingIndexes = new List<int>();
            _generalIndexes = new List<int>();
            _characterIndexes = new List<int>();

            LoadLabels(csvPath);

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
                        AddLogEntry("正在初始化ONNX推理会话（尝试使用GPU）");

                        try
                        {
                            sessionOptions.AppendExecutionProvider_CUDA(gpuDeviceId);
                            AddLogEntry("使用GPU");
                            IsGpuLoaded = true;
                        }
                        catch (Exception ex)
                        {
                            AddLogEntry($"GPU初始化失败: {ex.Message}");
                            AddLogEntry("使用CPU");
                            IsGpuLoaded = false;
                        }
                    }
                    else
                    {
                        AddLogEntry("正在初始化ONNX推理会话（使用CPU）");
                        IsGpuLoaded = false;
                    }
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _model = new InferenceSession(modelPath, sessionOptions);
                    break;
                }
                catch (Exception ex)
                {
                    AddLogEntry($"ONNX推理会话初始化失败：{ex.Message}");
                    retryCount++;

                    if (retryCount >= maxRetries)
                    {
                        AddLogEntry("已达到最大重试次数。模型加载失败。");
                        throw;
                    }

                    AddLogEntry($"尝试重新下载模型。尝试次数： {retryCount}");
                    File.Delete(modelPath);
                    AddLogEntry($"已删除现有的模型文件: {modelPath}");
                    (_, modelPath) = await DownloadModel(modelRepo);
                }
            }
            
            AddLogEntry($"模型加载已完成。");
            _modelTargetSize = _model.InputMetadata.First().Value.Dimensions[2];
            AddLogEntry($"目标尺寸：{_modelTargetSize}");
            _isModelLoaded = true;
            AddLogEntry("初始化已完成。");
        }

        private async Task<(string, string)> DownloadModel(string modelRepo)
        {
            using var client = new HttpClient();
            var modelDir = Path.Combine(Directory.GetCurrentDirectory(), "models", modelRepo.Split('/').Last());
            Directory.CreateDirectory(modelDir);
            var csvPath = Path.Combine(modelDir, LABEL_FILENAME);
            var modelPath = Path.Combine(modelDir, MODEL_FILENAME);
            AddLogEntry($"开始下载模型文件: {modelRepo}");
            // AddLogEntry($"字典下载路径: {csvPath}");
            AddLogEntry($"模型下载路径: {modelPath}");

            if (!File.Exists(csvPath) || !File.Exists(modelPath))
            {
                await DownloadFileWithRetry(client, $"https://huggingface.co/{modelRepo}/resolve/main/{LABEL_FILENAME}", csvPath);
            
                await DownloadFileWithRetry(client, $"https://huggingface.co/{modelRepo}/resolve/main/{MODEL_FILENAME}", modelPath);

                if (!VerifyFileIntegrity(modelPath))
                {
                    throw new Exception("下载的模型文件已损坏或不完整。");
                }
                AddLogEntry("已下载新的模型文件。");
            }
            else
            {
                AddLogEntry("使用现有的模型文件。");
            }

            return (csvPath, modelPath);
        }

        private async Task DownloadFileWithRetry(HttpClient client, string url, string filePath, int maxRetries = 3, IProgress<double> progress = null)
        {
            for (int i = 0; i < maxRetries; i++)
            {
                try
                {
                    AddLogEntry($"正在下载文件： {url}");
                    using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();
                    long? totalBytes = response.Content.Headers.ContentLength;
                    using var contentStream = await response.Content.ReadAsStreamAsync();
                    using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

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
                    }
                    while (isMoreToRead);

                    AddLogEntry($"文件下载已完成：{filePath}");
                    return;
                }
                catch (Exception ex)
                {
                    if (File.Exists(filePath))
                    {
                        File.Delete(filePath);
                        AddLogEntry($"下载失败，已删除不完整的文件: {filePath}");
                    }

                    if (i == maxRetries - 1)
                        throw new Exception($"{maxRetries}上次尝试后，文件下载失败: {ex.Message}");
                    AddLogEntry($"下载尝试{{i + 1}}次失败了。将要重新尝试...");
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
                AddLogEntry($"文件一致性检查失败: {ex.Message}");
                return false;
            }
        }

        private async Task DownloadFile(HttpClient client, string url, string filePath)
        {
            using var response = await client.GetAsync(url);
            using var stream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(filePath, FileMode.Create);
            await stream.CopyToAsync(fileStream);
        }

        private void LoadLabels(string csvPath)
        {
            var lines = File.ReadAllLines(csvPath).Skip(1);
            foreach (var (line, index) in lines.Select((l, i) => (l, i)))
            {
                var parts = line.Split(',');
                var name = parts[1].Replace("_", " ");
                _tagNames.Add(name);

                var category = int.Parse(parts[2]);
                switch (category)
                {
                    case 9:
                        _ratingIndexes.Add(index);
                        break;
                    case 0:
                        _generalIndexes.Add(index);
                        break;
                    case 4:
                        _characterIndexes.Add(index);
                        break;
                }
            }
        }
        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            DenseTensor<float> inputTensor,
            float generalThresh,
            bool generalMcutEnabled,
            float characterThresh,
            bool characterMcutEnabled)
        {
            if (!_isModelLoaded)
            {
                AddLogEntry("未加载模型。请在运行 Predict 之前调用 LoadModel。");
                return ("", new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }

            // AddLogEntry($"generalThresh: {generalThresh}");
            // AddLogEntry($"generalMcutEnabled: {generalMcutEnabled}");
            // AddLogEntry($"characterThresh: {characterThresh}");
            // AddLogEntry($"characterMcutEnabled: {characterMcutEnabled}");

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_model.InputMetadata.First().Key, inputTensor) };

            using (var outputs = _model.Run(inputs))
            {
                var predictions = outputs.First().AsEnumerable<float>().ToArray();
                var labels = _tagNames.Zip(predictions, (name, pred) => (name, pred)).ToList();

                var rating = _ratingIndexes.Select(i => labels[i]).ToDictionary(x => x.name, x => x.pred);
                var general = GetFilteredTags(_generalIndexes, labels, generalThresh, generalMcutEnabled);
                var characters = GetFilteredTags(_characterIndexes, labels, characterThresh, characterMcutEnabled);

                var sortedGeneralStrings = string.Join(", ", general.OrderByDescending(x => x.Value).Select(x => x.Key));

                return (sortedGeneralStrings, rating, characters, general);
            }
        }

        public DenseTensor<float> PrepareTensor(Bitmap image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, _modelTargetSize, _modelTargetSize, 3 });

            // Resize image to target size
            using (var resizedImage = new Bitmap(_modelTargetSize, _modelTargetSize))
            {
                using (var graphics = Graphics.FromImage(resizedImage))
                {
                    graphics.DrawImage(image, 0, 0, _modelTargetSize, _modelTargetSize);
                }

                // Process pixels safely using GetPixel
                for (int y = 0; y < _modelTargetSize; y++)
                {
                    for (int x = 0; x < _modelTargetSize; x++)
                    {
                        Color pixel = resizedImage.GetPixel(x, y);
                        tensor[0, y, x, 2] = pixel.R; // R
                        tensor[0, y, x, 1] = pixel.G; // G
                        tensor[0, y, x, 0] = pixel.B; // B
                    }
                }
            }

            return tensor;
        }

        public Bitmap TensorToBitmap(DenseTensor<float> tensor)
        {
            int width = tensor.Dimensions[1];
            int height = tensor.Dimensions[2];
            var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            // Process pixels safely using SetPixel
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = Color.FromArgb(
                        255, // A (fully opaque)
                        (byte)tensor[0, y, x, 2], // R
                        (byte)tensor[0, y, x, 1], // G
                        (byte)tensor[0, y, x, 0]  // B
                    );
                    bitmap.SetPixel(x, y, pixel);
                }
            }

            return bitmap;
        }

        private Dictionary<string, float> GetFilteredTags(
            List<int> indexes,
            List<(string name, float pred)> labels,
            float threshold,
            bool mcutEnabled)
        {
            var tags = indexes.Select(i => labels[i]).ToList();

            if (mcutEnabled)
            {
                threshold = McutThreshold(tags.Select(x => x.pred).ToArray());
            }

            return tags.Where(x => x.pred > threshold).ToDictionary(x => x.name, x => x.pred);
        }

        private float McutThreshold(float[] probs)
        {
            var sortedProbs = probs.OrderByDescending(x => x).ToArray();
            var diffs = sortedProbs.Zip(sortedProbs.Skip(1), (a, b) => a - b).ToArray();
            var t = Array.IndexOf(diffs, diffs.Max());
            return (sortedProbs[t] + sortedProbs[t + 1]) / 2;
        }

        public void Dispose()
        {
            _model?.Dispose();
            AddLogEntry("已释放WDPredictor的资源");
        }
    }
