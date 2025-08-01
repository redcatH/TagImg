using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks.Dataflow;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TagImg;


public class ImageProcessor
{
    private CancellationTokenSource _cts; // 用于取消异步操作
    private int _loadImgProcessedImagesCount; // 跟踪已加载的图像数量
    private int _predictProcessedImagesCount; // 跟踪已预测的图像数量
    private int _totalProcessedImagesCount; // 跟踪总处理完成的图像数量
    private readonly List<ImageInfo> _imageInfos; // 图像信息列表
    private readonly int _vlmUpdateIntervalMs = 500; // 处理速度更新间隔（毫秒）
    private readonly VLMPredictor _vlmPredictor; // VLM 预测器
    private readonly float _generalThresh = 0.35f; // 标签预测阈值
    private readonly bool _generalMcutEnabled = false; // 假设禁用
    private readonly float _characterThresh = 0.85f; // 字符标签阈值
    private readonly bool _characterMcutEnabled = false; // 假设禁用

    private TextTranslator _textTranslator;
    // 构造函数
    public ImageProcessor(List<ImageInfo> imageInfos, VLMPredictor vlmPredictor)
    {
        _textTranslator = new TextTranslator("danbooru-0-zh.txt");
        _imageInfos = imageInfos ?? throw new ArgumentNullException(nameof(imageInfos));
        _vlmPredictor = vlmPredictor ?? throw new ArgumentNullException(nameof(vlmPredictor));
        // 订阅 VLMPredictor 的日志事件
        _vlmPredictor.LogUpdated += (s, e) => Log(e);
    }

    // 主异步方法：处理图像的完整流水线
    public async Task ProcessImagesInAsyncPipeline(int cpuConcurrencyLimit)
    {
        // 初始化取消令牌
        _cts = new CancellationTokenSource();

        // 验证 CPU 并行度
        if (cpuConcurrencyLimit <= 0)
            cpuConcurrencyLimit = Math.Max(1, Environment.ProcessorCount - 2); // 默认并行度为 CPU 核心数减 2

        // 保存总图像数以计算进度
        var totalImages = _imageInfos.Count;
        if (totalImages == 0)
        {
            Log("没有图像需要处理。");
            return;
        }

        // 启动处理速度监控任务（使用 Task.Delay）
        var processingSpeedTask = Task.Run(async () =>
        {
            var stopwatch = Stopwatch.StartNew();
            while (!_cts.IsCancellationRequested)
            {
                // 计算每秒处理速度
                double loadImagesPerSecond = _loadImgProcessedImagesCount / (stopwatch.ElapsedMilliseconds / 1000.0);
                double predictImagesPerSecond = _predictProcessedImagesCount / (stopwatch.ElapsedMilliseconds / 1000.0);
                double totalImagesPerSecond = _totalProcessedImagesCount / (stopwatch.ElapsedMilliseconds / 1000.0);
                double progress = totalImages > 0 ? _totalProcessedImagesCount / (double)totalImages : 0;
                // 输出到控制台
                Log($"加载: {loadImagesPerSecond:F1} 预测: {predictImagesPerSecond:F1} 总计: {totalImagesPerSecond:F1} 张/秒, 进度: {progress:P1}");
                // 等待指定间隔
                await Task.Delay(_vlmUpdateIntervalMs, _cts.Token);
            }
        });

        // 创建信号量以限制 CPU 并行度，防止资源过载
        using var semaphoreCPU = new SemaphoreSlim(cpuConcurrencyLimit, cpuConcurrencyLimit);

        // 数据流块1：加载图像并准备张量
#pragma warning disable CA1416
        var prepareTensorBlock = new TransformBlock<(ImageInfo, Bitmap), (ImageInfo, DenseTensor<float>)?>(
            async input =>
            {
                await semaphoreCPU.WaitAsync(_cts.Token); // 等待信号量，限制并行度
                try
                {
                    Interlocked.Increment(ref _loadImgProcessedImagesCount); // 线程安全地增加计数
                    // 使用 VLMPredictor 准备张量
                    DenseTensor<float>? tensor = await Task.Run(() => _vlmPredictor.PrepareTensor(input.Item2), _cts.Token);
                    input.Item2.Dispose();
                    return tensor == null ? null : (input.Item1, tensor);
                }
                catch (Exception ex)
                {
                    Log($"准备张量失败: {ex.Message}");
                    return null;
                }
                finally
                {
                    semaphoreCPU.Release(); // 释放信号量
                }
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = DataflowBlockOptions.Unbounded, // 并行度由信号量控制
                CancellationToken = _cts.Token
            });
#pragma warning restore CA1416

        // 数据流块2：使用张量预测标签
        var predictTagsBlock = new TransformBlock<(ImageInfo, DenseTensor<float>)?, (ImageInfo, (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>))?>(
            async input =>
            {
                if (!input.HasValue) return null;
                await semaphoreCPU.WaitAsync(_cts.Token); // 限制 CPU 并行度
                try
                {
                    Interlocked.Increment(ref _predictProcessedImagesCount);
                    // 调用 VLMPredictor 的 Predict 方法
                    var result = await Task.Run(() => _vlmPredictor.Predict(
                        input.Value.Item2,
                        _generalThresh,
                        _generalMcutEnabled,
                        _characterThresh,
                        _characterMcutEnabled), _cts.Token);
                    return (input.Value.Item1, result);
                }
                catch (Exception ex)
                {
                    Log($"预测标签失败: {ex.Message}");
                    return null;
                }
                finally
                {
                    semaphoreCPU.Release();
                }
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = DataflowBlockOptions.Unbounded,
                CancellationToken = _cts.Token
            });

        // 数据流块3：处理预测的标签
        var processTagsBlock = new ActionBlock<(ImageInfo, (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>))?>(
            async input =>
            {
                if (!input.HasValue || input.Value.Item1 == null || _cts.Token.IsCancellationRequested) return;
                await semaphoreCPU.WaitAsync(_cts.Token);
                try
                {
                    Interlocked.Increment(ref _totalProcessedImagesCount);
                    // 处理预测的标签
                    ProcessPredictedTags(input.Value.Item1, input.Value.Item2);
                }
                catch (Exception ex)
                {
                    Log($"处理标签失败: {ex.Message}");
                }
                finally
                {
                    semaphoreCPU.Release();
                }
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = DataflowBlockOptions.Unbounded,
                CancellationToken = _cts.Token
            });

        // 链接数据流块，形成处理流水线
        prepareTensorBlock.LinkTo(predictTagsBlock, new DataflowLinkOptions { PropagateCompletion = true });
        predictTagsBlock.LinkTo(processTagsBlock, new DataflowLinkOptions { PropagateCompletion = true });

        // 异步加载图像并输入到流水线
        await foreach (var (imageInfo, bitmap) in LoadImagesAsync())
        {
            if (_cts.IsCancellationRequested) break;
            await semaphoreCPU.WaitAsync(_cts.Token);
            try
            {
                // 将图像信息和 Bitmap 发送到流水线
                await prepareTensorBlock.SendAsync((imageInfo, bitmap), _cts.Token);
                // bitmap.Dispose();
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Log($"加载图像失败: {ex.Message}");
            }
            finally
            {
                semaphoreCPU.Release();
            }
        }

        // 通知流水线输入完成
        prepareTensorBlock.Complete();

        // 等待所有处理完成
        try
        {
            await processTagsBlock.Completion;
        }
        catch (OperationCanceledException)
        {
            // 取消操作
        }
        catch (Exception ex)
        {
            Log($"流水线完成时出错: {ex.Message}");
        }

        // 停止处理速度监控
        _cts.Cancel();
        try
        {
            await processingSpeedTask;
        }
        catch (OperationCanceledException)
        {
            // 忽略取消异常
        }

        // 重置计数器
        _loadImgProcessedImagesCount = 0;
        _predictProcessedImagesCount = 0;
        _totalProcessedImagesCount = 0;

        Log("图像处理完成。");
    }

    // 异步加载图像
    private async IAsyncEnumerable<(ImageInfo, Bitmap)> LoadImagesAsync()
    {
        foreach (var imageInfo in _imageInfos)
        {
            if (_cts.Token.IsCancellationRequested) yield break;
                // 在线程池上加载图像，减少主线程阻塞
                var bitmap = new Bitmap(imageInfo.ImagePath);
                yield return (imageInfo, bitmap);
        }
    }

    // 处理预测的标签
    private void ProcessPredictedTags(ImageInfo imageInfo, (string Description, Dictionary<string, float> GeneralTags, Dictionary<string, float> CharacterTags, Dictionary<string, float> OtherTags) result)
    {
        // Combine all tag keys
        var ts = result.Description.Split(",").ToList();
        var allTags =
            result.GeneralTags.Keys
                .Concat(result.CharacterTags.Keys)
                .Concat(result.OtherTags.Keys)
                .Distinct()
                .ToList();
        allTags.AddRange(ts);
        imageInfo.Tags = allTags;
        imageInfo.TagsChineses = _textTranslator.GetChineseTranslations(allTags).Values.ToList();
    
        var logMessage = new StringBuilder();
        logMessage.AppendLine($"图像 {imageInfo.ImagePath}:");
        logMessage.AppendLine($"  描述: {result.Description}");
        logMessage.AppendLine($"  中文标签: {string.Join(", ", imageInfo.TagsChineses)}");
        logMessage.AppendLine($"  通用标签: {string.Join(", ", result.GeneralTags.Select(kv => $"{kv.Key}: {kv.Value:F2}"))}");
        logMessage.AppendLine($"  角色标签: {string.Join(", ", result.CharacterTags.Select(kv => $"{kv.Key}: {kv.Value:F2}"))}");
        logMessage.AppendLine($"  其他标签: {string.Join(", ", result.OtherTags.Select(kv => $"{kv.Key}: {kv.Value:F2}"))}");
        Log(logMessage.ToString());
    }

    // 日志方法（输出到控制台）
    private void Log(string message)
    {
        Console.WriteLine($"{DateTime.Now:HH:mm:ss} - {message}");
    }
}