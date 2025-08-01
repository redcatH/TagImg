using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using TagImg;

public class ImageTagger
{
    private static ConcurrentDictionary<string, ImageInfo> imageCache = new ConcurrentDictionary<string, ImageInfo>();
    private static string cacheFilePath;
    private static string targetFolder;
    private static string[] targetTags;
    private static VLMPredictor predictor;

    public static async Task ProcessImages(string imageFolder, string targetTagsInput, string targetFolderName,
        bool watchDirectory = false, string vlmModelNmae = "fancyfeast/joytag")
    {
        // 1. Validate inputs
        if (string.IsNullOrWhiteSpace(imageFolder) || !Directory.Exists(imageFolder))
        {
            Console.WriteLine("图像文件夹路径无效。");
            return;
        }

        if (string.IsNullOrEmpty(targetTagsInput))
        {
            Console.WriteLine("未输入有效标签。");
            return;
        }

        if (string.IsNullOrWhiteSpace(targetFolderName))
        {
            Console.WriteLine("文件夹名称无效。");
            return;
        }

        // 2. Initialize cache and global variables
        cacheFilePath = Path.Combine(Directory.GetCurrentDirectory(), "image_tags_cache.json");
        targetFolder = Path.Combine(Directory.GetCurrentDirectory(), targetFolderName);
        targetTags = targetTagsInput.Split(',')
            .Select(tag => tag.Trim())
            .Where(tag => !string.IsNullOrEmpty(tag))
            .ToArray();

        if (targetTags.Length == 0)
        {
            Console.WriteLine("未提供有效标签。");
            return;
        }

        // Create target folder
        try
        {
            if (!Directory.Exists(targetFolder))
            {
                Directory.CreateDirectory(targetFolder);
                Console.WriteLine($"创建文件夹: {targetFolder}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"创建文件夹失败: {ex.Message}");
            return;
        }

        // 3. Load cache
        if (File.Exists(cacheFilePath))
        {
            try
            {
                string json = File.ReadAllText(cacheFilePath);
                imageCache = JsonSerializer.Deserialize<ConcurrentDictionary<string, ImageInfo>>(json) ??
                             new ConcurrentDictionary<string, ImageInfo>();
                Console.WriteLine($"Loaded {imageCache.Count} cached image tags.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load cache: {ex.Message}");
            }
        }

        // 4. Initialize predictor
        predictor = new VLMPredictor();
        try
        {
            await predictor.LoadModel(vlmModelNmae, useGpu: false);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"加载模型失败: {ex.Message}");
            predictor?.Dispose();
            return;
        }

        // 5. Process existing images
        var imagePaths = Directory
            .EnumerateFiles(imageFolder, "*.*")
            .Where(file => file.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                           file.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            .GroupBy(file =>
            {
                var fileName = Path.GetFileNameWithoutExtension(file);
                // Extract base name by removing _0 or _720 suffix
                var parts = fileName.Split('_');
                if (parts.Length >= 5 && (parts[^1] == "0" || parts[^1] == "720"))
                {
                    return string.Join("_", parts.Take(parts.Length - 1)) + Path.GetExtension(file);
                }

                return fileName + Path.GetExtension(file);
            })
            .Select(group =>
            {
                var files = group.OrderByDescending(f =>
                {
                    var fileName = Path.GetFileNameWithoutExtension(f);
                    var suffix = fileName.Split('_').Last();
                    return suffix == "720" ? 1 : 0; // Prioritize _720 over _0
                }).ToList();
                return new ImageInfo { ImagePath = files.First(),ImageFileName = Path.GetFileName(files.First()) };
            })
            .ToList();

        if (!imagePaths.Any())
        {
            Console.WriteLine("未找到图像文件。");
        }
        else
        {
            await ProcessImageBatch(imagePaths);
        }

        // 6. Start directory watching if requested
        if (watchDirectory)
        {
            StartDirectoryWatcher(imageFolder);
            Console.WriteLine($"开始监听目录{imageFolder}");
            // await Task.Delay(-1); // Keep the program running
        }
        else
        {
            predictor?.Dispose();
        }
    }

    private static async Task ProcessImageBatch(List<ImageInfo> imagePaths)
    {
        // Filter images that need processing
        var imagesToProcess = imagePaths
            .Where(img => !imageCache.ContainsKey(ComputeMD5Hash(img.ImagePath)))
            .ToList();

        // Process new images
        if (imagesToProcess.Any())
        {
            var processor = new ImageProcessor(imagesToProcess, predictor);
            try
            {
                await processor.ProcessImagesInAsyncPipeline(cpuConcurrencyLimit: 0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"处理图像失败: {ex.Message}");
            }

            // Update cache with new results
            foreach (var imageInfo in imagesToProcess)
            {
                string md5 = ComputeMD5Hash(imageInfo.ImagePath);
                imageCache[md5] = imageInfo;
            }

            // Save updated cache
            try
            {
                string json = JsonSerializer.Serialize(imageCache, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(cacheFilePath, json);
                Console.WriteLine($"Saved {imageCache.Count} image tags to cache.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save cache: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("All images are cached, skipping analysis.");
        }

        // Merge cached tags
        foreach (var imageInfo in imagePaths)
        {
            string md5 = ComputeMD5Hash(imageInfo.ImagePath);
            if (imageCache.TryGetValue(md5, out var cachedInfo))
            {
                imageInfo.Tags = cachedInfo.Tags;
                imageInfo.TagsChineses = cachedInfo.TagsChineses;
            }
        }

        // Move images with matching tags
        int movedCount = 0;
        foreach (var imageInfo in imagePaths)
        {
            await ProcessSingleImage(imageInfo);
            if (imageInfo.ImagePath.StartsWith(targetFolder))
                movedCount++;
        }

        Console.WriteLine($"\n移动完成，共移动 {movedCount} 张图像。");
    }

    private static readonly object cacheLock = new object();


    private static async Task ProcessSingleImage(ImageInfo imageInfo)
    {
        string md5 = ComputeMD5Hash(imageInfo.ImagePath);
        if (!imageCache.ContainsKey(md5))
        {
            // Process single image
            var singleImageList = new List<ImageInfo> { imageInfo };
            var processor = new ImageProcessor(singleImageList, predictor);
            try
            {
                await processor.ProcessImagesInAsyncPipeline(cpuConcurrencyLimit: 1);
                imageCache[md5] = imageInfo;

                // Save updated cache
                try
                {
                    string json =
                        JsonSerializer.Serialize(imageCache, new JsonSerializerOptions { WriteIndented = true });
                    File.WriteAllText(cacheFilePath, json);
                    Console.WriteLine($"Updated cache for new image: {imageInfo.ImagePath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to save cache: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"处理单张图像失败: {ex.Message}");
            }
        }
        else
        {
            // Load tags from cache
            imageInfo.Tags = imageCache[md5].Tags;
            imageInfo.TagsChineses = imageCache[md5].TagsChineses;
        }

        // Check for matching tags and move if necessary
        bool hasMatch = targetTags.Any(targetTag =>
            imageInfo.Tags?.Any(tag => tag.Contains(targetTag, StringComparison.OrdinalIgnoreCase)) == true ||
            imageInfo.TagsChineses?.Any(tag => tag.Contains(targetTag, StringComparison.OrdinalIgnoreCase)) == true);

        if (hasMatch)
        {
            try
            {
                string fileName = Path.GetFileName(imageInfo.ImagePath);
                string baseFileName = fileName;
                string newFileName = fileName;

                // 判断文件名是否包含时间戳（格式：yyyyMMdd_HHmmss_fff_）
                string pattern = @"^\d{8}_\d{6}_\d{3}_";
                if (Regex.IsMatch(fileName, pattern))
                {
                    // 如果包含时间戳，提取原始文件名（去掉时间戳前缀）
                    baseFileName = Regex.Replace(fileName, pattern, "");
                }

                // 为新文件名添加当前时间戳
                string timeStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                newFileName = $"{timeStamp}_{baseFileName}";

                // 检查目标文件夹中是否已存在相同原始文件名的文件（忽略时间戳）
                string destinationPath = Path.Combine(targetFolder, newFileName);
                bool fileExists = Directory.GetFiles(targetFolder)
                    .Any(f => Regex.Replace(Path.GetFileName(f), pattern, "") == baseFileName);

                if (!fileExists)
                {
                    File.Copy(imageInfo.ImagePath, destinationPath);
                    Console.WriteLine($"移动图像 {fileName} 到 {targetFolder}，新文件名：{newFileName}");
                    string newMd5 = ComputeMD5Hash(destinationPath);
                    imageCache[newMd5] = imageInfo;
                    imageInfo.ImagePath = destinationPath;

                    // 保存更新后的缓存
                    try
                    {
                        string json = JsonSerializer.Serialize(imageCache,
                            new JsonSerializerOptions { WriteIndented = true });
                        File.WriteAllText(cacheFilePath, json);
                        Console.WriteLine($"更新缓存成功，新文件名：{newFileName}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"更新缓存失败：{ex.Message}");
                    }
                }
                else
                {
                    Console.WriteLine($"目标文件（原始文件名：{baseFileName}）已存在，跳过复制。");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"移动图像 {imageInfo.ImagePath} 失败: {ex.Message}");
            }
        }
    }

    private static void StartDirectoryWatcher(string imageFolder)
    {
        var watcher = new FileSystemWatcher
        {
            Path = imageFolder,
            NotifyFilter = NotifyFilters.FileName | NotifyFilters.CreationTime,
            Filter = "*.*"
        };

        watcher.Created += async (sender, e) =>
        {
            lock (cacheLock)
            {
                if (e.FullPath.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                    e.FullPath.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine($"检测到新文件: {e.FullPath}");
                    // await Task.Delay(100); // Wait briefly to ensure file is fully written
                    var imageInfo = new ImageInfo { ImagePath = e.FullPath };
                    ProcessSingleImage(imageInfo).ConfigureAwait(false).GetAwaiter().GetResult();
                }
            }
        };
        watcher.IncludeSubdirectories = false; // 可选：是否监控子目录
        watcher.EnableRaisingEvents = true;
    }

    private static string ComputeMD5Hash(string filePath)
    {
        try
        {
            using (var md5 = MD5.Create())
            {
                using (var stream = File.OpenRead(filePath))
                {
                    byte[] hash = md5.ComputeHash(stream);
                    return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                }
            }
        }
        catch
        {
            return string.Empty; // Handle file access errors gracefully
        }
    }
}