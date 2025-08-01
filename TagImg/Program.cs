using Microsoft.Extensions.Configuration;


List<(string Name, double GeneralThreshold)> _vlmModels = new List<(string, double)>
{
    ("SmilingWolf/wd-eva02-large-tagger-v3", 0.50),
    ("SmilingWolf/wd-vit-large-tagger-v3", 0.25),
    ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", 0.35),
    ("SmilingWolf/wd-vit-tagger-v3", 0.25),
    ("SmilingWolf/wd-swinv2-tagger-v3", 0.25),
    ("SmilingWolf/wd-convnext-tagger-v3", 0.25),
    ("SmilingWolf/wd-v1-4-moat-tagger-v2", 0.35),
    ("SmilingWolf/wd-v1-4-convnext-tagger-v2", 0.35),
    ("SmilingWolf/wd-v1-4-vit-tagger-v2", 0.35),
    ("SmilingWolf/wd-v1-4-convnextv2-tagger-v2", 0.35),
    ("fancyfeast/joytag", 0.5)
};

// 获取当前环境（默认使用 Production）
string environment = Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT") ?? "Development";

// 构建配置
IConfiguration configuration = new ConfigurationBuilder()
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
    .AddJsonFile($"appsettings.{environment}.json", optional: true, reloadOnChange: true)
    .Build();

// 读取配置
string imageFolder = configuration["ImageProcessing:ImageFolder"]
                     ?? throw new InvalidOperationException("ImageFolder 配置未找到");
string targetTags = configuration["ImageProcessing:TargetTags"]
                    ?? throw new InvalidOperationException("TargetTags 配置未找到");
string targetFolderName = configuration["ImageProcessing:TargetFolderName"]
                          ?? throw new InvalidOperationException("TargetFolderName 配置未找到");

// 验证路径是否存在
if (!Directory.Exists(imageFolder))
{
    throw new DirectoryNotFoundException($"图片文件夹 {imageFolder} 不存在");
}

await ImageTagger.ProcessImages(imageFolder, targetTags, targetFolderName, true, _vlmModels[10].Name);

Console.WriteLine($"按 Ctrl+C 停止...");
await Task.Delay(-1);