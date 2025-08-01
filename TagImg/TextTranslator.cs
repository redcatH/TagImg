using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class TextTranslator
{
    private readonly Dictionary<string, string> translationMap;

    public TextTranslator(string filePath)
    {
        translationMap = new Dictionary<string, string>();
        LoadTranslations(filePath);
    }

    private void LoadTranslations(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Translation file not found: {filePath}");
        }

        try
        {
            var lines = File.ReadAllLines(filePath);
            foreach (var line in lines)
            {
                // Split by comma and ensure at least 3 columns
                var columns = line.Split(',');
                if (columns.Length >= 3)
                {
                    string original = columns[0].Trim();
                    string chinese = columns[2].Trim();
                    if (!string.IsNullOrEmpty(original) && !string.IsNullOrEmpty(chinese))
                    {
                        translationMap[original] = chinese;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            throw new Exception($"Error reading translation file: {ex.Message}");
        }
    }

    public string GetChineseTranslation(string originalText)
    {
        if (string.IsNullOrEmpty(originalText))
        {
            return string.Empty;
        }

        return translationMap.TryGetValue(originalText.Trim(), out var translation) 
            ? translation 
            : originalText.Trim();
    }

    public Dictionary<string, string> GetChineseTranslations(IEnumerable<string> originalTexts)
    {
        if (originalTexts == null)
        {
            return new Dictionary<string, string>();
        }

        return originalTexts
            .Where(text => !string.IsNullOrEmpty(text))
            .Distinct()
            .ToDictionary(
                text => text,
                text => GetChineseTranslation(text),
                StringComparer.OrdinalIgnoreCase
            );
    }
}