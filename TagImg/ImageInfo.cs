namespace TagImg;

public class ImageInfo
{
    public string ImagePath { get; set; }
    public string AssociatedText { get; set; }
    public List<string> Tags { get; set; }
    public List<string> TagsChineses { get; set; }
    public string ImageFileName { get; set; }

    public ImageInfo()
    {
        Tags = new List<string>();
    }

    public override string ToString()
    {
        return System.IO.Path.GetFileName(ImagePath);
    }
}