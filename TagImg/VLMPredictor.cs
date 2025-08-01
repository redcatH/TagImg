using System.Drawing;

namespace TagImg;

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;


    public class VLMPredictor
    {
        private object _currentPredictor;
        private bool _isModelLoaded = false;

        public event EventHandler<string> LogUpdated;
        public bool IsGpuLoaded { get; private set; }

        public async Task LoadModel(string modelRepo, bool useGpu = true)
        {
            if (modelRepo.Contains("joytag"))
            {
                _currentPredictor = new JoyPredictor();
                ((JoyPredictor)_currentPredictor).LogUpdated += OnPredictorLogUpdated;
                await ((JoyPredictor)_currentPredictor).LoadModel(modelRepo, useGpu);
                IsGpuLoaded = ((JoyPredictor)_currentPredictor).IsGpuLoaded;
            }
            else
            {
                _currentPredictor = new WDPredictor();
                ((WDPredictor)_currentPredictor).LogUpdated += OnPredictorLogUpdated;
                await ((WDPredictor)_currentPredictor).LoadModel(modelRepo, useGpu);
                IsGpuLoaded = ((WDPredictor)_currentPredictor).IsGpuLoaded;
            }
            _isModelLoaded = true;
        }

        private void OnPredictorLogUpdated(object sender, string message)
        {
            Console.WriteLine(message);
            // LogUpdated?.Invoke(this, message);
        }

        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            Bitmap image,
            float generalThresh,
            bool generalMcutEnabled,
            float characterThresh,
            bool characterMcutEnabled)
        {
            if (!_isModelLoaded)
            {
                return ("", new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }

            if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.Predict(PrepareTensor(image)!, generalThresh);
            else if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.Predict(PrepareTensor(image)!, generalThresh, generalMcutEnabled, characterThresh, characterMcutEnabled);
            else
                throw new InvalidOperationException("No predictor loaded");
        }

        public DenseTensor<float>? PrepareTensor(Bitmap image)
        {
            if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.PrepareTensor(image);
            else if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.PrepareTensor(image);
            else
                throw new InvalidOperationException("No predictor loaded");
        }

        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            DenseTensor<float> tensor, 
            float generalThresh,
            bool generalMcutEnabled,
            float characterThresh,
            bool characterMcutEnabled)
        {
            if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.Predict(tensor, generalThresh);
            else if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.Predict(tensor, generalThresh, generalMcutEnabled, characterThresh, characterMcutEnabled);
            else
                throw new InvalidOperationException("No predictor loaded");
        }

        public void Dispose()
        { if (_currentPredictor is JoyPredictor joyPredictor)
            {
                joyPredictor.Dispose();
            }
            _isModelLoaded = false;
        }
    }