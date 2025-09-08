using UnityEngine;
using Unity.InferenceEngine;
using System.IO;
using System.Threading.Tasks;
using System.Collections;

public class ModelInference : MonoBehaviour
{
    public ModelAsset onnxModelAsset;
    private Model runtimeModel;
    private Worker worker;
    Tensor<float> inputTensor;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    private bool isStartExecuted = false;
    private RenderTexture _renderTexture;
    private int targetSize = 1216;
    public Rect bbox;
    public IEnumerator m_Schedule;
    int k_LayersPerFrame = 19;

    void Start()
    {
            runtimeModel = ModelLoader.Load(onnxModelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
            inputTensor = new Tensor<float>(new TensorShape(1, 3, targetSize, targetSize));
            _renderTexture = new RenderTexture(targetSize, targetSize, 0, RenderTextureFormat.ARGB32);
            bbox = new Rect(-1, -1, -1, -1); ;
    }

    void NormalizeInPlace(Tensor<float> t, float divisor)
    {
        
        for (int i = 0; i < t.count; i++)
            t[i] /= divisor;
    }

    Texture2D Resize(Texture2D texture2D, int targetX, int targetY)
    {
        RenderTexture rt = new RenderTexture(targetX, targetY, 24);
        RenderTexture.active = rt;
        Graphics.Blit(texture2D, rt);
        Texture2D result = new Texture2D(targetX, targetY);
        result.ReadPixels(new Rect(0, 0, targetX, targetY), 0, 0);
        result.Apply();
        return result;
    }

    public IEnumerator RunYOLO(Texture2D inputImage)
    {


        //Color pixelValue = inputImage.GetPixel(20, 100);
        //Debug.Log("Valor pixel Input Image en (20,100): " + pixelValue.r + ", " + pixelValue.g + ", " + pixelValue.b);
        //Texture2D resizedTex = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);
        //Graphics.ConvertTexture(inputImage,resizedTex);

        //Texture2D resizedTex = Resize(inputImage, targetSize, targetSize);
        Graphics.Blit(inputImage, _renderTexture);

        //inputImage.Reinitialize(targetSize, targetSize);
        // Texture2D resizedTex = inputImage;

        //byte[] bytes = resizedTex.EncodeToPNG();

        //File.WriteAllBytes(Application.dataPath + "\\..\\Saved.png",bytes);

        
        //pixelValue = resizedTex.GetPixel(20,100);
        //Debug.Log("Valor pixel en (20,100): " + pixelValue.r + ", " + pixelValue.g + ", "+ pixelValue.b);
        TextureConverter.ToTensor(_renderTexture, inputTensor);
        //NormalizeInPlace(inputTensor, 255);
        //worker.Schedule(inputTensor);

        //Debug.Log("B Started");
        m_Schedule = worker.ScheduleIterable(inputTensor);
        int it = 0;

        while (m_Schedule.MoveNext())
        {
            if (++it % k_LayersPerFrame == 0)
                yield return null;
        }
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        outputTensor.ReadbackRequest();

        // luego cuando esté listo:
        Tensor<float> pred = outputTensor.ReadbackAndClone();
        //float[] pred = result.DownloadToArray();

        //float[] pred = outputTensor.DownloadToArray();

        //int numPreds = outputTensor.shape[1];
        //int numVals = outputTensor.shape[0];

        int numPreds = outputTensor.shape[2];
        int numVals = outputTensor.shape[1];

        float bestConf = 0.0f;
        Rect bestBox = new Rect();
        int bestClass = -1;

        int originalW = inputImage.width;
        int originalH = inputImage.height;
        float scaleX = (float)originalW / targetSize;
        float scaleY = (float)originalH / targetSize;

        for( int i = 0 ; i < numPreds ; i++ )
        {
            //float x = pred[i * numVals + 0];
            //float y = pred[i * numVals + 1];
            //float w = pred[i * numVals + 2];
            //float h = pred[i * numVals + 3];
            //float conf = pred[i * numVals + 4];
            float x = pred[i + (numPreds * 0)];
            float y = pred[i + (numPreds * 1)];
            float w = pred[i + (numPreds * 2)];
            float h = pred[i + (numPreds * 3)];
            float conf = pred[i + (numPreds * 4)];


            if ( conf > bestConf )
            {
                float maxVal = float.MinValue;
                int classId = -1;
                for( int c = 5; c < numVals ; c++ )
                {
                    float val = pred[i + (numPreds * c)];
                    if( val > maxVal )
                    {
                        maxVal = val;
                        classId = c - 5;
                    }
                }

                int x1 = Mathf.RoundToInt((x - w / 2) * scaleX);
                int y1 = Mathf.RoundToInt((y - h / 2) * scaleY);
                int x2 = Mathf.RoundToInt((x + w / 2) * scaleX);
                int y2 = Mathf.RoundToInt((y + h / 2) * scaleY);

                bestBox = new Rect(x1, y1, x2 - x1, y2 - y1);
                bestConf = conf;
                bestClass = classId;
            }
        }
        if (bestConf > 0)
        {
            //Debug.Log("Rect: " + bestBox.x + ", " + bestBox.y + ", " + bestBox.width + ", " + bestBox.height);
            bbox = bestBox;
            //return bestBox;

        }
        else { bbox = new Rect(0, 0, 0, 0); }
        //Debug.Log("B Finished");
        //Debug.Log("Rect: 0,0,0,0");
        //return new Rect(0, 0, 0, 0);


    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        _renderTexture?.Release();
    }
}
