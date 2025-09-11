using UnityEngine;
using Unity.InferenceEngine;
using System.IO;
using System.Threading.Tasks;
using System.Collections;
using Unity.VisualScripting;
using System.Collections.Generic;
using static UnityEngine.Rendering.ProbeAdjustmentVolume;
using System.Linq;
using static UnityEditor.PlayerSettings;
using System;
using UnityEngine.UI;
using System.Net.NetworkInformation;

public class ModelInference : MonoBehaviour
{
    public ModelAsset onnxModelAsset;
    public ModelAsset onnxModelAssetOwn;
    private Model runtimeModel;
    private Model runtimeModelOwn;
    private Worker worker;
    private Worker workerOwn;
    Tensor<float> inputTensor;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    private bool isStartExecuted = false;
    private RenderTexture _renderTexture;
    private RenderTexture _renderTextureOwn;
    private int targetSize = 1216;
    private int targetSizeOwn = 256;
    public Rect bbox;
    public IEnumerator m_Schedule;
    public IEnumerator m_ScheduleOwn;
    int k_LayersPerFrame = 18;
    int k_LayersPerFrameOwn = 70;
    public RawImage test_raw;
    private string[] classes = {"animal snake", "animal spider", "car", "no object", "scorpion" };
    void Start()
    {
        runtimeModel = ModelLoader.Load(onnxModelAsset);
        runtimeModelOwn = ModelLoader.Load(onnxModelAssetOwn);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        workerOwn = new Worker(runtimeModelOwn, BackendType.GPUCompute);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, targetSize, targetSize));
        _renderTexture = new RenderTexture(targetSize, targetSize, 0, RenderTextureFormat.ARGB32);
        _renderTextureOwn = new RenderTexture(targetSizeOwn, targetSizeOwn, 0, RenderTextureFormat.ARGB32);
        bbox = new Rect(-1, -1, -1, -1);
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

    private int getCenterPosition(byte[,] kernel)
    {
        return (int)(kernel.GetLength(0) / 2);
    }


    int[,] Dilate_Operator(byte[,] kernel_struct, int[,] image, int imageX, int imageY)
    {
        int centerPosRow = getCenterPosition(kernel_struct);
        int centerPosColumn = getCenterPosition(kernel_struct);
        int deltaColumn_Left = centerPosColumn - 0;
        int deltaColumn_Right = (kernel_struct.GetLength(0) - 1) - centerPosColumn;
        int deltaRow_Up = centerPosRow - 0;
        int deltaRow_Down = (kernel_struct.GetLength(0) - 1) - centerPosRow;
        int[,] processed_image = new int[imageY, imageX];
        for ( int i = 0 ; i < imageY ; i++ ){
            for( int j = 0 ; j < imageX ; j++ )
            {
                int bound_left = j - deltaColumn_Left;
                int bound_right = j + deltaColumn_Right;
                int bound_up = i - deltaRow_Up;
                int bound_down = i + deltaRow_Down;
                if(bound_left >= 0 && bound_right< (imageX-1) && bound_up >= 0 && bound_down< (imageY - 1))
                {
                    int alt_i = 0;
                    int alt_j = 0;
                    int pixel_value = 0;
                    for( int w = bound_up ; w <= bound_down ; w++ )
                    {
                        alt_j = 0;
                        for( int z = bound_left ; z <= bound_right ; z++ )
                        {
                            if (image[w, z] == kernel_struct[alt_i, alt_j])
                            {
                                pixel_value = 1;
                            }
                            alt_j += 1;
                        }
                        alt_i += 1;
                    }
                    processed_image[i, j] = pixel_value;                            
                }
                else
                {
                    processed_image[i, j] = 0;
                }

            }
        }
        return processed_image;
    }

    int[,] Erosion_Operator(byte[,] kernel_struct, int[,] image, int imageX, int imageY)
    {
        int centerPosRow = getCenterPosition(kernel_struct);
        int centerPosColumn = getCenterPosition(kernel_struct);
        int deltaColumn_Left = centerPosColumn - 0;
        int deltaColumn_Right = (kernel_struct.GetLength(0) - 1) - centerPosColumn;
        int deltaRow_Up = centerPosRow - 0;
        int deltaRow_Down = (kernel_struct.GetLength(0) - 1) - centerPosRow;
        int[,] processed_image = new int[imageY, imageX];
        for (int i = 0; i < imageY; i++)
        {
            for (int j = 0; j < imageX; j++)
            {
                int bound_left = j - deltaColumn_Left;
                int bound_right = j + deltaColumn_Right;
                int bound_up = i - deltaRow_Up;
                int bound_down = i + deltaRow_Down;
                if (bound_left >= 0 && bound_right < (imageX - 1) && bound_up >= 0 && bound_down < (imageY - 1))
                {
                    int alt_i = 0;
                    int alt_j = 0;
                    int pixel_value = 1;
                    for (int w = bound_up; w <= bound_down; w++)
                    {
                        alt_j = 0;
                        for (int z = bound_left; z <= bound_right; z++)
                        {
                            if (image[w, z] != kernel_struct[alt_i, alt_j])
                            {
                                pixel_value = 0;
                            }
                            alt_j += 1;
                        }
                        alt_i += 1;
                    }
                    processed_image[i, j] = pixel_value;
                }
                else
                {
                    processed_image[i, j] = 0;
                }

            }
        }
        return processed_image;
    }

    private int[,] ToGrayscale(int[,,] src, int imageX, int imageY)
    {
        int w = imageX;
        int h = imageY;
        int[,] gray = new int[h, w];

        //Color32[] pixels = tex.GetPixels32();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                //Color32 c = pixels[y * w + x];
                gray[y, x] = (byte)(0.299f * src[0,y,x] + 0.587f * src[1, y, x] + 0.114f * src[2, y, x]);
            }
        }
        return gray;
    }

    private int[,,] ConvertScaleAbs(int[,,]src, int imageX, int imageY, float alpha, float beta)
    {
        //Texture2D dst = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false);
        //Color[] pixels = src.GetPixels();
        int[,,] image = new int[3, imageY, imageX];
        for ( int i = 0 ; i < imageY ; i++ )
        {
            for( int j = 0 ; j < imageX ; j++ )
            {
                //Color c = pixels[i];
                // Escalamos valores RGB [0,1] a [0,255]
                float r = Mathf.Clamp(Mathf.Abs(alpha * src[0,i,j] + beta),0.0f,255.0f);
                float g = Mathf.Clamp(Mathf.Abs(alpha * src[1, i, j] + beta), 0.0f, 255.0f);
                float b = Mathf.Clamp(Mathf.Abs(alpha * src[2, i, j] + beta), 0.0f, 255.0f);

                //pixels[i] = new Color(r, g, b, c.a);
                image[0, i, j] = (int)(r);
                image[1, i, j] = (int)(g);
                image[2, i, j] = (int)(b);

            }           
        }
        return image;
    }

    private int[,,] Texture2D2Byte(Texture2D tex)
    {
        int w = tex.width;
        int h = tex.height;
        int[,,] image = new int[3, h, w];
        Color32[] pixels = tex.GetPixels32();
        for( int y = 0 ; y < h ; y++ )
        {
            for( int x = 0 ; x < w ; x++ )
            {
                Color32 c = pixels[y * w + x];
                image[0, y, x] = c.r;
                image[1, y, x] = c.g;
                image[2, y, x] = c.b;
            }
        }
        return image;
    }

    private int[,,] Correlation(int[,,] originalImage, int imageX, int imageY, int[,] _kernel, int kernelSize)
    {
        if (originalImage == null || _kernel == null)
            return null;
        //int** kernel = new int*[3];
        //for (int i = 0; i < 3; ++i)
        //{
        //    kernel[i] = new int[3];
        //}
        //FlipKernel(_kernel, kernel);
        int centerPosRow = 1, centerPosColumn = 1;
        int deltaColumn_Left = centerPosColumn - 0;
        int deltaColumn_Right = (kernelSize - 1) - centerPosColumn;
        int deltaRow_Up = centerPosRow - 0;
        int deltaRow_Down = (kernelSize - 1) - centerPosRow;

        //byte[,] processed_image = new byte[imageY, imageX];
        //processed_image->Width = imageX;
        //processed_image->Height = imageY;

        // Matriz temporal para almacenar intensidades
        //int** tempPixelValues = new int*[imageX];
        //for (int i = 0; i < imageX; ++i)
        //{
        //    tempPixelValues[i] = new int[imageY];
        //}

        int[,,] tempPixelValues = new int[3,imageY, imageX];

        // Calcular los valores de los píxeles utilizando el kernel
        for (int i = 0; i < imageY; ++i)
        {
            for (int j = 0; j < imageX; ++j)
            {
                int bound_left = j - deltaColumn_Left;
                int bound_right = j + deltaColumn_Right;
                int bound_up = i - deltaRow_Up;
                int bound_down = i + deltaRow_Down;

                if (bound_left >= 0 && bound_right < imageX && bound_up >= 0 && bound_down < imageY)
                {
                    int alt_i = 0, alt_j = 0;
                    int r_pixel_value = 0;
                    int g_pixel_value = 0;
                    int b_pixel_value = 0;

                    for (int w = bound_up; w <= bound_down; ++w)
                    {
                        alt_j = 0;
                        for (int z = bound_left; z <= bound_right; ++z)
                        {
                            r_pixel_value += originalImage[0, w, z] * _kernel[alt_i,alt_j];
                            g_pixel_value += originalImage[1, w, z] * _kernel[alt_i, alt_j];
                            b_pixel_value += originalImage[2, w, z] * _kernel[alt_i, alt_j];
                            alt_j++;
                        }
                        alt_i++;
                    }

                    // Guardar el valor calculado en la matriz temporal
                    tempPixelValues[0,i,j] = r_pixel_value;
                    tempPixelValues[1, i, j] = g_pixel_value;
                    tempPixelValues[2, i, j] = b_pixel_value;
                }
                else
                {
                    tempPixelValues[0, i, j] = 0; // Rellenar bordes con 0
                    tempPixelValues[1, i, j] = 0;
                    tempPixelValues[2, i, j] = 0;
                }
            }
        }
        return tempPixelValues;
    }

    private List<List<Vector2Int>> FindContours(int[,] binary, int imageX, int imageY)
    {
        int w = imageX;
        int h = imageY;

        int[,] visited = new int[h, w];
        List<List<Vector2Int>> contours = new List<List<Vector2Int>>();

        int[] dx = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dy = { -1, -1, -1, 0, 0, 1, 1, 1 };
                
        for (int y = 1; y < h - 1; y++)
        {
            for (int x = 1; x < w - 1; x++)
            {
                if ( binary[y, x] == 1 && visited[y, x] == 0 )
                {
                    List<Vector2Int> contour = new List<Vector2Int>();
                    Stack<Vector2Int> stack = new Stack<Vector2Int>();
                    stack.Push(new Vector2Int(x, y));
                    visited[y, x] = 1;

                    while (stack.Count > 0)
                    {
                        Vector2Int p = stack.Pop();
                        contour.Add(p);

                        for (int k = 0; k < 8; k++)
                        {
                            int nx = p.x + dx[k];
                            int ny = p.y + dy[k];
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                            {
                                if (binary[ny, nx] == 1 && visited[ny, nx] == 0)
                                {
                                    visited[ny, nx] = 1;
                                    stack.Push(new Vector2Int(ny, nx));
                                }
                            }
                        }
                    }
                    contours.Add(contour);
                }
            }
        }
        return contours;
    }

    private float GetArea(List<Vector2Int> contour, ref Rect bbox)
    {
        int minY = 999999999;
        int maxY = -999999999;
        int minX = 999999999;
        int maxX = -999999999;
        foreach (Vector2Int p in contour)
        {
            if( p.y < minY )
            {
                minY = p.y;
            }
            if( p.y > maxY )
            {
                maxY = p.y;
            }
            if( p.x < minX )
            {
                minX = p.x;
            }
            if( p.x > maxX )
            {
                maxX = p.x;
            }
        }
        bbox.Set(minX,minY, maxX - minX, maxY - minY);
        float area = (maxX-minX) * (maxY-minY);
        return area;
    }

    private int[,] NormalizeArray(int[,] arrayUnormalized)
    {
        int minR = 999999999;
        int maxR = -999999999;
        
        int y = arrayUnormalized.GetLength(0);
        int x = arrayUnormalized.GetLength(1);


        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                minR = arrayUnormalized[i, j] < minR ? arrayUnormalized[i, j] : minR;
                
            }
        }

        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                maxR = arrayUnormalized[i, j] > maxR ? arrayUnormalized[i, j] : maxR;
                
                //maxG = GetGValue(arrayUnormalized->Canvas->Pixels[i][j]) > maxG ? GetGValue(arrayUnormalized->Canvas->Pixels[i][j]) : maxG;
                //maxB = GetBValue(arrayUnormalized->Canvas->Pixels[i][j]) > maxB ? GetBValue(arrayUnormalized->Canvas->Pixels[i][j]) : maxB;
            }
        }

        float r;
        float g;
        float b;
        int[,] result = new int[y, x];
        //result->Height = y;
        //result->Width = x;
        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                r = (float)(arrayUnormalized[i, j] - minR) / (float)(maxR - minR);
               
                //r = (arrayUnormalized[i,j] - minR) * 255 / (maxR - minR);
                //int g = ((GetGValue(arrayUnormalized->Canvas->Pixels[i][j]) - minR) * 255) / (maxR - minR);
                //int b = ((GetBValue(arrayUnormalized->Canvas->Pixels[i][j]) - minR) * 255) / (maxR - minR);
                result[i, j] = (int)(r * 255);
                
                //result[i][j][0] = ((arrayUnormalized[i][j][0] - minR) * 255) / (maxR - minR);
                //result[i][j][1] = ((arrayUnormalized[i][j][1] - minG) * 255) / (maxG - minG);
                //result[i][j][2] = ((arrayUnormalized[i][j][2] - minB) * 255) / (maxB - minB);
            }
        }

        //delete arrayUnormalized;
        return result;
    }

    private int[,,]NormalizeArray(int[,,] arrayUnormalized, int x, int y)
    {
        int minR = 999999999;
        int maxR = -999999999;
        int minG = 999999999;
        int maxG = -999999999;
        int minB = 999999999;
        int maxB = -999999999;


        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                minR = arrayUnormalized[0,i,j] < minR ? arrayUnormalized[0,i,j] : minR;
                minG = arrayUnormalized[1,i,j] < minG ? arrayUnormalized[1,i,j] : minG;
                minB = arrayUnormalized[2,i,j] < minB ? arrayUnormalized[2,i,j] : minB;
            }
        }

        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                maxR = arrayUnormalized[0, i, j] > maxR ? arrayUnormalized[0, i, j] : maxR;
                maxG = arrayUnormalized[1, i, j] > maxG ? arrayUnormalized[1, i, j] : maxG;
                maxB = arrayUnormalized[2, i, j] > maxB ? arrayUnormalized[2, i, j] : maxB;
                //maxG = GetGValue(arrayUnormalized->Canvas->Pixels[i][j]) > maxG ? GetGValue(arrayUnormalized->Canvas->Pixels[i][j]) : maxG;
                //maxB = GetBValue(arrayUnormalized->Canvas->Pixels[i][j]) > maxB ? GetBValue(arrayUnormalized->Canvas->Pixels[i][j]) : maxB;
            }
        }

        float r;
        float g;
        float b;
        int[,,] result = new int[3,y,x];
        //result->Height = y;
        //result->Width = x;
        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                r = (float)(arrayUnormalized[0, i, j] - minR) / (float)(maxR - minR);
                g = (float)(arrayUnormalized[1, i, j] - minG) / (float)(maxG - minG);
                b = (float)(arrayUnormalized[2, i, j] - minB) / (float)(maxB - minB);
                //r = (arrayUnormalized[i,j] - minR) * 255 / (maxR - minR);
                //int g = ((GetGValue(arrayUnormalized->Canvas->Pixels[i][j]) - minR) * 255) / (maxR - minR);
                //int b = ((GetBValue(arrayUnormalized->Canvas->Pixels[i][j]) - minR) * 255) / (maxR - minR);
                result[0,i,j] = (int)(r * 255);
                result[1, i, j] = (int)(g * 255);
                result[2, i, j] = (int)(b * 255);
                //result[i][j][0] = ((arrayUnormalized[i][j][0] - minR) * 255) / (maxR - minR);
                //result[i][j][1] = ((arrayUnormalized[i][j][1] - minG) * 255) / (maxG - minG);
                //result[i][j][2] = ((arrayUnormalized[i][j][2] - minB) * 255) / (maxB - minB);
            }
        }

        //delete arrayUnormalized;
        return result;
    }

    private int[,,] ClampImage(int[,,] arrayUnormalized)
    {
        
        int y = arrayUnormalized.GetLength(1);
        int x = arrayUnormalized.GetLength(2);
        int[,,] result = new int[3, y, x];
        float r;
        float g;
        float b;
        for (int i = 0; i < y; ++i)
        {
            for (int j = 0; j < x; ++j)
            {
                r = Mathf.Clamp(arrayUnormalized[0, i, j],0,255);
                g = Mathf.Clamp(arrayUnormalized[1, i, j], 0, 255);
                b = Mathf.Clamp(arrayUnormalized[2, i, j], 0, 255);

                result[0, i, j] = (int)r;
                result[1, i, j] = (int)g;
                result[2, i, j] = (int)b;
            }
        }

        return result;
    }
    public List<Rect> Rects(Texture2D inputImage)
    {
    // texture = (Texture2D)SpecMat.GetTexture(name);
        //Texture2D textureClone = new Texture2D(inputImage.width, inputImage.height, inputImage.format, false);
        //textureClone.LoadRawTextureData(inputImage.GetRawTextureData());
        //textureClone.Apply();

        //yield return null;
        int division_value = 100;
        int[,,] image_t = Texture2D2Byte(inputImage);
        //ImShow(image_t);
        image_t = ConvertScaleAbs(image_t, inputImage.width, inputImage.height, 3.5f, 10.0f);
        //ImShow(image_t);
        int[,] laplacianFilter = { { 0, 1, 0 }, { 1, -4, 1 }, { 0, 1, 0 } };
        image_t = Correlation(image_t, inputImage.width, inputImage.height, laplacianFilter, 3);
        //ImShow(image_t);
        image_t = ClampImage(image_t);
        //ImShow(image_t);
        byte[,] kernel_Rectangle = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
        int[,] image_t_gray = ToGrayscale(image_t, inputImage.width, inputImage.height);
        //ImShow(image_t_gray);
        int[,] image_t_B = new int[inputImage.height, inputImage.width];
        for (int i = 0; i < inputImage.height; i++)
        {
            for (int j = 0; j < inputImage.width; j++)
            {
                if (image_t_gray[i, j] > division_value)
                    image_t_B[i, j] = 1;
                else
                {
                    image_t_B[i, j] = 0;
                }
            }
        }

        int[,] image_t_gray_B = Dilate_Operator(kernel_Rectangle, image_t_B, inputImage.width, inputImage.height);
        //ImShow(NormalizeArray(image_t_gray_B));
        int[,] image_B = new int[inputImage.height, inputImage.width];
        for (int i = 0; i < inputImage.height; i++)
        {
            for (int j = 0; j < inputImage.width; j++)
            {
                if (image_t_gray[i, j] > division_value)
                    image_B[i, j] = 1;
                else
                {
                    image_B[i, j] = 0;
                }
            }
        }

        int[,] dilate_image = Dilate_Operator(kernel_Rectangle, image_B, inputImage.width, inputImage.height);
        int[,] erosion_image = Erosion_Operator(kernel_Rectangle, image_B, inputImage.width, inputImage.height);
        int[,] gradiente = new int[inputImage.height, inputImage.width];


        for (int i = 0; i < inputImage.height; i++)
        {
            for (int j = 0; j < inputImage.width; j++)
            {
                gradiente[i, j] = Mathf.Abs(dilate_image[i, j] - erosion_image[i, j]);
            }
        }
        //ImShow(NormalizeArray(gradiente));

        int[,] closed = Dilate_Operator(kernel_Rectangle, gradiente, inputImage.width, inputImage.height);
        closed = Dilate_Operator(kernel_Rectangle, closed, inputImage.width, inputImage.height);

        //ImShow(NormalizeArray(closed));

        int[,] closed_uint8 = new int[inputImage.height, inputImage.width];
        for (int i = 0; i < inputImage.height; i++)
        {
            for (int j = 0; j < inputImage.width; j++)
            {
                closed_uint8[i, j] = closed[i, j] | image_t_B[i,j];
            }
        }
        //ImShow(NormalizeArray(closed_uint8));
        closed_uint8 = Erosion_Operator(kernel_Rectangle, closed_uint8, inputImage.width, inputImage.height);
        //ImShow(NormalizeArray(closed_uint8));
        //for (int i = 0; i < inputImage.height; i++)
        //{
        //    for (int j = 0; j < inputImage.width; j++)
        //    {
        //        closed_uint8[i, j] = closed_uint8[i, j] * 255;
        //    }
        //}

        List<List<Vector2Int>> contours = FindContours(closed_uint8,inputImage.width,inputImage.height);

        List<Rect> bounding_boxes = new List<Rect>();

        float area = 0;
        for( int i = 0 ; i < contours.Count ; i++)
        {
            Rect tempBoundingBox = new Rect();
            area = GetArea(contours[i],ref tempBoundingBox);
            if (area > 120 && area < (inputImage.width* inputImage.height* 0.80) && tempBoundingBox.width > 15 && tempBoundingBox.height > 15){
                //x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.Add(tempBoundingBox);
                //bounding_boxes.append([x, y, w, h])
                //cv2.rectangle(am_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            }
        }
        //gradiente[:,:] = numpy.absolute(dilate_image[:,:] - erosion_image[:,:])
        return bounding_boxes;
    }

    private int ArgMax(Tensor<float> values, ref float? maxVal)
    {
        maxVal = null; //nullable so this works even if you have all super-low negatives
        int index = -1;
        for ( int i = 0 ; i < values.count ; i++)
        {
            float thisNum = values[i];
            if (!maxVal.HasValue || thisNum > maxVal.Value)
            {
                maxVal = thisNum;
                index = i;
            }
        }
        return index;
    }

    public static Texture2D ToTexture2D(int[,,] array)
    {
        int channels = array.GetLength(0);
        int height = array.GetLength(1);
        int width = array.GetLength(2);

        if (channels < 3)
        {
            Debug.LogError("The array must have 3 channels (R,G,B).");
            return null;
        }

        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color32[] pixels = new Color32[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte r = (byte)Mathf.Clamp(array[0, y, x], 0, 255);
                byte g = (byte)Mathf.Clamp(array[1, y, x], 0, 255);
                byte b = (byte)Mathf.Clamp(array[2, y, x], 0, 255);

                // Nota: invertimos el eje Y para que la imagen no salga "al revés"
                int idx = y * width + x;//(height - 1 - y) * width + x;
                pixels[idx] = new Color32(r, g, b, 255);
            }
        }

        tex.SetPixels32(pixels);
        tex.Apply();


        return tex;
    }

    public static Texture2D ToTexture2D(int[,] array)
    {
        //
        //int channels = array.GetLength(0);
        int channels = 1;
        int height = array.GetLength(0);
        int width = array.GetLength(1);

        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color32[] pixels = new Color32[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte r = (byte)Mathf.Clamp(array[y, x], 0, 255);
                byte g = (byte)Mathf.Clamp(array[y, x], 0, 255);
                byte b = (byte)Mathf.Clamp(array[y, x], 0, 255);

                
                int idx = y * width + x;//(height - 1 - y) * width + x;
                pixels[idx] = new Color32(r, g, b, 255);
            }
        }

        tex.SetPixels32(pixels);
        tex.Apply();


        return tex;
    }

    private void ImShow(int[,,] img)
    {
        Texture2D tex = ToTexture2D(img);       
        byte[] rawData = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes("../debug_frame.png", rawData);
        Debug.Log("Guardada imagen en debug_frame.png");
    }

    private void ImShow(int[,] img)
    {
        Texture2D tex = ToTexture2D(img);
        byte[] rawData = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes("../debug_frame.png", rawData);
        Debug.Log("Guardada imagen en debug_frame.png");
    }

    public IEnumerator RunOwn(Texture2D inputImage)
    {
        List<Rect> bounding_boxes = Rects(inputImage);
        int num_show = Mathf.Min(2500, bounding_boxes.Count);

        float supa_max = -9999999;
        for (int i = 0; i < num_show; i++)
        {
            Rect r = bounding_boxes[i];
            int x = Mathf.Clamp((int)r.x, 0, inputImage.width - 1);
            int y = Mathf.Clamp((int)r.y, 0, inputImage.height - 1);
            int w = Mathf.Clamp((int)r.width, 1, inputImage.width - x);
            int h = Mathf.Clamp((int)r.height, 1, inputImage.height - y);
            Color[] pixels = inputImage.GetPixels((int)x, (int)y, (int)w, (int)h);
            Texture2D cropped = new Texture2D((int)w, (int)h, TextureFormat.RGB24, false);
            cropped.SetPixels(pixels);
            cropped.Apply();

            //subimages.Add(cropped);
            //pos.Add(new Vector2Int(r.x, r.y));
            //size_img.Add(new Vector2Int(r.width, r.height));
            //Texture2D resized = new Texture2D(256, 256, TextureFormat.RGB24, false);
            //Graphics.ConvertTexture(cropped, resized);
            Graphics.Blit(cropped, _renderTextureOwn);
            Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, targetSizeOwn, targetSizeOwn));
            TextureConverter.ToTensor(_renderTextureOwn, inputTensor);

            int it = 0;
            m_ScheduleOwn = workerOwn.ScheduleIterable(inputTensor);
            while ( m_ScheduleOwn.MoveNext() )
            {
                if ( ++it % k_LayersPerFrameOwn == 0 )
                    yield return null;
            }
            Tensor<float> outputTensor = workerOwn.PeekOutput() as Tensor<float>;

            outputTensor.ReadbackRequest();

            Tensor<float> pred = outputTensor.ReadbackAndClone();
            //float maxpred = 0;
            float? max_score = 0;
            int predictedIndex = ArgMax(pred, ref max_score);

            string predictedClass = classes[predictedIndex];

            if( predictedClass.Equals("no object") || predictedClass.Equals("car") )
            {
                continue;
            }

            if( max_score.HasValue)
            {
                if( max_score > supa_max)
                {
                    bbox = bounding_boxes[i];
                }
            }

            // Liberar memoria
            inputTensor.Dispose();
            //output.Dispose();
        }

        yield return null; 
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
