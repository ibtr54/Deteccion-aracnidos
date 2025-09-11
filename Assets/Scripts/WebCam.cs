using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Threading.Tasks;

#if UNITY_ANDROID
using UnityEngine.Android;
#endif

public class WebCam : MonoBehaviour
{
    private bool isCamAvailable;
    private WebCamTexture actualCam;
    private Texture defaultBackground;

    [Header("UI References")]
    public RawImage background;
    public AspectRatioFitter fit;
    public GameObject ButtonCamera;
    public Transform canvasTransform;
    public RectTransform CCLabel;

    private GameObject[] ButtonsCamera;
    private int indexCamera;
    private WebCamDevice[] devices;
    public ModelInference inf;
    private Texture2D texture2D;
    RenderTexture rt;
    public GameObject boundingBox;
    private Rect boundingBoxYOLO;
    private Canvas canvas;
    private RectTransform canvasRectTransform;
    private Vector3 originScreen;
    private Vector3[] cornersPos;
    private RectTransform webCamRectTransform;

#if UNITY_IOS || UNITY_WEBGL
    private bool CheckPermissionAndRaiseCallbackIfGranted(UserAuthorization authenticationType, Action authenticationGrantedAction)
    {
        if (Application.HasUserAuthorization(authenticationType))
        {
            authenticationGrantedAction?.Invoke();
            return true;
        }
        return false;
    }

    private IEnumerator AskForPermissionIfRequired(UserAuthorization authenticationType, Action authenticationGrantedAction)
    {
        if (!CheckPermissionAndRaiseCallbackIfGranted(authenticationType, authenticationGrantedAction))
        {
            yield return Application.RequestUserAuthorization(authenticationType);
            if (!CheckPermissionAndRaiseCallbackIfGranted(authenticationType, authenticationGrantedAction))
                Debug.LogWarning($"Permission {authenticationType} Denied");
        }
    }
#elif UNITY_ANDROID
    private void PermissionCallbacksPermissionGranted(string permissionName)
    {
        StartCoroutine(DelayedCameraInitialization());
    }

    private IEnumerator DelayedCameraInitialization()
    {
        yield return null;
        InitializeCamera();
    }

    private void PermissionCallbacksPermissionDenied(string permissionName)
    {
        Debug.LogWarning($"Permission {permissionName} Denied");
    }

    private void AskCameraPermission()
    {
        var callbacks = new PermissionCallbacks();
        callbacks.PermissionDenied += PermissionCallbacksPermissionDenied;
        callbacks.PermissionGranted += PermissionCallbacksPermissionGranted;
        Permission.RequestUserPermission(Permission.Camera, callbacks);
    }
#endif

    void Start()
    {
        canvas = FindFirstObjectByType<Canvas>();
        canvasRectTransform = canvas.GetComponent<RectTransform>();
        QualitySettings.vSyncCount = 0;
        cornersPos = new Vector3[4];
        Application.targetFrameRate = 60;
        webCamRectTransform = this.GetComponent<RectTransform>();
#if UNITY_IOS || UNITY_WEBGL
        StartCoroutine(AskForPermissionIfRequired(UserAuthorization.WebCam, () => { InitializeCamera(); }));
        return;
#elif UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            AskCameraPermission();
            return;
        }
#endif
        InitializeCamera();
        //float h = canvasRectTransform.rect.height;
        //float w = canvasRectTransform.rect.width;
        //float offset = h > 1500 ? 200 : 60;
        //originScreen = new Vector3(-w - offset, 0, 0);
    }

    public void SetIndexCamera(int index)
    {
        if (devices == null || devices.Length == 0)
        {
            Debug.LogWarning("No cameras available.");
            return;
        }

        if (index < 0 || index >= devices.Length)
        {
            Debug.LogWarning("Invalid camera index.");
            return;
        }

        indexCamera = index;
        UpdateWebcam();
    }

    public void UpdateWebcam()
    {
        if (devices == null || devices.Length == 0) return;

        if (actualCam != null && actualCam.isPlaying)
        {
            actualCam.Stop();
        }

        if (indexCamera < 0 || indexCamera >= devices.Length)
        {
            Debug.LogWarning("IndexCamera out of range, you are trying to access to a inmaginary camera.");
            return;
        }

        actualCam = new WebCamTexture(devices[indexCamera].name, Screen.width, Screen.height);
        actualCam.Play();

        if (background != null)
            background.texture = actualCam;

        isCamAvailable = true;
    }

    private void InitializeCamera()
    {
        if (background != null)
            defaultBackground = background.texture;

        devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogWarning("No Camera Detected.");
            isCamAvailable = false;
            return;
        }

        ButtonsCamera = new GameObject[devices.Length];
        float offsetPosLY = 85.0f;

        for (int i = 0; i < devices.Length; ++i)
        {
            int supa_i = i;

            if (ButtonCamera != null && canvasTransform != null)
            {
                ButtonsCamera[i] = Instantiate(ButtonCamera, canvasTransform);
                ButtonsCamera[i].transform.SetParent(canvasTransform, false);

                TextMeshProUGUI tmpText = ButtonsCamera[i].GetComponentInChildren<TextMeshProUGUI>();
                if (tmpText != null)
                    tmpText.text = devices[i].name;

                if (CCLabel != null)
                    ButtonsCamera[i].GetComponent<RectTransform>().position = new Vector3(CCLabel.position.x - 25.0f, CCLabel.position.y - offsetPosLY, CCLabel.position.z);

                Button btn = ButtonsCamera[i].GetComponent<Button>();
                if (btn != null)
                    btn.onClick.AddListener(() => { SetIndexCamera(supa_i); });
            }

            offsetPosLY += 70.0f;

            actualCam = new WebCamTexture(devices[i].name, Screen.width, Screen.height);
            indexCamera = supa_i;
        }

        if (actualCam == null)
        {
            Debug.LogWarning("No backCamera Detected.");
            return;
        }

        actualCam.Play();
        if (background != null)
            background.texture = actualCam;

        isCamAvailable = true;

        webCamRectTransform.GetWorldCorners(cornersPos);
        originScreen = cornersPos[1];
        //Move Bounding Box to origin
        //I have no idea, why the below code works.
        boundingBox.GetComponent<RectTransform>().position = originScreen;
        StartCoroutine(DelayedInference(1.2f));

    }

    IEnumerator DelayedInference(float seconds)
    {
        //yield return new WaitForSeconds(0.7f);
        texture2D = new Texture2D(actualCam.width, actualCam.height, TextureFormat.ARGB32, false);
        RectTransform bboxRTransform = boundingBox.GetComponent<RectTransform>();
        //bboxRTransform.position = originScreen;
        float h = canvasRectTransform.rect.height;
        float w = canvasRectTransform.rect.width;
        //Vector3 originScreen = new Vector3(-w - (h > 1500? 200 : 60) , 0, 0);
        //yield return new WaitForSeconds(0.5f);

        float scale_factor_x = (cornersPos[2].x - cornersPos[1].x) / actualCam.width;
        float scale_factor_y = (cornersPos[1].y - cornersPos[0].y) / actualCam.height;

        //Debug.Log("Scale X: " + scale_factor_x);
        //Debug.Log("Scale Y: " + scale_factor_y);
       
        while (true)
        {
            if (isCamAvailable && actualCam != null && actualCam.isPlaying)
            {
                //await Task.Delay((int)(seconds * 1000.0f));
                
                yield return new WaitForSeconds((int) seconds);

                //What the fuck?
                //float scale_factor_x = h > 1700 ? 2.0f : h > 1596 && h < 1602 ? 1.9f : h > 1500 ? 2.0f : h < 1350 ? 1.61f : 1.7f;
                //float scale_factor_y = h > 1700 ? 2.35f : h > 1596 && h < 1602 ? 2.1f : h > 1500 ? 2.15f : h < 1350 ? 1.75f : 1.85f;




                texture2D.SetPixels32(actualCam.GetPixels32());
                //Color pixelValue = actualCam.GetPixel(20, 100);
                //Debug.Log("Valor WebCamTexture pixel en (20,100): " + pixelValue.r + ", " + pixelValue.g + ", " + pixelValue.b);
                texture2D.Apply(false);
                //_ = Resources.UnloadUnusedAssets();
                //bboxRTransform.anchoredPosition = new Vector3(-actualCam.width, actualCam.height, 0);

                //Debug.Log(canvasRectTransform.localScale.x);


                //Dont use Get WorldCorners, just get the Top Left Corner anchored position and then convert it to world position.
                //The anchor position of webCamera is the center of all canvas.
                //webCamRectTransform.GetWorldCorners(cornersPos);
                originScreen = new Vector3(-webCamRectTransform.rect.width * 0.5f, webCamRectTransform.rect.height * 0.5f, 0);
                //bboxRTransform.position = originScreen;

                //Debug.Log("A Started");

                //yield return StartCoroutine(inf.RunYOLO(texture2D));
                yield return StartCoroutine(inf.RunOwn(texture2D));
                //RunOwn
                //yield return null;
                boundingBoxYOLO = inf.bbox;

                float scale = Mathf.Min(webCamRectTransform.rect.width / texture2D.width, webCamRectTransform.rect.height / texture2D.height);

                //webCamRectTransform.rect.width is different to (cornersPos[2].x - cornersPos[1].x) and I dont now Why??????
                //Debug.Log(webCamRectTransform.rect.width);
                //Debug.Log((cornersPos[2].x - cornersPos[1].x));

                float offsetX = (webCamRectTransform.rect.width - (texture2D.width * scale)) * 0.5f;
                float offsetY = (webCamRectTransform.rect.height - (texture2D.height * scale)) * 0.5f;

                float scaledW = (boundingBoxYOLO.width / texture2D.width) * (texture2D.width * scale);
                float scaledH = (boundingBoxYOLO.height / texture2D.height) * (texture2D.height * scale);

                float scaledX = (boundingBoxYOLO.x / texture2D.width) * (texture2D.width * scale);
                float scaledY = (boundingBoxYOLO.y / texture2D.height) * (texture2D.height * scale);


                float localX = scaledX + scaledW / 2f;
                float localY = -scaledY - scaledH / 2f;

                localX += offsetX;
                localY += offsetY;

                Vector3 worldPos = webCamRectTransform.TransformPoint(originScreen + new Vector3(localX, localY, 0f));


                bboxRTransform.position = worldPos;
                bboxRTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, scaledW);
                bboxRTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, scaledH);
                //Debug.Log("A finised");
            }
        }
       
    }

    private void Update()
    {
        //GetWorldCorners function save my life, thanks to the guy who programs this function.
        //webCamRectTransform.GetWorldCorners(cornersPos);


        //float h = canvasRectTransform.rect.height;
        //float w = canvasRectTransform.rect.width;
        //float offset = h>1596 && h<1602 ? 220 : h > 1500 ? 260 : h < 1350 ? -20 : 60;
        //originScreen = new Vector3(-w - offset, 0, 0);
        //originScreen = cornersPos[1];
        if (!isCamAvailable || actualCam == null) return;

        if (fit != null)
        {
            float ratio = (float)actualCam.width / (float)actualCam.height;
            fit.aspectRatio = ratio;
        }

        if (background != null)
        {
            float scaleY = actualCam.videoVerticallyMirrored ? -1.0f : 1.0f;
            background.rectTransform.localScale = new Vector3(1.0f, scaleY, 1.0f);
            int orient = -actualCam.videoRotationAngle;
            background.rectTransform.localEulerAngles = new Vector3(0, 0, orient);
        }
    }
}
