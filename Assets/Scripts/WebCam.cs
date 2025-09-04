using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

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
        QualitySettings.vSyncCount = 0;

        Application.targetFrameRate = 60;

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

        //Move Bounding Box to origin
        //I have no idea, why the below code works.
        boundingBox.GetComponent<RectTransform>().anchoredPosition = new Vector3(-actualCam.width,actualCam.height,0);

        StartCoroutine(DelayedInference(1.2f));

    }

    IEnumerator DelayedInference(float seconds)
    {
        yield return new WaitForSeconds(0.5f);
        while (true)
        {
            if (isCamAvailable && actualCam != null && actualCam.isPlaying)
            {
                yield return new WaitForSeconds(seconds);

                texture2D = new Texture2D(actualCam.width, actualCam.height, TextureFormat.ARGB32, false);
                texture2D.SetPixels32(actualCam.GetPixels32());
                Color pixelValue = actualCam.GetPixel(20, 100);
                //Debug.Log("Valor WebCamTexture pixel en (20,100): " + pixelValue.r + ", " + pixelValue.g + ", " + pixelValue.b);
                texture2D.Apply();
                Resources.UnloadUnusedAssets();
                boundingBox.GetComponent<RectTransform>().anchoredPosition = new Vector3(-actualCam.width, actualCam.height, 0);
                boundingBoxYOLO = inf.RunYOLO(texture2D);
                Vector2 actualPos = boundingBox.GetComponent<RectTransform>().anchoredPosition;
                boundingBox.GetComponent<RectTransform>().anchoredPosition = new Vector2(actualPos.x + (boundingBoxYOLO.x*2.0f)+180, actualPos.y - (boundingBoxYOLO.y*2.0f)-40);
                boundingBox.GetComponent<RectTransform>().sizeDelta=new Vector2(boundingBoxYOLO.width*2,boundingBoxYOLO.height * 2);
            }
        }
       
    }

    private void Update()
    {
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
