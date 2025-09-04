using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
public class SettingsTab : MonoBehaviour
{
    private Animator settingsTabAnim;
    public Button settingsButton;
    public RawImage WebCam;
    private EventTrigger trigger;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        settingsTabAnim = this.GetComponent<Animator>();
        settingsButton.onClick.AddListener(SettingsButtonClicked);
        trigger = WebCam.gameObject.GetComponent<EventTrigger>();
        EventTrigger.Entry clickEvent = new EventTrigger.Entry() { 
            eventID = EventTriggerType.PointerClick
        };
        clickEvent.callback.AddListener(rawImageWebCamClicked);
        trigger.triggers.Add(clickEvent);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void rawImageWebCamClicked(BaseEventData eventData)
    {
        settingsTabAnim.SetBool("isSettingsTabShowed", false);
    }

    void SettingsButtonClicked()
    {
        settingsTabAnim.SetBool("isSettingsTabShowed", true);
    }
}
