using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Mic : MonoBehaviour
{
    public AudioSource aud;
    float time;

    // Start is called before the first frame update
    void Start()
    {
        aud = GetComponent<AudioSource>();    
    }



    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            RecSnd();
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            PlaySnd();
        }
    }


    void PlaySnd()
    {
        aud.Play();
    }

    //C:\Users\dla12\AppData\LocalLow\DefaultCompany\metaverse

    void RecSnd()
    { // 현재 마이크, loop false, 3초, 44100 hz
        aud.clip = Microphone.Start(Microphone.devices[0].ToString(), false, 3, 44100);
        AudioClip myAudio = aud.clip;
        Debug.Log(myAudio);

        Invoke("testInvoke", 3f);
        //Invoke("myInvoke", 3f); // 1초 뒤 시작
        Debug.Log("dd");

    }

    void testInvoke()
    {
        Debug.Log("invoke 3초");
        SavWav.Save("audio.wav", aud.clip);
        Debug.Log("파일 저장완료");
    }

    //public void myInvoke(AudioClip myAudio)
    //{
    //    Debug.Log("invoke 3초");
    //    SavWav.Save("audio.wav", myAudio);
    //    Debug.Log("파일 저장완료");
    //}
}
