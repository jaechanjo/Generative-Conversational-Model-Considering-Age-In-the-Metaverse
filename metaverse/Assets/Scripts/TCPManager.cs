using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;
using UnityEditor;
using System.Text;
using System;
using System.IO;
using System.Runtime.InteropServices;
using VikingCrewDevelopment.Demos;
using SpeechBubbleManager = VikingCrewTools.UI.SpeechBubbleManager;
using UnityEditor.Animations;


public class TCPManager : MonoBehaviour
{
    TcpClient client;
    string serverIP = "127.0.0.1";
    int port = 8000;
    byte[] receivedBuffer;
    StreamReader reader;
    bool socketReady = false;
    NetworkStream stream;
    private Animator ani;

    [SerializeField] private SayRandomThingsBehaviour _playerComment, _NPC_Comment;

    private AudioSource audio;
    public GameObject gameObject;

    // Start is called before the first frame update
    void Start()
    {
        CheckReceive();
        ani = gameObject.GetComponent<Animator>();
        audio = GetComponent<AudioSource>();
    }

    // Update is called once per frame
    void Update()
    {

        if (socketReady)
        {
            if (stream.DataAvailable)
            {
                receivedBuffer = new byte[100];
                stream.Read(receivedBuffer, 0, receivedBuffer.Length); // stream에 있던 바이트배열 내려서 새로 선언한 바이트배열에 넣기
                string msg = Encoding.UTF8.GetString(receivedBuffer, 0, receivedBuffer.Length); // byte[] to string
                Debug.Log(msg);
                
                
                
                AssetDatabase.Refresh();

                string comment = ReadTextOneLine("chat_input");

                Debug.Log(string.Format("Char Input : {0}", comment));

                _playerComment.thingsToSay[0] = comment;

                _playerComment.SaySomething(comment, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
                _playerComment.thingsToSay[0] = "";

                AssetDatabase.Refresh();
                
                Invoke("Wait2Sec", 1f);

                string comment2 = ReadTextOneLine("chat_output");

                Debug.Log(string.Format("Char output : {0}", comment2));

                _NPC_Comment.thingsToSay[0] = comment2;

                _NPC_Comment.SaySomething(comment2, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
                _NPC_Comment.thingsToSay[0] = "";

                Debug.Log("사운드 실행되어야하는데...");


             
                    if(audio.isPlaying)
                {
                    
                    ani.SetBool("face", true);
                   // audio.Stop();
                }
                else
                {
                     audio.Play();
                    ani.SetBool("face", false);
                }
           

                
                
   
                
                

                // if (msg == "chat_input") // 사용자가 말하는 것 chat_input.txt
                // {
                // string comment = ReadTextOneLine("chat_input");
                // Debug.Log(string.Format("Char Input : {0}", comment));
                // _playerComment.thingsToSay[0] = comment;
                // _playerComment.SaySomething(comment, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
                // }// msg.Split()
                
                // else // AI가 답하는 것
                // {
                //     if (_NPC_Comment != null)
                //     {
                //         string comment = ReadTextOneLine("chat_output");
                //         Debug.Log(string.Format("Char output : {0}", comment));
                //         _NPC_Comment.thingsToSay[0] = comment;
                //         _NPC_Comment.SaySomething(comment, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
                //     }
                // }
               
                // if (msg == "chat_input.txt") // 사용자가 말하는 것 chat_input.txt
                // {
                //     string comment = ReadTextOneLine(msg);
                //     Debug.Log(string.Format("Char Input : {0}", comment));
                //     _playerComment.thingsToSay[0] = comment;
                // }
                // else // AI가 답하는 것
                // {
                //     if (_NPC_Comment != null)
                //     {
                //         string comment = ReadTextOneLine(msg);
                //     Debug.Log(string.Format("Char Input : {0}", comment));
                //         _playerComment.thingsToSay[0] = comment;
                //     }
                // }
            }
        }
    }
    string ReadTextOneLine(string _filename)
    {
        TextAsset data = Resources.Load(_filename) as TextAsset;
        // Debug.Log(data);
        StringReader stringReader = new StringReader(data.text);
        string a = stringReader.ReadLine();
        stringReader.Close();
        return a;
    }

    void CheckReceive()
    {
        if (socketReady) return;
        try
        {
            client = new TcpClient(serverIP, port);

            if (client.Connected)
            {
                stream = client.GetStream();
                Debug.Log("Connect Success");
                socketReady = true;
            }

        }
        catch (Exception e)
        {
            Debug.Log("On client connect exception " + e);
        }

    }

    void OnApplicationQuit()
    {
        CloseSocket();
    }

    void CloseSocket()
    {
        if (!socketReady) return;

        reader.Close();
        client.Close();
        socketReady = false;
    }

    void Wait2Sec()
    {
        Debug.Log("1초가 지났습니다.");
    }

}