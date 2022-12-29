using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using DllTest;
using Python.Runtime;
using System.IO;
using System;
using System.Text;
using System.Runtime.InteropServices;
using VikingCrewDevelopment.Demos;
using SpeechBubbleManager = VikingCrewTools.UI.SpeechBubbleManager;
using UnityEditor.Animations;
using System.Threading;
using System.Threading.Tasks;
using Winterdust;

public class pythonnet : MonoBehaviour
{
    private Animator ani;

    [SerializeField] private SayRandomThingsBehaviour _playerComment, _NPC_Comment;

    private AudioSource aud;
    public GameObject GameObject;

    public string char_input;
    public string char_output;
    private AudioSource output; // output 재생기
    [SerializeField] private AudioClip[] clip; // output 

    public int result = 0;

   

    // Start is called before the first frame update
    void Start()
    {
        MakeDllClass debug = new MakeDllClass();
        debug.Test();
        ani = gameObject.GetComponent<Animator>();
        aud = GetComponent<AudioSource>();
        output = GetComponent<AudioSource>();
        Debug.Log("Run() invoked in Start()");
        Debug.Log("Run() returns");
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            RecSnd();
        }

        if (Input.GetKeyDown(KeyCode.Z))
        {
            Run();
        }
    }

    async void Run()
    {
        await Task.Run(() =>
        {

            Debug.Log("Z키 누름");
            // 가상환경 경로
            var pythonPath = @"C:\Users\dla12\anaconda3\envs\sesac";

            // Python 홈 설정
            PythonEngine.PythonHome = pythonPath;

            // Python 엔진 초기화
            PythonEngine.Initialize();

            // Python을 불러오는 모든 코드는 아래의 블럭 안에 있어야 한다.
            using (Py.GIL())
            {
                dynamic os = Py.Import("os"); // python에서의 'os'모듈을 import 하는 방법
                dynamic sys = Py.Import("sys"); // 위와 동일

                // 실행할 Python 파일 경로
                var python_file_path = @"C:\Users\dla12\source\repos\PythonProject\python_file\pythonnet_test.py";
                //var python_file_path = @"C:\Users\dla12\Documents\Developer\Generative-Conversational-Model-Considering-Age-In-the-Metaverse\python\integrated.py";
                sys.path.append(os.path.dirname(os.path.expanduser(python_file_path)));
                var fromFile = Py.Import(Path.GetFileNameWithoutExtension(python_file_path));

                // pythonnet_test.py 에서 replace 메소드를 호출
                var a = fromFile.InvokeMethod("unity_run");
                //Debug.Log("파이썬 실행 완료");

                string char_input = a[0].ToString();
                string char_output = a[1].ToString();


                Debug.Log(char_input);
                Debug.Log(char_output);
                //Debug.Log(a.GetType());
                
            }
            // python 환경을 종료한다.
            PythonEngine.Shutdown();
            //Debug.Log("Python 실행됨");
            
        });
        Debug.Log(char_input);
        Talk();
        PlaySnd();
        //bvh_play();
    }



    void RecSnd()
    { // 현재 마이크, loop false, 3초, 44100 hz
        aud.clip = Microphone.Start(Microphone.devices[0].ToString(), false, 3, 44100);
        AudioClip myAudio = aud.clip;
        Debug.Log(myAudio);

        Invoke("testInvoke", 3f);
        //Invoke("myInvoke", 3f); // 1초 뒤 시작
        Debug.Log("dd");

    }


    void Talk()
    {
        Debug.Log("대화창시작");
        char_input = System.IO.File.ReadAllText("C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/chat_input.txt"); // 텍스트 파일 내용 불러오기
        char_output = System.IO.File.ReadAllText("C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/MetaVH/Assets/output/chat_output.txt"); // 텍스트 파일 내용 불러오기

        AssetDatabase.Refresh();
        //                string comment = ReadTextOneLine("chat_input");
        string comment = char_input;


        Debug.Log(string.Format("Char Input : {0}", comment));

        _playerComment.thingsToSay[0] = comment;

        _playerComment.SaySomething(comment, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
        _playerComment.thingsToSay[0] = "";




        AssetDatabase.Refresh();
        //string comment2 = ReadTextOneLine("chat_output");
        string comment2 = char_output;

        Debug.Log(string.Format("Char output : {0}", comment2));

        _NPC_Comment.thingsToSay[0] = comment2;

        _NPC_Comment.SaySomething(comment2, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
        _NPC_Comment.thingsToSay[0] = "";
    }

    void PlaySnd()
    {
        aud.Play();
        output.clip = clip[0];
        output.Play();
    }


    // bublle message
    string ReadTextOneLine(string _filename)
    {
        TextAsset data = Resources.Load(_filename) as TextAsset;
        // Debug.Log(data);
        StringReader stringReader = new StringReader(data.text);
        string a = stringReader.ReadLine();
        stringReader.Close();
        return a;
    }

    void testInvoke()
    {
        Debug.Log("invoke 3초");
        SavWav.Save("audio.wav", aud.clip);
        Debug.Log("파일 저장완료");
    }

    void bvh_play()
    {

    }
}
