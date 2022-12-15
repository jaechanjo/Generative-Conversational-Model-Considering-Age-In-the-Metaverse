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


public class pythonnet : MonoBehaviour
{
    private Animator ani;

    [SerializeField] private SayRandomThingsBehaviour _playerComment, _NPC_Comment;

    private AudioSource audio;
    public GameObject gameObject;

    // Start is called before the first frame update
    void Start()
    {
        MakeDllClass debug = new MakeDllClass();
        debug.Test();
        ani = gameObject.GetComponent<Animator>();
        audio = GetComponent<AudioSource>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Z))
        {
            Debug.Log("ZŰ ����");
            // ����ȯ�� ���
            var pythonPath = @"C:\Users\dla12\anaconda3\envs\sesac";

            // Python Ȩ ����
            PythonEngine.PythonHome = pythonPath;

            // Python ���� �ʱ�ȭ
            PythonEngine.Initialize();

            // Python�� �ҷ����� ��� �ڵ�� �Ʒ��� ���� �ȿ� �־�� �Ѵ�.
            using (Py.GIL())
            {
                dynamic os = Py.Import("os"); // python������ 'os'����� import �ϴ� ���
                dynamic sys = Py.Import("sys"); // ���� ����

                // ������ Python ���� ���
                var python_file_path = @"C:\Users\dla12\source\repos\PythonProject\python_file\pythonnet_test.py";
                sys.path.append(os.path.dirname(os.path.expanduser(python_file_path)));
                var fromFile = Py.Import(Path.GetFileNameWithoutExtension(python_file_path));

                // pythonnet_test.py ���� replace �޼ҵ带 ȣ��
                var a = fromFile.InvokeMethod("test");
                
                Debug.Log(a);

                AssetDatabase.Refresh();

                //                string comment = ReadTextOneLine("chat_input");
                string comment = "hi";


                Debug.Log(string.Format("Char Input : {0}", comment));

                _playerComment.thingsToSay[0] = comment;

                _playerComment.SaySomething(comment, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
                _playerComment.thingsToSay[0] = "";

                AssetDatabase.Refresh();

                Invoke("Wait2Sec", 1f);

                //string comment2 = ReadTextOneLine("chat_output");
                string comment2 = "hi2";

                Debug.Log(string.Format("Char output : {0}", comment2));

                _NPC_Comment.thingsToSay[0] = comment2;

                _NPC_Comment.SaySomething(comment2, SpeechBubbleManager.Instance.GetRandomSpeechbubbleType());
                _NPC_Comment.thingsToSay[0] = "";

            }

            // python ȯ���� �����Ѵ�.
            PythonEngine.Shutdown();

            Console.WriteLine("Press any key...");
            Debug.Log("������Ʈ ��");

            
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
}