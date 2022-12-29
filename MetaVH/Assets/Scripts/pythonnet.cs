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

public class pythonnet : MonoBehaviour
{
    private Animator ani;

    [SerializeField] private SayRandomThingsBehaviour _playerComment, _NPC_Comment;

    private AudioSource Audio;
    public GameObject GameObject;

    public string char_input;
    public string char_output;

    public int result = 0;

    // Start is called before the first frame update
    void Start()
    {
        MakeDllClass debug = new MakeDllClass();
        debug.Test();
        ani = gameObject.GetComponent<Animator>();
        Audio = GetComponent<AudioSource>();
        Debug.Log("Run() invoked in Start()");
        Debug.Log("Run() returns");
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Z))
        {
            Run();
        }
    }

    async void Run()
    {
        await Task.Run(() =>
        {

            Debug.Log("ZŰ ����");
            // ����ȯ�� ���
            var pythonPath = @"C:\Users\dla12\anaconda3\envs\sesac";

            // Python Ȩ ����
            PythonEngine.PythonHome = pythonPath;

            // Python ���� �ʱ�ȭ
            PythonEngine.Initialize();

            // Python�� �ҷ����� ��� �ڵ�� �Ʒ��� �� �ȿ� �־�� �Ѵ�.
            using (Py.GIL())
            {
                dynamic os = Py.Import("os"); // python������ 'os'����� import �ϴ� ���
                dynamic sys = Py.Import("sys"); // ���� ����

                // ������ Python ���� ���
                var python_file_path = @"C:\Users\dla12\source\repos\PythonProject\python_file\pythonnet_test.py";
                //var python_file_path = @"C:\Users\dla12\Documents\Developer\Generative-Conversational-Model-Considering-Age-In-the-Metaverse\python\integrated.py";
                sys.path.append(os.path.dirname(os.path.expanduser(python_file_path)));
                var fromFile = Py.Import(Path.GetFileNameWithoutExtension(python_file_path));

                // pythonnet_test.py ���� replace �޼ҵ带 ȣ��
                var a = fromFile.InvokeMethod("unity_run");
                //Debug.Log("���̽� ���� �Ϸ�");

                string char_input = a[0].ToString();
                string char_output = a[1].ToString();


                Debug.Log(char_input);
                Debug.Log(char_output);
                //Debug.Log(a.GetType());
                
            }
            // python ȯ���� �����Ѵ�.
            PythonEngine.Shutdown();
            //Debug.Log("Python �����");
            
        });
        Debug.Log(char_input);
        Talk(char_input, char_output);
    }



    void Talk(string char_input, string char_output)
    {
        Debug.Log("��ȭâ����");
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
}
