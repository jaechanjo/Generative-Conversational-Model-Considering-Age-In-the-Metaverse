using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DllTest;
using Python.Runtime;
using System.IO;
using System;
using System.Text;

public class dlltest : MonoBehaviour
{



    // Start is called before the first frame update
    void Start()
    {
        MakeDllClass debug = new MakeDllClass();
        debug.Test();        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Z))
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
                sys.path.append(os.path.dirname(os.path.expanduser(python_file_path)));
                var fromFile = Py.Import(Path.GetFileNameWithoutExtension(python_file_path));
                
                // pythonnet_test.py 에서 replace 메소드를 호출
                fromFile.InvokeMethod("replace");
                Console.Write(fromFile.InvokeMethod("replace"));
                Debug.Log(fromFile.InvokeMethod("replace"));
            }
            
            // python 환경을 종료한다.
            PythonEngine.Shutdown();

            Console.WriteLine("Press any key...");
            Debug.Log("python");
        }
    }
}
