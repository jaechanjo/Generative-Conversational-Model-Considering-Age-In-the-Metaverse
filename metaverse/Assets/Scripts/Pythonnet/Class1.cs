using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Pythonnet
{
    public class Class1
    {
        public int c;
        public void AddValues(int a, int b)
        {
            c = a + b;
        }

        public void Test()
        {
            Debug.Log("DLL에서 로드 했습니다.");
        }

        public static int GenerationRandom(int min, int max)
        {
            System.Random rand = new System.Random();
            return rand.Next(min, max);
        }
    }
}
