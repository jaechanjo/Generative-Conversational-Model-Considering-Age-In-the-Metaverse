using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DllTest;


public class dlltest : MonoBehaviour
{

//#region  Unity Method

    // Start is called before the first frame update
    void Start()
    {
        MakeDllClass debug = new MakeDllClass();
        debug.Test();        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
