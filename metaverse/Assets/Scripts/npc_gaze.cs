using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class npc_gaze : MonoBehaviour
{
    public Transform Player;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void FixedUpdate () 
    {
        transform.LookAt(Player);
    }
}
