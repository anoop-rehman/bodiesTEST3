using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class RLAlgorithm : MonoBehaviour
{
    public List<float> allCreaturesObservationVector = new();
    const int OBSERVATION_VECTOR_SIZE = 47;
    public GameObject agent1;
    public GameObject agent2;
    public GameObject agent3;
    public GameObject agent4;
    public GameObject agent5;
    public GameObject agent6;
    public GameObject agent7;
    public GameObject agent8;
    public GameObject agent9;

    private List<GameObject> agents;


    // Start is called before the first frame update
    void Start()
    {
        agents = new List<GameObject>
        {
            agent1,
            agent2,
            agent3,
            agent4,
            agent5,
            agent6,
            agent7,
            agent8,
            agent9
        };
    }

    // Update is called once per frame
    void Update()
    {
        foreach (GameObject agent in agents)
        {
            allCreaturesObservationVector.AddRange(agent.GetComponent<CreatureGenerator>().CreatureObservationVector);
        }

        Debug.Log(allCreaturesObservationVector);
    }
}
