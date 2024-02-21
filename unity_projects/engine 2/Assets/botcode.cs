using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class CreatureGenerator : Agent
{
    public float spawnHeight = 2.0f;
    public LayerMask groundLayer;
    public PhysicMaterial creaturePhysicsMaterial;
    public float circleRadius = 7f; // Radius for target cube placement

    private Rigidbody torsoRb;
    private HingeJoint[] joints;
    private float[] motorInputValues;
    private GameObject targetCube;
    private Vector3 previousVelocity;

    public override void Initialize()
    {
        Random.InitState(System.DateTime.Now.Millisecond);

        torsoRb = EnsureRigidbody();
        SetTorsoProperties();
        joints = new HingeJoint[12];
        motorInputValues = new float[joints.Length];
        previousVelocity = Vector3.zero;
    }

    public override void OnEpisodeBegin()
    {
        ResetCreature();
        GenerateRandomLimbs();
        CreateTargetCube();
    }

    private Rigidbody EnsureRigidbody()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
            rb.useGravity = true;
            rb.drag = 0.5f;
            rb.angularDrag = 1.0f;
        }
        return rb;
    }

    private void SetTorsoProperties()
    {
        float torsoWidth = Mathf.Max(Random.Range(0.25f, 1.0f), 0.5f);
        float torsoHeight = Mathf.Max(Random.Range(0.5f, 1.5f), 0.6f);
        float torsoDepth = Mathf.Max(Random.Range(0.25f, 1.0f), 0.5f);
        transform.localScale = new Vector3(torsoWidth, torsoHeight, torsoDepth);
        transform.position = new Vector3(transform.position.x, spawnHeight, transform.position.z);
    }

    private void ResetCreature()
    {
        foreach (Transform child in transform)
        {
            Destroy(child.gameObject);
        }
        SetTorsoProperties();
    }

    private void GenerateRandomLimbs()
    {
        int limbCount = Random.Range(1, 5);
        bool[] usedEdges = new bool[12];

        for (int i = 0; i < limbCount; i++)
        {
            int edgeIndex;
            do
            {
                edgeIndex = Random.Range(0, 12);
            } while (usedEdges[edgeIndex]);

            usedEdges[edgeIndex] = true;
            Vector3 limbPosition = GetLimbPositionOnEdge(edgeIndex);
            Vector3 limbDirection = Random.onUnitSphere.normalized;
            GenerateLimb(limbPosition, limbDirection);
        }
    }

    private Vector3 GetLimbPositionOnEdge(int edgeIndex)
    {
        Vector3 edgePosition = CalculateEdgePosition(edgeIndex);
        return transform.TransformPoint(edgePosition);
    }

    private void GenerateLimb(Vector3 position, Vector3 direction)
    {
        Color limbColor = Random.ColorHSV();
        Rigidbody connectedBody = torsoRb;
        int limbParts = Random.Range(2, 4);

        for (int j = 0; j < limbParts; j++)
        {
            GameObject limbPart = CreateLimbPart(connectedBody, position, direction, limbColor);
            connectedBody = limbPart.GetComponent<Rigidbody>();
            position = limbPart.transform.position + limbPart.transform.rotation * Vector3.up * limbPart.transform.localScale.y;
        }
    }

    private GameObject CreateLimbPart(Rigidbody connectedBody, Vector3 position, Vector3 direction, Color color)
    {
        GameObject limbPart = GameObject.CreatePrimitive(PrimitiveType.Cube);
        float limbPartLength = Random.Range(0.75f, 1.5f);
        float limbPartThickness = 0.25f;
        limbPart.transform.localScale = new Vector3(limbPartThickness, limbPartLength, limbPartThickness);
        limbPart.transform.rotation = Quaternion.LookRotation(direction);
        limbPart.transform.position = position;
        limbPart.GetComponent<Renderer>().material.color = color;

        Rigidbody limbPartRb = limbPart.AddComponent<Rigidbody>();
        limbPartRb.mass = Random.Range(1f, 2f);
        limbPartRb.drag = 0.5f;
        limbPartRb.angularDrag = 1.0f;
        limbPartRb.useGravity = true;

        HingeJoint joint = limbPart.AddComponent<HingeJoint>();
        joint.connectedBody = connectedBody;
        ConfigureJoint(joint, direction);

        return limbPart;
    }

    private void ConfigureJoint(HingeJoint joint, Vector3 direction)
    {
        joint.axis = Vector3.Cross(direction, Vector3.up).normalized;
        joint.anchor = new Vector3(0, -0.5f, 0);
        JointLimits limits = new JointLimits
        {
            min = -90,
            max = 90
        };
        joint.limits = limits;
        joint.useLimits = true;

        JointMotor motor = new JointMotor
        {
            force = Random.Range(150, 300),
            targetVelocity = Random.Range(-100, 100),
            freeSpin = false
        };
        joint.motor = motor;
        joint.useMotor = true;
    }

    private Vector3 CalculateEdgePosition(int edgeIndex)
    {
        Vector3 scale = transform.localScale / 2;
        Vector3[] edges = new Vector3[12] {
            new Vector3(0, scale.y, scale.z), new Vector3(0, scale.y, -scale.z),
            new Vector3(scale.x, scale.y, 0), new Vector3(-scale.x, scale.y, 0),
            new Vector3(scale.x, -scale.y, 0), new Vector3(-scale.x, -scale.y, 0),
            new Vector3(0, -scale.y, scale.z), new Vector3(0, -scale.y, -scale.z),
            new Vector3(scale.x, 0, scale.z), new Vector3(-scale.x, 0, scale.z),
            new Vector3(scale.x, 0, -scale.z), new Vector3(-scale.x, 0, -scale.z)
        };

        return edges[edgeIndex % edges.Length];
    }

    private void CreateTargetCube()
    {
        if (targetCube != null)
        {
            Destroy(targetCube);
        }

        targetCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        targetCube.transform.localScale = new Vector3(1f, 1f, 1f);

        float angle = Random.Range(0f, 360f);
        float x = Mathf.Cos(angle * Mathf.Deg2Rad) * circleRadius;
        float z = Mathf.Sin(angle * Mathf.Deg2Rad) * circleRadius;
        Vector3 localPosition = new Vector3(x, spawnHeight + 1f, z);
        targetCube.transform.localPosition = localPosition;
        targetCube.transform.SetParent(this.transform, false);

        Rigidbody cubeRb = targetCube.AddComponent<Rigidbody>();
        cubeRb.useGravity = true;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        List<float> allObservations = new List<float>();

        sensor.AddObservation(torsoRb.transform.localPosition);
        sensor.AddObservation(torsoRb.transform.localRotation);
        sensor.AddObservation(torsoRb.velocity);
        sensor.AddObservation(torsoRb.angularVelocity);

        Vector3 acceleration = (torsoRb.velocity - previousVelocity) / Time.fixedDeltaTime;
        sensor.AddObservation(acceleration);
        previousVelocity = torsoRb.velocity;

        sensor.AddObservation(torsoRb.angularVelocity);

        RaycastHit hit;
        if (Physics.Raycast(torsoRb.position, Vector3.down, out hit, 10f, groundLayer))
        {
            sensor.AddObservation(hit.distance);
        }
        else
        {
            sensor.AddObservation(10f);
        }

        foreach (var joint in joints)
        {
            if (joint != null)
            {
                sensor.AddObservation(joint.angle);
                sensor.AddObservation(joint.velocity);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        if (targetCube != null)
        {
            sensor.AddObservation(transform.InverseTransformPoint(targetCube.transform.position));
        }


        //Debug.Log($"Observations: {torsoRb.transform.localPosition}, {observation2}");
        if (transform.parent.name == "environment")
        {
            Debug.Log($"agent 1s torsoRb.transform.localPosition: {torsoRb.transform.localPosition}");
        }


    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Ensure the action array has the correct size
        if (actionBuffers.ContinuousActions.Length != 12)
        {
            Debug.LogWarning("Expected 12 actions.");
            return;
        }

        // Apply each action to a corresponding joint
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                var motor = joints[i].motor;
                motor.targetVelocity = actionBuffers.ContinuousActions[i] * 100; // Scale the action value
                motor.force = 10000; // You can adjust the force as needed
                joints[i].motor = motor;
            }
        }

        // Reward calculation and target cube logic (if applicable)
        if (targetCube != null)
        {
            float distanceToTarget = Vector3.Distance(this.transform.position, targetCube.transform.position);
            AddReward(-distanceToTarget * 0.001f);

            if (distanceToTarget < 1f)
            {
                AddReward(1.0f);
                EndEpisode();
                CreateTargetCube();
            }
        }
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        for (int i = 0; i < joints.Length; i++)
        {
            // Generate a random value between -1 and 1, then scale it
            continuousActionsOut[i] = UnityEngine.Random.Range(-100f, 100f) * 1000;
        }
    }

}
