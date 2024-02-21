using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class FloorModifier : MonoBehaviour
{
    public float bumpiness = 0.5f;
    private Mesh mesh;

    void Start()
    {
        mesh = GetComponent<MeshFilter>().mesh;
        Vector3[] vertices = mesh.vertices;

        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i].y += Mathf.PerlinNoise(vertices[i].x * bumpiness, vertices[i].z * bumpiness);
        }

        mesh.vertices = vertices;
        mesh.RecalculateNormals(); // To ensure proper lighting
    }
}
