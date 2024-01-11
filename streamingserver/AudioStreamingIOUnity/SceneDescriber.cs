using UnityEngine;
using System.Collections.Generic;
using System.Text;

public class SceneDescriber : MonoBehaviour
{
    [System.Serializable]
    public class TagObservability
    {
        public string Tag;
        public float MinDistance;
    }

    public List<TagObservability> observableTags = new List<TagObservability>();
    public bool generateFuzzyDescriptions = true;
    public bool generateRelativeCoordinates = true;
    public bool generateAbsoluteCoordinates = true;

    public string GetSceneDescription()
    {
        StringBuilder descriptionBuilder = new StringBuilder();

        descriptionBuilder.AppendLine("/SceneDescription ");

        foreach (TagObservability tagObservability in observableTags)
        {
            GameObject[] objects = GameObject.FindGameObjectsWithTag(tagObservability.Tag);
            foreach (GameObject obj in objects)
            {
                float distanceToObj = Vector3.Distance(transform.position, obj.transform.position);
                if (distanceToObj <= tagObservability.MinDistance)
                {
                    if (generateFuzzyDescriptions)
                    {
                        string fuzzyDescription = GenerateFuzzyDescription(obj, distanceToObj);
                        descriptionBuilder.AppendLine(fuzzyDescription);
                    }

                    if (generateRelativeCoordinates)
                    {
                        Vector3 relativeCoords = GenerateRelativeCoordinates(obj);
                        descriptionBuilder.AppendLine($"{obj.name} relative coordinates: {relativeCoords}");
                    }

                    if (generateAbsoluteCoordinates)
                    {
                        Vector3 absoluteCoords = obj.transform.position;
                        descriptionBuilder.AppendLine($"{obj.name} absolute coordinates: {absoluteCoords}");
                    }
                }
            }
        }

        return descriptionBuilder.ToString();
    }

    private string GenerateFuzzyDescription(GameObject obj, float distance)
    {
        Vector3 directionToObj = obj.transform.position - transform.position;
        string clockDirection = GetClockDirection(directionToObj);

        return $"{obj.name} is approximately {distance:0.0} meters away, at {clockDirection} position.";
    }

    private Vector3 GenerateRelativeCoordinates(GameObject obj)
    {
        return transform.InverseTransformPoint(obj.transform.position);
    }

    private string GetClockDirection(Vector3 direction)
    {
        float angle = Vector3.SignedAngle(Vector3.forward, direction.normalized, Vector3.up);
        int clockHour = Mathf.RoundToInt((angle + 360) % 360 / 30);
        return $"{clockHour}:00";
    }
}
