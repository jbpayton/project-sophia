using UnityEngine;
using UnityEngine.AI;
using System.Collections;

public class CharacterNavigation : MonoBehaviour
{
    public NavMeshAgent agent;
    private Animator animator;

    public SceneDescriber sceneDescriber;
    public WebSocketClient webSocketClient;

    public float turnSpeed = 5f; // Speed at which the character turns

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponent<Animator>();

        // Disable automatic movement and rotation
        agent.updatePosition = false;
        agent.updateRotation = false;

        // Start the test routine
        StartCoroutine(SetRandomDestinationRoutine());
    }

    void Update()
    {
        // Calculate the character's speed and direction relative to its path
        Vector3 velocity = agent.desiredVelocity;
        float characterSpeed = velocity.magnitude;
        Vector3 localVelocity = transform.InverseTransformDirection(velocity);
        float walkAngle = Mathf.Atan2(localVelocity.x, localVelocity.z) / Mathf.PI;

        // Update the Animator with the calculated speed and walk angle
        animator.SetFloat("CharacterSpeed", characterSpeed);
        animator.SetFloat("WalkAngle", walkAngle);

        // Smoothly rotate towards the direction of movement
        if (velocity != Vector3.zero)
        {
            Quaternion targetRotation = Quaternion.LookRotation(velocity, Vector3.up);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, turnSpeed * Time.deltaTime);
        }

        // Update the character's position to the agent's position if using root motion
        if (agent.hasPath)
        {
            transform.position = agent.nextPosition;
        }
    }

    public void SetDestination(Vector3 destination)
    {
        agent.SetDestination(destination);
    }

    IEnumerator SetRandomDestinationRoutine()
    {
        while (true)
        {
            // Set a new random destination every 5 seconds
            SetRandomDestination();
            yield return new WaitForSeconds(5f);
        }
    }

    void SetRandomDestination()
    {
        // Define the bounds of your NavMesh area
        float areaSize = 25f;
        float halfArea = areaSize / 2f;

        // Attempt to find a valid random point on the NavMesh
        bool pointFound = false;
        Vector3 finalPosition = Vector3.zero;
        int attempts = 0;
        int maxAttempts = 100; // Prevent infinite loops

        while (!pointFound && attempts < maxAttempts)
        {
            // Generate a random point within the bounds
            Vector3 randomPoint = new Vector3(
                Random.Range(-halfArea, halfArea),
                0, // Assuming a flat surface, y-coordinate is 0
                Random.Range(-halfArea, halfArea)
            );

            NavMeshHit hit;
            if (NavMesh.SamplePosition(randomPoint, out hit, 1f, NavMesh.AllAreas))
            {
                finalPosition = hit.position;
                pointFound = true;
            }

            attempts++;
        }

        if (pointFound)
        {
            SetDestination(finalPosition);
            // log the position
            Debug.Log("New destination: " + finalPosition);

            string description = sceneDescriber.GetSceneDescription();
            webSocketClient.SendMessageToServer(description);
        }
        else
        {
            Debug.LogWarning("Failed to find a valid random point on the NavMesh");
        }
    }

}
