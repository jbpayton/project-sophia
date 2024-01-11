using System.Collections.Generic;
using UnityEngine;

public class AudioPlayer : MonoBehaviour
{
    public int initialBufferThreshold = 16000/10; // Default value 0.05s, can be adjusted in the Unity Editor
    private bool initialBufferFilled = false;
    public AudioSource audioSource;
    private Queue<float> audioDataQueue;
    private AudioClip dynamicAudioClip;
    private int sampleRate = 16000;
    private int samplesPerMessage = 2048; // Assuming each WebSocket message contains this many samples
    private float[] sampleBuffer;

    void Start()
    {
        if (audioSource == null)
        {
            Debug.LogError("AudioSource component not found!");
            return;
        }

        audioDataQueue = new Queue<float>();
        dynamicAudioClip = AudioClip.Create("DynamicClip", sampleRate, 1, sampleRate, true, OnAudioRead, OnAudioSetPosition);
        audioSource.clip = dynamicAudioClip;
        audioSource.loop = true;
        audioSource.Play();

        sampleBuffer = new float[samplesPerMessage];
    }

    public void PlayReceivedAudio(byte[] data)
    {
        float[] receivedSamples = ConvertBytesToAudioSamples(data);
        foreach (var sample in receivedSamples)
        {
            audioDataQueue.Enqueue(sample);
        }
    }

    private void OnAudioRead(float[] data)
{
    if (!initialBufferFilled)
    {
        // Check if the buffer has enough data to start playback
        if (audioDataQueue.Count >= initialBufferThreshold)
        {
            initialBufferFilled = true;
        }
        else
        {
            // Fill the data with silence if the buffer threshold is not reached
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 0f;
            }
            return;
        }
    }

    // Normal playback
    for (int i = 0; i < data.Length; i++)
    {
        data[i] = audioDataQueue.Count > 0 ? audioDataQueue.Dequeue() : 0f;
    }
}


    private void OnAudioSetPosition(int newPosition)
    {
        // Handle any necessary logic when the audio position is set, if needed
        // (But this is not needed for this kind of application)
    }

    private float[] ConvertBytesToAudioSamples(byte[] data)
    {
        int sampleCount = data.Length / 2; // 2 bytes per sample for 16-bit audio
        float[] audioSamples = new float[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            short sample = System.BitConverter.ToInt16(data, i * 2);
            audioSamples[i] = sample / 32768f; // Convert to range between -1.0 and 1.0
        }

        return audioSamples;
    }
}
