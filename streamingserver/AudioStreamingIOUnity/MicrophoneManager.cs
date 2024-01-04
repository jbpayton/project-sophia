using UnityEngine;

public class MicrophoneManager : MonoBehaviour
{
    private AudioClip microphoneClip;
    private int sampleRate = 16000;
    private int lastSample = 0;
    private float[] sampleBuffer;

    void Start()
    {
        microphoneClip = Microphone.Start(null, true, 10, sampleRate); // 10 seconds buffer
        sampleBuffer = new float[1024 * 10]; // Adjust buffer size based on expected data rate
    }

    public byte[] GetAudioData()
    {
        if (Microphone.IsRecording(null))
        {
            int currentPosition = Microphone.GetPosition(null);
            if (currentPosition < lastSample)
            {
                // Handle microphone loop-around
                lastSample = 0;
            }

            int sampleCount = currentPosition - lastSample;
            if (sampleCount > 0)
            {
                if (sampleCount > sampleBuffer.Length)
                {
                    // Avoid buffer overflow, adjust buffer size if this happens frequently
                    sampleCount = sampleBuffer.Length;
                }

                microphoneClip.GetData(sampleBuffer, lastSample);
                lastSample = currentPosition;

                return ConvertAudioSamplesToBytes(sampleBuffer, sampleCount);
            }
        }
        return null;
    }

    private byte[] ConvertAudioSamplesToBytes(float[] samples, int sampleCount)
    {
        byte[] audioData = new byte[sampleCount * 2];
        int rescaleFactor = 32767; // to convert float to Int16

        for (int i = 0; i < sampleCount; i++)
        {
            short tempShort = (short)(samples[i] * rescaleFactor);
            byte[] tempBytes = System.BitConverter.GetBytes(tempShort);

            audioData[i * 2] = tempBytes[0];
            audioData[i * 2 + 1] = tempBytes[1];
        }

        return audioData;
    }
}
