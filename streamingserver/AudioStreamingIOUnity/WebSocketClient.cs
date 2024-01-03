using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

public class WebSocketClient : MonoBehaviour
{
    WebSocket websocket;
    public MicrophoneManager micManager;
    public AudioPlayer audioPlayer;

    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8765");

        websocket.OnOpen += OnWebSocketOpen;
        websocket.OnMessage += OnWebSocketMessage;
        websocket.OnClose += OnWebSocketClose;
        
        await websocket.Connect();
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
        websocket.DispatchMessageQueue();
        #endif

        if (websocket.State == WebSocketState.Open)
        {
            SendAudioData();
        }
    }

    private void OnWebSocketOpen()
    {
        Debug.Log("WebSocket connected!");
    }

    private void OnWebSocketMessage(byte[] data)
    {
        audioPlayer.PlayReceivedAudio(data);
    }

    private void OnWebSocketClose(WebSocketCloseCode code)
    {
        Debug.Log("WebSocket closed with code: " + code.ToString());
    }

    private async void SendAudioData()
    {
        byte[] audioData = micManager.GetAudioData();
        if (audioData != null)
        {
            await websocket.Send(audioData);
        }
    }

    private void OnDestroy()
    {
        websocket.Close();
    }
}
