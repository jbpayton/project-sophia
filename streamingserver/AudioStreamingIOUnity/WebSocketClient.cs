using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

public class WebSocketClient : MonoBehaviour
{
    WebSocket websocket;
    public MicrophoneManager micManager;
    public AudioPlayer audioPlayer;
    public string webSocketAddress = "ws://192.168.2.232:8765";
    public Animator animator;

    async void Start()
    {
        websocket = new WebSocket(webSocketAddress);

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
        // messages have a 4 byte header that we need to check and remove
        // the first byte is the message type, the next 3 bytes are unused
        byte messageType = data[0];

        // message data is the rest of the bytes if possible without copying
        byte[] messageData = new byte[data.Length - 4];
        System.Buffer.BlockCopy(data, 4, messageData, 0, messageData.Length);

        if (messageType == 0x01)
        {
            // audio data
            audioPlayer.PlayReceivedAudio(messageData);
        }
        else if (messageType == 0x02)
        {
            // convert action data to a string
            string actionData = System.Text.Encoding.UTF8.GetString(messageData);
            // check to see of the message is an emote (begins with /)
            if (actionData.StartsWith("/"))
            {
                // get the emote name
                string emoteName = actionData.Substring(1);
                // print the emote name to the console
                Debug.Log("(emote) /" + emoteName);
                // Set the trigger on the Animator
                if (animator != null)
                {
                    animator.SetTrigger(emoteName);
                }
            }
        }
        else if (messageType == 0x00)
        {
            // convert action data to a string
            string client_message = System.Text.Encoding.UTF8.GetString(messageData);

            // print the emote name to the console
            Debug.Log("(Client Message) " + client_message);
        }
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
            // append a 4 bye header starting with a 0x01 byte to the beginning of the message to indicate audio data
            byte[] messageData = new byte[audioData.Length + 4];
            messageData[0] = 0x01;
            System.Buffer.BlockCopy(audioData, 0, messageData, 4, audioData.Length);
            await websocket.Send(messageData);
        }
    }

    public void SendMessageToServer(string message)
    {
        SendClientMessage(message);
    }

    private async void SendClientMessage(string clientData)
    {
        if (websocket == null || websocket.State != WebSocketState.Open)
        {
            Debug.Log("WebSocket is not connected.");
            return;
        }

        try
        {
            // add header to indicate client message
            byte[] messageData = new byte[clientData.Length + 4];
            messageData[0] = 0x03;
            System.Buffer.BlockCopy(System.Text.Encoding.UTF8.GetBytes(clientData), 0, messageData, 4, clientData.Length);
            await websocket.Send(messageData);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error sending message: {ex.Message}");
        }
    }

    private void OnDestroy()
    {
        websocket.Close();
    }
}
