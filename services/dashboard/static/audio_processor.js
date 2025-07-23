/**
 * VORTA Ultra High-Grade Audio Processor
 * Browser-based audio recording - NO PyAudio needed!
 */

class VortaAudioProcessor {
  constructor() {
    this.mediaRecorder = null;
    this.audioContext = null;
    this.stream = null;
    this.websocket = null;
    this.isRecording = false;
  }

  async initialize() {
    try {
      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000, // Whisper optimal rate
        },
      });

      // Create audio context for real-time processing
      this.audioContext = new AudioContext({ sampleRate: 16000 });

      // Setup MediaRecorder for streaming
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      // Setup WebSocket connection
      this.websocket = new WebSocket(`ws://localhost:8080/ws/voice`);

      this.setupEventHandlers();
      return true;
    } catch (error) {
      console.error('❌ Audio initialization failed:', error);
      return false;
    }
  }

  setupEventHandlers() {
    // Handle audio data chunks
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0 && this.websocket.readyState === WebSocket.OPEN) {
        // Send audio chunk to backend for STT processing
        this.websocket.send(event.data);
      }
    };

    // Handle WebSocket messages (transcriptions)
    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'transcription') {
        this.displayTranscription(data.text);
      } else if (data.type === 'ai_response') {
        this.handleAIResponse(data.response);
      }
    };
  }

  startRecording() {
    if (this.mediaRecorder && !this.isRecording) {
      this.mediaRecorder.start(100); // Send chunks every 100ms
      this.isRecording = true;
      console.log('🎤 Recording started');
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      console.log('⏹️ Recording stopped');
    }
  }

  displayTranscription(text) {
    const transcriptionDiv = document.getElementById('transcription-display');
    transcriptionDiv.textContent = text;
  }

  async handleAIResponse(response) {
    // Display AI response
    const responseDiv = document.getElementById('ai-response');
    responseDiv.textContent = response;

    // Convert to speech and play
    await this.playTTSAudio(response);
  }

  async playTTSAudio(text) {
    try {
      const response = await fetch('/api/v1/voice/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text }),
      });

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      console.error('❌ TTS playback failed:', error);
    }
  }
}

// Initialize audio processor
const vortaAudio = new VortaAudioProcessor();
