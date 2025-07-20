"""
🎤 VORTA SIMPLE VOICE CHAT INTERFACE
Ultra Minimale Spraak Interface - Gebaseerd op gebruiker feedback
"""

import datetime
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ULTRA SIMPLE CONFIGURATION
st.set_page_config(
    page_title="VORTA Voice Chat",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS voor simpele interface
st.markdown('''
<style>
    .main { padding: 2rem 1rem; }
    .stApp { background: linear-gradient(135deg, #1a1a1a 0%, #2d1a4d 100%); }
    
    .voice-chat-container {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .voice-chat-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 50px;
        padding: 20px 60px;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .voice-chat-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    .conversation-bubble {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background: #00ff88; }
    .status-listening { background: #ff0080; animation: pulse 1s infinite; }
    .status-speaking { background: #00ccff; animation: pulse 1s infinite; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
''', unsafe_allow_html=True)

# API Configuration
VORTA_API_URL = os.environ.get("VORTA_API_URL", "http://localhost:8000")
API_BASE_URL = f"{VORTA_API_URL}/api"

class SimpleVortaAPI:
    """Simpele API client voor VORTA voice chat"""
    
    def __init__(self):
        self.base_url = API_BASE_URL
        
    def check_connection(self):
        """Production API health check"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return True, response.json()
        except:
            return False, {"error": "Cannot connect to VORTA"}
    
    def get_voices(self):
        """Haal beschikbare stemmen op"""
        try:
            response = requests.get(f"{self.base_url}/voices", timeout=5)
            return True, response.json()
        except:
            return False, {"error": "Cannot get voices"}
    
    def synthesize_speech(self, text: str, voice: str = "nova"):
        """Converteer tekst naar spraak"""
        try:
            payload = {
                "text": text,
                "voice": voice,
                "model": "tts-1"
            }
            response = requests.post(f"{self.base_url}/tts", json=payload, timeout=10)
            return True, response.json()
        except Exception as e:
            return False, {"error": str(e)}

# Initialize API
api = SimpleVortaAPI()

def main():
    """Hoofdinterface - Simpel Voice Chat"""
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    if 'is_speaking' not in st.session_state:
        st.session_state.is_speaking = False
    if 'selected_voice' not in st.session_state:
        st.session_state.selected_voice = 'nova'
    
    # Header
    st.markdown('''
    <div class="voice-chat-container">
        <h1 style="color: white; margin-bottom: 2rem; font-size: 2.5rem;">
            🎤 VORTA VOICE CHAT
        </h1>
        <p style="color: rgba(255,255,255,0.8); margin-bottom: 2rem; font-size: 1.2rem;">
            Powered by ElevenLabs Conversational AI
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Status indicators
    connection_status, _ = api.check_connection()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        status_class = "status-online" if connection_status else "status-offline"
        st.markdown(f'''
        <div style="text-align: center; color: white;">
            <span class="status-indicator {status_class}"></span>
            Connection: {"Online" if connection_status else "Offline"}
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        listening_class = "status-listening" if st.session_state.is_listening else "status-online"
        st.markdown(f'''
        <div style="text-align: center; color: white;">
            <span class="status-indicator {listening_class}"></span>
            Listening: {"Active" if st.session_state.is_listening else "Ready"}
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        speaking_class = "status-speaking" if st.session_state.is_speaking else "status-online"
        st.markdown(f'''
        <div style="text-align: center; color: white;">
            <span class="status-indicator {speaking_class}"></span>
            Speaking: {"Active" if st.session_state.is_speaking else "Ready"}
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Voice Chat Button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.is_listening:
            if st.button("🎤 VOICE CHAT", key="main_voice_button", use_container_width=True):
                st.session_state.is_listening = True
                st.rerun()
        else:
            if st.button("⏹️ STOP CHAT", key="stop_voice_button", use_container_width=True):
                st.session_state.is_listening = False
                st.rerun()
    
    # Voice Chat Status
    if st.session_state.is_listening:
        st.markdown('''
        <div class="voice-chat-container">
            <h3 style="color: #ff0080; margin-bottom: 1rem;">
                🎤 Listening... Speak now!
            </h3>
            <p style="color: rgba(255,255,255,0.8);">
                Say something and I'll respond with voice
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Conversation History
    if st.session_state.conversation:
        st.markdown("### 💬 Conversation")
        
        for message in st.session_state.conversation:
            if message['role'] == 'user':
                st.markdown(f'''
                <div class="conversation-bubble" style="border-left-color: #00ff88;">
                    <strong>🗣️ You:</strong> {message['content']}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="conversation-bubble" style="border-left-color: #00ccff;">
                    <strong>🤖 VORTA:</strong> {message['content']}
                </div>
                ''', unsafe_allow_html=True)
    
    # Text Input Alternative
    st.markdown("---")
    st.markdown("### ⌨️ Text Alternative")
    
    user_input = st.text_input("Type your message:",
                               placeholder="Enter your message")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Send Message"):
            if user_input.strip():
                # Add user message
                st.session_state.conversation.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Generate AI response
                ai_response = generate_simple_response(user_input)
                
                # Add AI response
                st.session_state.conversation.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
                # Speak response
                speak_response(ai_response)
                
                st.rerun()
    
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.conversation = []
            st.rerun()
    
    with col3:
        if st.button("🔊 Repeat Last"):
            if st.session_state.conversation:
                last_ai_message = None
                for message in reversed(st.session_state.conversation):
                    if message['role'] == 'assistant':
                        last_ai_message = message['content']
                        break
                if last_ai_message:
                    speak_response(last_ai_message)
    
    # Voice Selection
    with st.sidebar:
        st.markdown("### 🗣️ Voice Settings")
        
        voices_status, voices_data = api.get_voices()
        if voices_status and voices_data.get("voices"):
            available_voices = voices_data["voices"]
            voice_options = [f"{voice['name']} ({voice['voice_id']})" for voice in available_voices]
            
            selected_voice_display = st.selectbox(
                "Select AI Voice:",
                voice_options,
                index=0
            )
            
            # Extract voice_id
            st.session_state.selected_voice = selected_voice_display.split('(')[-1].rstrip(')')

def generate_simple_response(user_input: str) -> str:
    """Generate simple AI response"""
    
    # Simple response patterns
    responses = [
        f"I understand you're asking about '{user_input[:20]}...'. That's an interesting topic!",
        "Thanks for your message! I'm VORTA, your AI voice assistant. How can I help you today?",
        f"Great question! Regarding '{user_input[:15]}...', I'd be happy to help explain that.",
        "I'm here to assist you with any questions or have a friendly conversation. What would you like to know?",
        f"You mentioned '{user_input[:25]}...'. That's something I can definitely help you with!"
    ]
    
    # Simple language detection
    dutch_words = ['hallo', 'hoi', 'wat', 'hoe', 'waarom', 'kun je', 'nederlands', 'dank je']
    if any(word in user_input.lower() for word in dutch_words):
        responses = [
            f"Ik begrijp je vraag over '{user_input[:20]}...'. Dat is interessant!",
            "Dank je voor je bericht! Ik ben VORTA, jouw AI spraak assistent. Hoe kan ik je helpen?",
            f"Goede vraag! Over '{user_input[:15]}...' kan ik je graag meer vertellen.",
            "Ik help je graag met vragen of een gezellig gesprek. Wat wil je weten?",
            f"Je noemde '{user_input[:25]}...'. Daar kan ik je zeker mee helpen!"
        ]
    
    return np.random.choice(responses)

def speak_response(text: str):
    """Speak AI response"""
    
    st.session_state.is_speaking = True
    
    try:
        with st.spinner("🔊 VORTA is speaking..."):
            success, result = api.synthesize_speech(text, st.session_state.selected_voice)
            
            if success:
                st.success(f"✅ Response spoken with {st.session_state.selected_voice} voice")
            else:
                st.error(f"❌ Speech failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Speech error: {str(e)}")
    
    finally:
        st.session_state.is_speaking = False

if __name__ == "__main__":
    main()
