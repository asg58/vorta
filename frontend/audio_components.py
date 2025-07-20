"""
🎤 VORTA Audio Components
Advanced audio processing utilities for the Streamlit dashboard
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def create_audio_waveform(audio_data: bytes, sample_rate: int = 16000):
    """
    Create an audio waveform visualization
    
    Args:
        audio_data: Raw audio bytes
        sample_rate: Audio sample rate
        
    Returns:
        Plotly figure object
    """
    try:
        # Convert bytes to numpy array
        if isinstance(audio_data, bytes):
            # Assume 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = np.array(audio_data)
        
        # Normalize audio data
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Create time axis
        duration = len(audio_array) / sample_rate
        time_axis = np.linspace(0, duration, len(audio_array))
        
        # Create waveform plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=audio_array,
            mode='lines',
            name='Waveform',
            line=dict(color='#667eea', width=1),
            hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🌊 Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=300,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waveform: {e}")
        return None

def create_audio_spectrogram(audio_data: bytes, sample_rate: int = 16000):
    """
    Create an audio spectrogram visualization
    
    Args:
        audio_data: Raw audio bytes
        sample_rate: Audio sample rate
        
    Returns:
        Plotly figure object
    """
    try:
        # Convert bytes to numpy array
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
        else:
            audio_array = np.array(audio_data)
        
        # Compute spectrogram using FFT
        window_size = 1024
        hop_size = 512
        
        # Create overlapping windows
        num_frames = (len(audio_array) - window_size) // hop_size + 1
        spectrogram = np.zeros((window_size // 2, num_frames))
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + window_size
            if end <= len(audio_array):
                window = audio_array[start:end] * np.hanning(window_size)
                fft = np.fft.fft(window)
                spectrogram[:, i] = np.abs(fft[:window_size // 2])
        
        # Convert to dB scale
        spectrogram = 20 * np.log10(spectrogram + 1e-10)
        
        # Create time and frequency axes
        duration = len(audio_array) / sample_rate
        time_axis = np.linspace(0, duration, num_frames)
        freq_axis = np.linspace(0, sample_rate / 2, window_size // 2)
        
        # Create spectrogram plot
        fig = go.Figure(data=go.Heatmap(
            x=time_axis,
            y=freq_axis,
            z=spectrogram,
            colorscale='Viridis',
            colorbar=dict(title="Magnitude (dB)")
        ))
        
        fig.update_layout(
            title="🎵 Audio Spectrogram",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating spectrogram: {e}")
        return None

def audio_player_component(audio_data: bytes, format: str = "mp3"):
    """
    Enhanced audio player component
    
    Args:
        audio_data: Audio data bytes
        format: Audio format (mp3, wav, etc.)
    """
    try:
        if audio_data:
            st.audio(audio_data, format=f'audio/{format}')
            
            # Show audio info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("File Size", f"{len(audio_data) / 1024:.1f} KB")
            
            with col2:
                st.metric("Format", format.upper())
            
            with col3:
                # Estimate duration (rough calculation)
                if format.lower() == 'mp3':
                    # MP3 rough estimation: ~1KB per second for typical bitrates
                    duration = len(audio_data) / 1024
                else:
                    duration = "Unknown"
                
                st.metric("Duration", f"~{duration:.1f}s" if isinstance(duration, float) else duration)
                
    except Exception as e:
        st.error(f"Error playing audio: {e}")

def create_voice_comparison_chart():
    """Create a comparison chart for different TTS voices"""
    
    # Sample voice characteristics data
    voices_data = {
        'Voice': ['Alloy', 'Echo', 'Fable', 'Nova', 'Onyx', 'Shimmer'],
        'Naturalness': [8.5, 8.0, 9.0, 8.7, 8.3, 9.2],
        'Clarity': [9.0, 8.5, 8.8, 9.1, 8.7, 8.9],
        'Speed': [8.0, 7.5, 8.2, 8.8, 8.1, 8.4],
        'Expressiveness': [7.8, 8.8, 9.5, 8.5, 9.0, 8.2]
    }
    
    # Create radar chart
    fig = go.Figure()
    
    categories = ['Naturalness', 'Clarity', 'Speed', 'Expressiveness']
    
    # Add trace for each voice
    colors = px.colors.qualitative.Set3
    
    for i, voice in enumerate(voices_data['Voice']):
        values = [voices_data[cat][i] for cat in categories]
        values.append(values[0])  # Close the polygon
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=voice,
            fillcolor=colors[i % len(colors)],
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="🗣️ Voice Quality Comparison",
        height=500,
        showlegend=True
    )
    
    return fig

def create_performance_gauge(value: float, title: str, max_value: float = 100):
    """
    Create a performance gauge chart
    
    Args:
        value: Current value
        title: Gauge title
        max_value: Maximum value for the gauge
    """
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': max_value * 0.8},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "lightgray"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_real_time_metrics():
    """Create real-time metrics dashboard"""
    
    # Generate sample real-time data
    import random
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        response_time = random.uniform(1.5, 3.5)
        fig = create_performance_gauge(response_time, "⚡ Response Time (s)", 5.0)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        accuracy = random.uniform(92, 98)
        fig = create_performance_gauge(accuracy, "🎯 Accuracy (%)", 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        cpu_usage = random.uniform(15, 45)
        fig = create_performance_gauge(cpu_usage, "💻 CPU Usage (%)", 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        memory_usage = random.uniform(200, 800)
        fig = create_performance_gauge(memory_usage, "🧠 Memory (MB)", 1000)
        st.plotly_chart(fig, use_container_width=True)

def create_usage_heatmap():
    """Create usage pattern heatmap"""
    
    # Generate sample usage data (24 hours x 7 days)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    # Generate realistic usage patterns
    usage_data = []
    for day_idx, day in enumerate(days):
        day_data = []
        for hour in hours:
            # Higher usage during work hours (9-17) on weekdays
            if day_idx < 5 and 9 <= hour <= 17:
                base_usage = np.random.normal(15, 5)
            elif 19 <= hour <= 22:  # Evening usage
                base_usage = np.random.normal(8, 3)
            else:  # Low usage during night/early morning
                base_usage = np.random.normal(2, 1)
            
            day_data.append(max(0, base_usage))
        usage_data.append(day_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=usage_data,
        x=hours,
        y=days,
        colorscale='Blues',
        colorbar=dict(title="API Calls"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="📅 Usage Patterns (API Calls per Hour)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig

def create_language_distribution():
    """Create language usage distribution chart"""
    
    languages = ['English', 'Dutch', 'German', 'French', 'Spanish', 'Italian']
    usage_counts = [45, 25, 12, 8, 6, 4]
    
    fig = px.bar(
        x=languages,
        y=usage_counts,
        title="🌍 Language Usage Distribution",
        color=usage_counts,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Language",
        yaxis_title="Usage Count",
        showlegend=False,
        height=400
    )
    
    return fig
