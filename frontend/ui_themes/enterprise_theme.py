"""
VORTA Enterprise UI Theme System
Professional styling and theme management for VORTA AI Platform

Author: High-Grade Development Team
Version: 2.0.0-enterprise
"""

from typing import Any, Dict

import streamlit as st


class VortaEnterpriseTheme:
    """
    Enterprise-grade UI theme system with professional styling
    
    Features:
    - Consistent brand identity
    - Professional color schemes
    - Responsive design patterns
    - Accessibility compliance
    - High-contrast readability
    """
    
    # ENTERPRISE COLOR PALETTE
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'accent': '#00ff88',
        'warning': '#ffaa00',
        'error': '#ff4444',
        'success': '#00ff88',
        'info': '#00ccff',
        'background_dark': '#0c0c0c',
        'background_medium': '#1a1a1a',
        'background_light': '#2d2d2d',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#888888',
        'border_color': '#404040'
    }
    
    @staticmethod
    def apply_enterprise_styling():
        """Apply comprehensive enterprise styling to Streamlit interface"""
        st.markdown(f"""
        <style>
        /* ENTERPRISE GLOBAL THEME */
        .stApp {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['background_dark']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['background_medium']} 100%);
            color: {VortaEnterpriseTheme.COLORS['text_primary']};
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }}
        
        /* ENTERPRISE HEADER STYLING */
        .enterprise-header {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['primary']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['secondary']} 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        .enterprise-header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .enterprise-header p {{
            font-size: 1.2rem;
            margin: 0.5rem 0;
            opacity: 0.9;
        }}
        
        /* PROFESSIONAL METRIC CARDS */
        .metric-card {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['background_medium']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['background_light']} 100%);
            border: 2px solid {VortaEnterpriseTheme.COLORS['primary']};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: {VortaEnterpriseTheme.COLORS['accent']};
        }}
        
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 700;
            color: {VortaEnterpriseTheme.COLORS['accent']};
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            font-size: 1rem;
            color: {VortaEnterpriseTheme.COLORS['text_secondary']};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* STATUS INDICATORS */
        .status-online {{
            color: {VortaEnterpriseTheme.COLORS['success']};
            font-weight: 600;
        }}
        
        .status-offline {{
            color: {VortaEnterpriseTheme.COLORS['error']};
            font-weight: 600;
        }}
        
        .status-warning {{
            color: {VortaEnterpriseTheme.COLORS['warning']};
            font-weight: 600;
        }}
        
        /* VOICE CHAT INTERFACE */
        .voice-chat-container {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            margin: 2rem auto;
            max-width: 700px;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        }}
        
        .voice-chat-button {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['primary']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['secondary']} 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 1rem 2rem;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .voice-chat-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }}
        
        /* CONVERSATION DISPLAY */
        .conversation-message {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid {VortaEnterpriseTheme.COLORS['primary']};
        }}
        
        .conversation-message.user {{
            background: rgba(102, 126, 234, 0.2);
            border-left-color: {VortaEnterpriseTheme.COLORS['accent']};
        }}
        
        .conversation-message.assistant {{
            background: rgba(118, 75, 162, 0.2);
            border-left-color: {VortaEnterpriseTheme.COLORS['info']};
        }}
        
        /* PROFESSIONAL BUTTONS */
        .stButton > button {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['primary']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['secondary']} 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        /* SIDEBAR STYLING */
        .css-1d391kg {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['background_dark']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['background_medium']} 100%);
        }}
        
        /* PROFESSIONAL TABS */
        .stTabs [data-baseweb="tab-list"] {{
            background: {VortaEnterpriseTheme.COLORS['background_light']};
            border-radius: 10px;
            padding: 0.25rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            color: {VortaEnterpriseTheme.COLORS['text_secondary']};
            border-radius: 8px;
            font-weight: 500;
            padding: 1rem 1.5rem;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {VortaEnterpriseTheme.COLORS['primary']} 0%, 
                                                {VortaEnterpriseTheme.COLORS['secondary']} 100%);
            color: white;
        }}
        
        /* LOADING INDICATORS */
        .stProgress .st-bo {{
            background: linear-gradient(90deg, {VortaEnterpriseTheme.COLORS['primary']} 0%, 
                                               {VortaEnterpriseTheme.COLORS['secondary']} 50%, 
                                               {VortaEnterpriseTheme.COLORS['accent']} 100%);
        }}
        
        /* RESPONSIVE DESIGN */
        @media (max-width: 768px) {{
            .enterprise-header h1 {{ font-size: 2rem; }}
            .voice-chat-container {{ padding: 2rem; }}
            .metric-card {{ padding: 1rem; }}
        }}
        
        /* ACCESSIBILITY IMPROVEMENTS */
        *:focus {{
            outline: 2px solid {VortaEnterpriseTheme.COLORS['accent']};
            outline-offset: 2px;
        }}
        
        /* HIGH CONTRAST TEXT */
        .high-contrast {{
            color: {VortaEnterpriseTheme.COLORS['text_primary']};
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_enterprise_header(title: str, subtitle: str = "", description: str = ""):
        """Render professional enterprise header"""
        st.markdown(f"""
        <div class="enterprise-header">
            <h1>{title}</h1>
            {f'<p style="font-size: 1.3rem; font-weight: 600;">{subtitle}</p>' if subtitle else ''}
            {f'<p style="font-size: 1rem; opacity: 0.8;">{description}</p>' if description else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_card(label: str, value: str, status: str = "neutral"):
        """Render professional metric card"""
        status_class = f"status-{status}" if status in ['online', 'offline', 'warning'] else ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {status_class}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_conversation_message(role: str, content: str):
        """Render professional conversation message"""
        role_display = "👤 You" if role == "user" else "🤖 VORTA"
        message_class = f"conversation-message {role}"
        
        st.markdown(f"""
        <div class="{message_class}">
            <strong>{role_display}:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_voice_chat_interface():
        """Render professional voice chat interface container"""
        st.markdown("""
        <div class="voice-chat-container">
            <h2 style="color: #667eea; margin-bottom: 1rem;">🎤 VORTA Voice Interface</h2>
            <p style="color: #cccccc; font-size: 1.1rem;">
                Professional AI Voice Assistant • Enterprise Ready • Ultra Responsive
            </p>
        </div>
        """, unsafe_allow_html=True)


class VortaComponentLibrary:
    """Professional component library for VORTA interface elements"""
    
    @staticmethod
    def professional_sidebar(api_client):
        """Render professional sidebar with enterprise styling"""
        with st.sidebar:
            VortaEnterpriseTheme.render_enterprise_header(
                "⚡ MISSION CONTROL",
                "Enterprise Command Interface"
            )
            
            # Connection status
            st.markdown("### 🔗 CONNECTION STATUS")
            health_response = api_client.health_check()
            
            if health_response.success:
                VortaEnterpriseTheme.render_metric_card(
                    "SYSTEM STATUS", "🟢 ONLINE", "online"
                )
                
                if health_response.response_time_ms:
                    VortaEnterpriseTheme.render_metric_card(
                        "RESPONSE TIME", f"{health_response.response_time_ms:.1f}ms", "online"
                    )
            else:
                VortaEnterpriseTheme.render_metric_card(
                    "SYSTEM STATUS", "🔴 OFFLINE", "offline"
                )
            
            # Performance metrics
            st.markdown("### 📊 PERFORMANCE")
            metrics = api_client.get_performance_metrics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Requests", metrics['total_requests'])
                st.metric("Success Rate", f"{metrics['successful_requests']}/{metrics['total_requests']}")
            
            with col2:
                st.metric("Avg Response", f"{metrics['average_response_time']:.1f}ms")
                st.metric("Failed Requests", metrics['failed_requests'])
    
    @staticmethod
    def professional_voice_interface(api_client):
        """Render professional voice interface"""
        VortaEnterpriseTheme.render_voice_chat_interface()
        
        # Voice selection
        voices_response = api_client.get_voice_profiles()
        
        if voices_response.success and 'voice_profiles' in voices_response.data:
            voice_options = []
            for profile in voices_response.data['voice_profiles']:
                quality_indicator = "💎" if profile.is_premium else "⭐"
                display_name = f"{quality_indicator} {profile.display_name}"
                voice_options.append((display_name, profile.voice_id))
            
            if voice_options:
                selected_display = st.selectbox(
                    "🗣️ Select Voice Profile:",
                    [display for display, _ in voice_options],
                    help="Choose your preferred AI voice for synthesis"
                )
                
                # Find selected voice_id
                selected_voice_id = next(
                    (voice_id for display, voice_id in voice_options if display == selected_display),
                    "alloy"
                )
                
                return selected_voice_id
        
        return "alloy"
