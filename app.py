from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import cv2
import anthropic
from datetime import datetime
import json
import threading
from queue import Queue
import base64
import subprocess
import shutil
import soundfile as sf
import requests

#Import the processing modules
import audio_processor
import image_processor

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted/frames'
RESULTS_FOLDER = 'results'
SAMPLE_RATE_SECONDS = 2
AUDIO_CLIP_DURATION_SECONDS = 3
AUDIO_SAMPLE_RATE = 44100 # The sample rate to work with
POKE_API_KEY = os.getenv('POKE_API_KEY', 'your-poke-api-key-here')
POKE_API_URL = 'https://poke.com/api/v1/inbound-sms/webhook'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')
)

# Global processing queue
processing_queue = Queue()
processing_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames_simple(video_path, output_folder, frame_rate=1):
    """Extract frames from video using OpenCV only"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = f"frame_{int(timestamp):04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append({
                'timestamp': timestamp,
                'frame_path': frame_path,
                'frame_filename': frame_filename
            })
        
        frame_count += 1
    
    cap.release()
    return extracted_frames

def format_timestamp(seconds):
    """Converts seconds into HH-MM-SS format for filenames."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}-{m:02d}-{s:02d}"

def extract_full_audio_track(video_path, temp_audio_path):
    """
    Uses ffmpeg to extract the full audio track to a temporary WAV file.
    This is much more robust than library-based in-memory extraction.
    Returns True on success, False on failure.
    """
    print("Extracting full audio track with ffmpeg...")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',            # No video
        '-acodec', 'pcm_s16le', # Use standard WAV codec
        '-ar', str(AUDIO_SAMPLE_RATE), # Set audio sample rate
        '-ac', '1',       # Set to mono
        '-y',             # Overwrite output file if it exists
        temp_audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Audio extraction successful.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg command failed. Is ffmpeg installed and in your system's PATH?")
        return False

def analyze_frame_emotion(frame_path):
    """Analyze emotions in a frame using Anthropic API"""
    try:
        with open(frame_path, 'rb') as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        message = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast model with high rate limits
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze the emotions in this image. Return JSON with emotion scores 0-10 for: joy, sadness, anger, fear, surprise, disgust, neutral. Format: {\"emotions\": {\"joy\": 5, \"sadness\": 2}, \"description\": \"what you see\"}"
                        }
                    ]
                }
            ]
        )
        
        response_text = message.content[0].text
        return {"visual_analysis": response_text}
        
    except Exception as e:
        print(f"Error analyzing frame emotion: {e}")
        return {"error": str(e)}
    
def send_to_poke(message):
    """Send a message to Poke API"""
    try:
        response = requests.post(
            POKE_API_URL,
            headers={
                'Authorization': f'Bearer {POKE_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={'message': message}
        )
        return response.json()
    except Exception as e:
        print(f"Error sending to Poke: {e}")
        return {"error": str(e)}
    
def analyze_video_feedback(video_data):
    """Analyze all visual analysis data and provide constructive feedback"""
    try:
        # Extract all visual analyses
        visual_analyses = []
        emotion_scores = []
        
        for result in video_data.get('results', []):
            if 'frame_emotion' in result and 'visual_analysis' in result['frame_emotion']:
                visual_analyses.append(result['frame_emotion']['visual_analysis'])
                # Try to extract emotion scores if in JSON format
                try:
                    emotion_data = json.loads(result['frame_emotion']['visual_analysis'])
                    if 'emotions' in emotion_data:
                        emotion_scores.append(emotion_data['emotions'])
                except:
                    pass
        
        # Create comprehensive analysis prompt
        prompt = f"""
        Based on this video emotion analysis data, provide constructive feedback about the person's presentation and emotional expression:

        Visual Analysis Data: {visual_analyses}

        Please provide:
        1. POSITIVE ASPECTS: What the person does well emotionally (genuine smiles, engaging expressions, confident body language, etc.)
        2. CONSTRUCTIVE FEEDBACK: Areas for improvement in emotional expression and presentation (maintaining consistent energy, reducing nervous expressions, etc.)
        3. OVERALL IMPRESSION: Summary of their emotional presence and charisma
        4. ACTIONABLE TIPS: Specific suggestions for improving their on-camera presence

        Keep feedback supportive and encouraging while being honest about areas for growth. Focus on presentation skills and emotional communication.
        """
        
        message = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        feedback = message.content[0].text
        
        # Send feedback to Poke
        poke_message = f"Video Analysis Feedback:\n\n{feedback}"
        poke_response = send_to_poke(poke_message)
        
        return {
            "feedback": feedback,
            "poke_response": poke_response,
            "total_frames_analyzed": len(visual_analyses)
        }
        
    except Exception as e:
        print(f"Error analyzing video feedback: {e}")
        return {"error": str(e)}

def process_video_enhanced(video_id, video_path):
    """Adapted from your teammate's process_local_video function"""
    try:
        # Create unique folder for this video (similar to your teammate's setup)
        video_output_dir = os.path.join(RESULTS_FOLDER, video_id)
        image_output_dir = os.path.join(video_output_dir, "processed_images")
        audio_output_dir = os.path.join(video_output_dir, "processed_audio")
        temp_dir = os.path.join(video_output_dir, "temp")
        
        # Clean up if exists (your teammate's approach)
        if os.path.exists(video_output_dir):
            print(f"Output directory {video_output_dir} already exists. Removing it.")
            shutil.rmtree(video_output_dir)
        
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(audio_output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Extract full audio track (your teammate's method)
        temp_audio_path = os.path.join(temp_dir, "full_audio.wav")
        audio_success = extract_full_audio_track(video_path, temp_audio_path)

        # Set up for sampling (your teammate's approach)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        results = []
        last_image_hash = None
        
        # Main sampling loop (adapted from your teammate's code)
        for t in range(0, int(duration), SAMPLE_RATE_SECONDS):
            timestamp_str = format_timestamp(t)
            print(f"\n--- Processing sample at {t}s ({timestamp_str}) ---")
            
            segment_result = {
                'timestamp': t,
                'timestamp_str': timestamp_str
            }
            
            # A. Process a Single Video Frame (your teammate's approach)
            frame_id = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if ret:
                image_data = image_processor.process_image_frame_from_memory(frame)
                if image_data:
                    current_hash = image_data['features']['perceptual_hash']
                    if current_hash == last_image_hash:
                        print("Skipping redundant image frame.")
                        segment_result['frame_emotion'] = {"skipped": "duplicate_frame"}
                    else:
                        last_image_hash = current_hash
                        image_savename = f"frame_{timestamp_str}.jpg"
                        image_savepath = os.path.join(image_output_dir, image_savename)
                        cv2.imwrite(image_savepath, image_data['processed_image'])
                        
                        # Use your existing emotion analysis
                        frame_emotion = analyze_frame_emotion(image_savepath)
                        segment_result['frame_emotion'] = frame_emotion
                        segment_result['image_features'] = image_data['features']
                        
                        print(f"Saved processed image: {image_savename}")

            # B. Process Audio Clip (your teammate's approach)
            if audio_success:
                try:
                    start_sample = int(t * AUDIO_SAMPLE_RATE)
                    end_sample = int((t + AUDIO_CLIP_DURATION_SECONDS) * AUDIO_SAMPLE_RATE)
                    
                    # Read the sample directly from the temporary WAV file
                    audio_array, _ = sf.read(temp_audio_path, start=start_sample, stop=end_sample, dtype='float32')

                    audio_data = audio_processor.process_audio_clip_from_numpy(audio_array, AUDIO_SAMPLE_RATE)
                    
                    if audio_data:
                        audio_savename = f"clip_{timestamp_str}.wav"
                        audio_savepath = os.path.join(audio_output_dir, audio_savename)
                        
                        sf.write(audio_savepath, audio_data['processed_audio_array'], audio_data['sample_rate'])
                        
                        segment_result['audio_features'] = audio_data['features']
                        print(f"Saved processed audio clip: {audio_savename}")

                except Exception as e:
                    print(f"Could not process audio clip at {t}s: {e}")
            
            results.append(segment_result)

        # Clean up and finalize (your teammate's approach)
        cap.release()
        shutil.rmtree(temp_dir) # Remove temporary audio file and folder
        
        # Save results (your existing format)
        results_file = os.path.join(RESULTS_FOLDER, f"{video_id}_analysis.json")
        with open(results_file, 'w') as f:
            json.dump({
                'video_id': video_id,
                'processed_at': datetime.now().isoformat(),
                'total_segments': len(results),
                'results': results
            }, f, indent=2)
        
        processing_results[video_id] = {
            'status': 'completed',
            'results_file': results_file,
            'total_segments': len(results)
        }
        
        print(f"Video {video_id} processing completed!")
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        processing_results[video_id] = {
            'status': 'error',
            'error': str(e)
        }


def worker():
    """Background worker to process videos"""
    while True:
        video_id, video_path = processing_queue.get()
        processing_results[video_id] = {'status': 'processing'}
        process_video_enhanced(video_id, video_path)
        processing_queue.task_done()

# Start background worker
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# Simple HTML template
UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Video Emotion Analysis — ReLive!</title>
    <style>
        :root {
            --bg-primary: #FFFAF2;
            --border-color: #b1b0fe;
            --window-bg: #c19acb;
            --cream: #FFFDD0;
            --light-grey: #e8e8e8;
            --gradient-pink-green: linear-gradient(to right, #f56ebd, #97f0b6);
            --gradient-red-lime: linear-gradient(to right, #fd5a47, #c0e264);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; font-family: 'Nunito', sans-serif; font-weight: 550; }
        
        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, #f8f4e9 100%);
            color: #f5f5f5;
            overflow-x: hidden;
        }
        
        header.appbar {
            height: 80px;
            display: flex; align-items: center; justify-content: center;
            padding: 0 1rem;
            border-bottom: 2px solid var(--border-color);
            background: var(--light-grey);
            position: fixed; left: 0; right: 0; top: 0; z-index: 1000;
        }
        
        header .app-title {
            font-weight: 700;
            color: #333;
            font-size: 3rem;
            text-align: center;
        }
        
        .main-container {
            margin-top: 56px;
            padding: 2rem;
            min-height: calc(100vh - 56px);
            display: flex;
            justify-content: center;
            align-items: stretch;
        }
        
        .window {
            width: 100%;
            height: calc(100vh - 56px - 4rem);
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.15);
            border-radius: 1rem;
            border: 1px solid var(--border-color);
            background-color: var(--window-bg);
            backdrop-filter: blur(8px);
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
        }
        
        .window:hover { 
            box-shadow: 0 15px 35px -5px rgba(0,0,0,0.2); 
        }
        
        .title-bar {
            display: flex; align-items: center; gap: .5rem;
            text-align: center;
            padding: .75rem 1rem;
            border-top-left-radius: 1rem;
            border-top-right-radius: 1rem;
            background: var(--gradient-pink-green);
            color: white;
            flex-shrink: 0;
        }
        
        .win-dot {
            width: 0.6rem; height: 0.6rem; border-radius: 50%;
            background-color: rgba(255,255,255,0.9);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        .title-bar-text { 
            font-size: 1.2rem; font-weight: 700; flex-grow: 1; 
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; 
        }
        
        .title-bar-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .title-bar-controls button {
            font-size: 1rem; font-weight: bold;
            background-color: rgba(255,255,255,0.15);
            padding: 0.25rem 0.6rem;
            border-radius: 0.5rem; border: none; color: white; cursor: pointer; 
            transition: all 0.2s ease;
            min-width: 1.8rem;
            text-align: center;
        }
        
        .title-bar-controls button:hover { 
            background-color: rgba(255,255,255,0.25); 
        }
        
        .window-body {
            padding: 2rem;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #f5f5f5;
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        
        .window.minimized .window-body { 
            display: none !important; 
        }
        
        .window.minimized .title-bar { 
            opacity: 0.85; 
        }
        
        .window.minimized {
            height: auto;
        }
        
        .upload-section {
            text-align: center;
            margin-bottom: 2rem;
            flex-shrink: 0;
        }
        
        .upload-section h2 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            color: #f5f5f5;
        }
        
        .upload-section p {
            opacity: 0.9;
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
        }
        
        .drop-zone {
            border: 3px dashed rgba(255,255,255,0.4);
            padding: 14rem 14rem 14rem 14rem;
            border-radius: 1rem;
            margin: 1.5rem 0;
            background: rgba(255,255,255,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .drop-zone:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.6);
            transform: translateY(-2px);
        }
        
        .drop-zone h3 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            color: #f5f5f5;
        }
        
        .drop-zone p {
            opacity: 0.8;
            font-size: 1rem;
        }
        
        .btn {
            padding: 1rem 2rem;
            border-radius: 0.75rem;
            background: var(--gradient-red-lime);
            color: white; border: none; font-weight: 700;
            font-size: 1rem; cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
            font-family: 'Nunito', sans-serif;
            text-decoration: none; display: inline-block;
            letter-spacing: 0.5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .content-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            overflow-y: auto;
        }
        
        .file-list {
            flex-shrink: 0;
        }
        
        .file-item {
            padding: 1.2rem;
            background: rgba(255,255,255,0.1);
            margin: 0.5rem 0;
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.2s ease;
            font-size: 1rem;
        }
        
        .file-item:hover {
            background: rgba(255,255,255,0.15);
        }
        
        .status {
            padding: 2rem;
            border-radius: 1rem;
            border: 1px solid rgba(255,255,255,0.2);
            flex: 1;
            overflow-y: auto;
        }
        
        .status-processing {
            background: rgba(255, 243, 205, 0.2);
            border-color: rgba(255, 234, 167, 0.4);
        }
        
        .status-completed {
            background: rgba(212, 237, 218, 0.2);
            border-color: rgba(195, 230, 203, 0.4);
        }
        
        .status h3 {
            margin-bottom: 1rem;
            color: #f5f5f5;
            font-size: 1.3rem;
        }
        
        .button-group {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .feedback-window {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 900px;
            max-height: 80vh;
            background: var(--window-bg);
            border-radius: 1rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            z-index: 2000;
            overflow: hidden;
        }
        
        .feedback-content {
            padding: 2rem;
            max-height: calc(80vh - 4rem);
            overflow-y: auto;
            color: #f5f5f5;
        }
        
        .feedback-item {
            margin: 1.5rem 0;
            padding: 1.5rem;
            background: rgba(255,255,255,0.1);
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .feedback-item h4 {
            margin-bottom: 1rem;
            color: #f5f5f5;
        }
        
        .close-feedback {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
        }
        
        .close-feedback:hover {
            background: rgba(255,255,255,0.3);
        }
        
        /* Custom scrollbar */
        .window-body::-webkit-scrollbar,
        .content-area::-webkit-scrollbar,
        .status::-webkit-scrollbar,
        .feedback-content::-webkit-scrollbar {
            width: 8px;
        }
        
        .window-body::-webkit-scrollbar-track,
        .content-area::-webkit-scrollbar-track,
        .status::-webkit-scrollbar-track,
        .feedback-content::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }
        
        .window-body::-webkit-scrollbar-thumb,
        .content-area::-webkit-scrollbar-thumb,
        .status::-webkit-scrollbar-thumb,
        .feedback-content::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.3);
            border-radius: 4px;
        }
        
        .window-body::-webkit-scrollbar-thumb:hover,
        .content-area::-webkit-scrollbar-thumb:hover,
        .status::-webkit-scrollbar-thumb:hover,
        .feedback-content::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.4);
        }
    </style>
</head>
<body>
    <header class="appbar">
        <div class="app-title">ReLive!</div>
    </header>

    <div class="main-container">
        <div class="window">
            <div class="title-bar">
                <div class="win-dot"></div>
                <div class="title-bar-text">Video Emotion Analysis</div>
                <div class="title-bar-controls">
                    <button id="minimizeBtn" aria-label="Minimize">−</button>
                    <button id="maximizeBtn" aria-label="Maximize" style="display:none;">□</button>
                </div>
            </div>
            
            <div class="window-body">
                <div class="upload-section">
                    <h2>🎭 Analyze Video Emotions</h2>
                    <p>Upload videos to analyze emotions from visual cues and get AI-powered feedback</p>
                    
                    <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
                        <h3>📁 Click to select videos</h3>
                        <p>Supports: MP4, AVI, MOV, MKV, WMV, FLV, WEBM</p>
                    </div>
                    
                    <input type="file" id="fileInput" multiple accept="video/*" style="display: none;">
                    <button id="uploadBtn" class="btn" style="display: none;">🚀 Upload & Analyze Videos</button>
                </div>
                
                <div class="content-area">
                    <div id="fileList" class="file-list"></div>
                    <div id="status" class="status" style="display: none;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        let uploadedVideos = [];
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            selectedFiles = Array.from(e.target.files);
            displayFiles();
            document.getElementById('uploadBtn').style.display = selectedFiles.length > 0 ? 'block' : 'none';
        });

        function displayFiles() {
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';
            selectedFiles.forEach(file => {
                const div = document.createElement('div');
                div.className = 'file-item';
                div.innerHTML = `📹 ${file.name} <span style="opacity: 0.7;">(${(file.size/1024/1024).toFixed(1)} MB)</span>`;
                fileList.appendChild(div);
            });
        }

        document.getElementById('uploadBtn').addEventListener('click', async () => {
            const formData = new FormData();
            selectedFiles.forEach(file => formData.append('videos', file));
            
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                uploadedVideos = result.videos;
                
                selectedFiles = [];
                document.getElementById('fileList').innerHTML = '';
                document.getElementById('uploadBtn').style.display = 'none';
                
                startPolling();
            } catch (error) {
                alert('Upload failed: ' + error);
            }
        });

        function startPolling() {
            const statusDiv = document.getElementById('status');
            statusDiv.style.display = 'block';
            statusDiv.className = 'status status-processing';
            statusDiv.innerHTML = '<h3>🔄 Processing videos...</h3><p>Please wait while we analyze your videos for emotional content.</p>';
            
            const interval = setInterval(async () => {
                let allCompleted = true;
                let statusHTML = '<h3>📊 Processing Status:</h3>';
                
                for (const video of uploadedVideos) {
                    const response = await fetch(`/status/${video.video_id}`);
                    const status = await response.json();
                    
                    const statusText = status.status === 'completed' ? '✅ Completed' : 
                                     status.status === 'processing' ? '🔄 Processing...' : 
                                     status.status === 'error' ? '❌ Error' : '⏳ Queued';
                    
                    statusHTML += `<div class="file-item">${video.filename}: ${statusText}</div>`;
                    
                    if (status.status !== 'completed') {
                        allCompleted = false;
                    }
                }
                
                statusDiv.innerHTML = statusHTML;
                
                if (allCompleted) {
                    clearInterval(interval);
                    statusDiv.className = 'status status-completed';
                    statusDiv.innerHTML += '<div class="button-group"><button class="btn" onclick="getFeedback()">💬 Get AI Feedback via Poke</button><button class="btn" onclick="viewResults()">📈 View Raw Results</button></div>';
                }
            }, 2000);
        }

        function viewResults() {
            window.open('/all_results', '_blank');
        }

        async function getFeedback() {
            try {
                const response = await fetch('/analyze_feedback');
                const data = await response.json();
                
                // Create feedback window
                const feedbackWindow = document.createElement('div');
                feedbackWindow.className = 'feedback-window';
                
                let feedbackHtml = `
                    <div class="title-bar">
                        <div class="win-dot"></div>
                        <div class="title-bar-text">AI Feedback Analysis</div>
                    </div>
                    <button class="close-feedback" onclick="this.parentElement.remove()">×</button>
                    <div class="feedback-content">
                        <h3>🤖 AI Analysis Results</h3>
                        <p style="margin-bottom: 1.5rem; opacity: 0.9;">Based on ${data.total_videos_analyzed} video(s) analyzed</p>
                `;
                
                data.feedback_results.forEach((result, index) => {
                    feedbackHtml += `
                        <div class="feedback-item">
                            <h4>📹 Video ${index + 1} Analysis</h4>
                            <p><strong>Frames Analyzed:</strong> ${result.total_frames_analyzed}</p>
                            <div style="margin-top: 1rem; white-space: pre-wrap; line-height: 1.6;">${result.feedback}</div>
                            ${result.poke_response ? '<p style="margin-top: 1rem; font-style: italic; opacity: 0.8;">✅ Feedback sent to Poke successfully!</p>' : ''}
                        </div>
                    `;
                });
                
                feedbackHtml += '</div>';
                feedbackWindow.innerHTML = feedbackHtml;
                
                document.body.appendChild(feedbackWindow);
                
            } catch (error) {
                alert('Failed to get feedback analysis: ' + error);
            }
        }

        // Close feedback window when clicking outside
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('feedback-window')) {
                e.target.remove();
            }
        });

        // Window controls functionality
        const mainWindow = document.querySelector('.window');
        const minimizeBtn = document.getElementById('minimizeBtn');
        const maximizeBtn = document.getElementById('maximizeBtn');
        const titleBar = document.querySelector('.title-bar');
        
        if (mainWindow && minimizeBtn && maximizeBtn && titleBar) {
            function toggleMinimize() {
                const isNowMinimized = mainWindow.classList.toggle('minimized');
                if (isNowMinimized) {
                    minimizeBtn.style.display = 'none';
                    maximizeBtn.style.display = 'inline-block';
                } else {
                    minimizeBtn.style.display = 'inline-block';
                    maximizeBtn.style.display = 'none';
                }
            }
            
            minimizeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleMinimize();
            });
            
            maximizeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleMinimize();
            });
            
            // Double-click title bar to minimize/maximize
            titleBar.addEventListener('dblclick', toggleMinimize);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return UPLOAD_HTML

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'videos' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('videos')
    uploaded_videos = []
    
    for file in files:
        if file and file.filename != '' and allowed_file(file.filename):
            video_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            
            video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_{filename}")
            file.save(video_path)
            
            processing_queue.put((video_id, video_path))
            processing_results[video_id] = {'status': 'queued'}
            
            uploaded_videos.append({
                'video_id': video_id,
                'filename': filename,
                'status': 'queued'
            })
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_videos)} videos',
        'videos': uploaded_videos
    })

@app.route('/status/<video_id>')
def get_status(video_id):
    if video_id not in processing_results:
        return jsonify({'error': 'Video not found'}), 404
    return jsonify(processing_results[video_id])

@app.route('/results/<video_id>')
def get_results(video_id):
    if video_id not in processing_results:
        return jsonify({'error': 'Video not found'}), 404
    
    result = processing_results[video_id]
    if result['status'] != 'completed':
        return jsonify({'error': 'Video processing not completed'}), 400
    
    try:
        with open(result['results_file'], 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error loading results: {e}'}), 500

@app.route('/all_results')
def get_all_results():
    all_results = []
    for video_id, status_info in processing_results.items():
        if status_info['status'] == 'completed':
            try:
                with open(status_info['results_file'], 'r') as f:
                    data = json.load(f)
                all_results.append(data)
            except:
                continue
    
    return jsonify(all_results)

@app.route('/analyze_feedback')
def get_analysis_feedback():
    """Analyze all videos and provide constructive feedback via Poke"""
    all_results = []
    for video_id, status_info in processing_results.items():
        if status_info['status'] == 'completed':
            try:
                with open(status_info['results_file'], 'r') as f:
                    data = json.load(f)
                all_results.append(data)
            except:
                continue
    
    if not all_results:
        return jsonify({'error': 'No completed video analyses found'}), 404
    
    # Analyze each video and combine feedback
    all_feedback = []
    for video_data in all_results:
        feedback = analyze_video_feedback(video_data)
        feedback['video_id'] = video_data['video_id']
        all_feedback.append(feedback)
    
    return jsonify({
        'total_videos_analyzed': len(all_results),
        'feedback_results': all_feedback,
        'analysis_generated_at': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Video Emotion Analysis Server...")
    print("📁 Make sure to set ANTHROPIC_API_KEY in your environment")
    print("🌐 Server will run at: http://localhost:5000")
    app.run(debug=True, port=5000)