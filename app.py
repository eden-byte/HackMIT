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
import soundfile as sf
import shutil

#Import the processing modules
import audio_processor
import image_processor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted/frames'
RESULTS_FOLDER = 'results'
SAMPLE_RATE_SECONDS = 2
AUDIO_CLIP_DURATION_SECONDS = 3
AUDIO_SAMPLE_RATE = 44100 # The sample rate to work with

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}


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
            model="claude-3-5-haiku-20241022",
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
                            "text": "Analyze the emotions on the face of the person in this image. Also factor into the analyzation the text from the audio sample. Return JSON with emotion scores 0-10 for: joy, sadness, anger, fear, surprise, disgust, neutral. Format: {\"emotions\": {\"joy\": 5, \"sadness\": 2}, \"description\": \"what you see\"}"
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
<html>
<head>
    <title>Video Emotion Analysis</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .drop-zone { border: 3px dashed #007bff; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .drop-zone:hover { background: #f8f9fa; }
        .file-list { margin-top: 20px; }
        .file-item { padding: 10px; background: #f8f9fa; margin: 5px 0; border-radius: 5px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .status { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .status-processing { background: #fff3cd; border: 1px solid #ffeaa7; }
        .status-completed { background: #d4edda; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Video Emotion Analysis</h1>
        <p>Upload videos to analyze emotions from visual cues</p>
        
        <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
            <h3>📁 Click to select videos</h3>
            <p>Supports: MP4, AVI, MOV, MKV, WMV</p>
        </div>
        
        <input type="file" id="fileInput" multiple accept="video/*" style="display: none;">
        <button id="uploadBtn" class="btn" style="display: none;">Upload Videos</button>
        
        <div id="fileList" class="file-list"></div>
        <div id="status" class="status" style="display: none;"></div>
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
                div.innerHTML = `📹 ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
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
            statusDiv.innerHTML = '<h3>🔄 Processing videos...</h3>';
            
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
                    statusDiv.innerHTML += '<br><button class="btn" onclick="viewResults()">📈 View Results</button>';
                }
            }, 2000);
        }

        function viewResults() {
            window.open('/all_results', '_blank');
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

if __name__ == '__main__':
    print("🚀 Starting Video Emotion Analysis Server...")
    print("📁 Make sure to set ANTHROPIC_API_KEY in your environment")
    print("🌐 Server will run at: http://localhost:5000")
    app.run(debug=True, port=5000)