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

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted/frames'
RESULTS_FOLDER = 'results'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key='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'
    
)

# Global processing queue
processing_queue = Queue()
processing_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames_simple(video_path, output_folder, frame_rate=0.5):
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
                            "text": "Analyze the emotions on the face of the person in video. Return JSON with emotion scores 0-10 for: joy, sadness, anger, fear, surprise, disgust. Format: {\"emotions\": {\"joy\": 5, \"sadness\": 2}, \"description\": \"what you see\"}"
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

def process_video_simple(video_id, video_path):
    """Simplified video processing - frames only"""
    try:
        # Create unique folder for this video
        video_frames_folder = os.path.join(FRAMES_FOLDER, video_id)
        os.makedirs(video_frames_folder, exist_ok=True)
        
        # Extract frames
        print(f"Processing video {video_id}: Extracting frames...")
        frames = extract_frames_simple(video_path, video_frames_folder)
        
        # Process each frame
        results = []
        for frame_data in frames:
            timestamp = frame_data['timestamp']
            
            print(f"Analyzing frame at {timestamp:.2f}s...")
            frame_emotion = analyze_frame_emotion(frame_data['frame_path'])
            
            segment_result = {
                'timestamp': timestamp,
                'frame_data': frame_data,
                'frame_emotion': frame_emotion
            }
            
            results.append(segment_result)
        
        # Save results
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
        process_video_simple(video_id, video_path)
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