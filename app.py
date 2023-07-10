import cv2
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import werkzeug
from flask import Flask, request, render_template, send_from_directory
import os

# AU descriptions
au_descriptions = {
    '01': 'Inner Brow Raiser',
    '02': 'Outer Brow Raiser',
    '04': 'Brow Lowerer',
    '05': 'Upper Lid Raiser',
    '06': 'Cheek Raiser',
    '07': 'Lid Tightener',
    '09': 'Nose Wrinkler',
    '10': 'Upper Lip Raiser',
    '12': 'Lip Corner Puller',
    '14': 'Dimpler',
    '15': 'Lip Corner Depressor',
    '17': 'Chin Raiser',
    '20': 'Lip Stretcher',
    '23': 'Lip Tightener',
    '25': 'Lips part',
    '26': 'Jaw Drop',
    '45': 'Blink'
}

def process_files(video_path, csv_path):
    # Load the csv data
    data = pd.read_csv(csv_path)

    # Extract columns related to AU_r
    au_r_columns = [col for col in data.columns if ' AU' in col and '_r' in col]

    # Set timestamp as the index
    data.set_index(' timestamp', inplace=True)

    # Calculate the sum of AU responses in 0.2 second windows
    window_size = int(1 * 30)  # Frame rate is 30 fps
    au_sum = data[au_r_columns].rolling(window_size).sum()

    # Find the top 5 AUs for each frame
    top_5_au = au_sum.apply(lambda x: x.nlargest(5).index.tolist(), axis=1)

    # Remove the leading space and '_r' from the AU names for readability
    top_5_au = top_5_au.apply(lambda x: [au.strip()[2:-2] for au in x])

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))  # Get the width of frames
    frame_height = int(cap.get(4))  # Get the height of frames
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width,frame_height))

    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Get the top 5 AUs for the current frame
            aus = top_5_au.iloc[frame_count]
            descriptions = [f'AU{au}: {au_descriptions.get(au, "")}' for au in aus]

            # Add subtitles to the frame
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            font = ImageFont.truetype('arial.ttf', 30)
            for i, description in enumerate(descriptions):
                draw.text((50, 50 + i * 35), description, font=font, fill='red')

            # Convert the image with subtitles to OpenCV format and write it to file
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            out.write(frame)
            
            frame_count += 1
        else:
            break

    cap.release()
    out.release()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        csv_file = request.files['csv']

        # Save the uploaded files
        video_filename = werkzeug.utils.secure_filename(video_file.filename)
        csv_filename = werkzeug.utils.secure_filename(csv_file.filename)
        video_file.save(os.path.join('uploads', video_filename))
        csv_file.save(os.path.join('uploads', csv_filename))

        # Process the files and generate the output video
        process_files(os.path.join('uploads', video_filename), os.path.join('uploads', csv_filename))
        
        return render_template('download.html')
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)