
import cv2
import numpy as np
from moviepy.editor import *
import os

def anonymize_face_pixelate(frame, face_x, face_w, face_y, face_h, blocks=10):
    image = frame[face_y:face_y+face_h, face_x:face_x+face_w]
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)

    frame[face_y:face_y+face_h, face_x:face_x+face_w] = image
    return frame

def apply_faces_to_video(timestamps, local_path_to_video, local_output, video_metadata, color=(255,0,0), thickness=2):
    # Extract video info
    frame_rate = video_metadata["FrameRate"]
    frame_height = video_metadata["FrameHeight"]
    frame_width = video_metadata["FrameWidth"]
    # Set up support for OpenCV
    frame_counter = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Create the file pointers
    v = cv2.VideoCapture(local_path_to_video)
    out = cv2.VideoWriter(
        filename=local_output, 
        fourcc=fourcc, 
        fps=int(frame_rate), 
        frameSize=(frame_width, frame_height)
    )
    # Open the video
    while v.isOpened():
        # Get frames until available
        has_frame, frame = v.read()
        if has_frame:
            for t in timestamps:
                faces = timestamps.get(t)
                lower_bound = int(int(t)/1000*int(frame_rate))
                upper_bound = int(int(t)/1000*int(frame_rate))+(int(frame_rate)/2)
                if (frame_counter >= lower_bound) and (frame_counter <= upper_bound):
                    for f in faces:
                        x = int(f['Left']*frame_width)
                        y = int(f['Top']*frame_height)
                        w = int(f['Width']*frame_width)
                        h = int(f['Height']*frame_height)
                        #frame = cv2.rectangle(frame, (x,y), (x+w,y+h), color, thickness)
                        frame = anonymize_face_pixelate(frame, x, w, y, h, 50)
            out.write(frame)
            frame_counter += 1
        else:
            break

    out.release()
    v.release()
    cv2.destroyAllWindows()
    print(f"Complete. {frame_counter} frames were written.")

def integrate_audio(original_video, output_video, audio_path='/tmp/audio.mp3'):
    # Extract audio
    my_clip = VideoFileClip(original_video)
    my_clip.audio.write_audiofile(audio_path)

    # Join output video with extracted audio
    videoclip = VideoFileClip(output_video)
    audioclip = AudioFileClip(audio_path)
    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile(output_video)

    # Delete audio
    os.remove(audio_path)

    print('Complete')
