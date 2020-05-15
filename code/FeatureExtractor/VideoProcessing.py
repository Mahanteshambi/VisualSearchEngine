# import libraries
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from scipy import spatial
import matplotlib.pyplot as plt

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
video_capture = cv2.VideoCapture('/content/gdrive/My Drive/FaceProcessing/video/Modi_trump.mp4')
detector = MTCNN()
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print('Total frames: ' + str(length))

#codec = int(video_capture.get(cv2.CAP_PROP_FOURCC))
codec = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_movie = cv2.VideoWriter('/content/gdrive/My Drive/FaceProcessing/video/Modi_trump_processed.mp4', codec, fps, (frame_width,frame_height))

def getFaceEmbedding(image):
    rects = detector.detect_faces(image)
    x1, y1, width, height = rects[0]['box']
    x2, y2 = x1 + width, y1 + height
    if y1 < 0 or y2 >= image.shape[0] or x1 < 0 or x2 >= image.shape[1]:
        print(str((x1, y1, x2, y2)) + ' is beyond image of size: ' + str(image.shape))
        if x1 < 0:
            x1 = max(x1, 0)
        if y1 < 0:
            y1 = max(y1, 0)
        if x2 >= image.shape[1]:
            x2 = min(x2, image.shape[1])
        if y2 >= image.shape[0]:
            y2 = min(y2, image.shape[0])

    face = image[y1:y2, x1:x2]
    face = Image.fromarray(face)
    face = face.resize((224, 224))
    face = np.asarray(face, 'float32')
    face = utils.preprocess_input(face, version=2)
    face = np.expand_dims(face, axis=0)
    modi_face_embeddings = model.predict(face)
    return modi_face_embeddings

modi_image_path = '/content/gdrive/My Drive/FaceProcessing/processed_square_images/000270.jpg'
image = plt.imread(modi_image_path)
modi_embedding = getFaceEmbedding(image)

frame_count = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Quit when the input video file ends
    frame_count += 1
    
    rects = detector.detect_faces(rgb_frame)
    face_names = []
    for rect in rects:
        x1, y1, width, height = rect['box']
        x2, y2 = x1 + width, y1 + height
        
        embedding = getFaceEmbedding(rgb_frame)
        distance = spatial.distance.cosine(modi_embedding, embedding)
        #print(distance)
        if distance < 0.5:
			# Draw a box around the face
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face_names.append('NaMo')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    		# Draw a label with a name below the face
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'NaMo', (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
        

    """for i, rect in enumerate(rects):
        if face_names[i] is None:
            continue
        x1, y1, width, height = rect['box']
        x2, y2 = x1 + width, y1 + height
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		# Draw a label with a name below the face
        cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face_names[i], (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)"""
				
        
    # Display the resulting image
    #cv2.imshow('Video', frame)
	# Write the resulting image to the output video file
	
    if frame_count % 10 == 0:
        print("Writing frame {} / {}".format(frame_count, length))

    output_movie.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
output_movie.release()
cv2.destroyAllWindows()