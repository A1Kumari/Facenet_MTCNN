import cv2
from pinecone import Pinecone
import os
from dotenv import load_dotenv , dotenv_values
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import scipy # for MTCNN Dependencies
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()
load_dotenv()
pc = Pinecone(os.getenv("pinecone_key"))
index = pc.Index(os.getenv("pinecone_index"))

def get_embedding(file):
  img = plt.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  try:
    faces = detector.detect_faces(img)
    if len(faces) > 1:
      return print("Multiple Faces")
    else:
      faces = faces[0]
      x1, y1, width, height = faces['box']
      x1, y1 = abs(x1), abs(y1)
      img = img[y1:y1+height, x1:x1+width]
      img = cv2.resize(img, (224, 224))
      img = np.expand_dims(img, axis=0)
      print(f"Detected Face in {file}")
      return embedder.embeddings(img)[0]
  except:
    print(f"Failed to detect faces in {file}")
    return None
  
def make_embeddings(file , name):
  embeddings = {}
  emb = get_embedding(file)
  if emb is not None:
    embeddings[name] = emb
  return embeddings

def make_meta(file , i , name):
  meta = {'image_id':f"Image_{i}"}
  meta['label'] = name
  return meta

def upsert_embeddings(face_embeddings , metadata):
  assert len(face_embeddings) == len(metadata) , f"Dimesnion mismatch, got embeddings len {len(face_embeddings)} and metadata length {len(metadata)}"
  assert face_embeddings is not None and metadata is not None , 'Input vectors cannot be None'
  assert face_embeddings[0].shape[0] == index.describe_index_stats()['dimension'] , 'Dimension mismatch'
  a = index.describe_index_stats()['total_vector_count'] + 1
  upsert_data = [(str(i + a), face_embeddings[i].tolist(), metadata[i]) for i in range(len(face_embeddings))] # index starts from last vector in database
  index.upsert(vectors=upsert_data)
  return True

def fetch_embeddings(embeddings):
  assert embeddings.shape[0] == index.describe_index_stats()['dimension'] , f"Expected{embeddings.shape[0]} got {index.describe_index_stats()['dimension']} instead"
  assert embeddings is not None , 'Input vectors cannot be None'

  out = index.query(
      vector=embeddings.tolist(),
      top_k=1,
      include_metadata=True
  )
  name , confidence = out['matches'][0]['metadata']['label'] , out['matches'][0]['score']
  return name , confidence

def multiple_faces(file):
  imga = plt.imread(file)
  #imga = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
  try:
    faces = detector.detect_faces(imga)
    embs = []
    boxes = []
    for face in faces:
      x1, y1, width, height = face['box']
      x1, y1 = abs(x1), abs(y1)
      img = imga[y1:y1+height, x1:x1+width]
      img = cv2.resize(img, (224, 224))
      img = np.expand_dims(img, axis=0)
      embs.append(embedder.embeddings(img)[0])
      boxes.append([x1, y1, width, height])
    return embs, boxes
  except Exception as e:
      print(f"Failed to detect faces in {file} due to {e}")
  return None

def draw_boxes(file, boxes ,labels, probs):
  img = plt.imread(file)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.copy(img)
  for box , label , prob in zip(boxes,labels, probs):
    x1, y1, width, height = box
    img = cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 1)
    if label == 'Unknown':
      img = cv2.putText(img, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    else:
      img = cv2.putText(img, f"{label} {np.round(prob , decimals = 2)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def write_and_upsert(dir ,  Name, upsert = True):
  embeddings = make_embeddings(dir , Name)
  metadata = [make_meta(dir , 1 , Name)]
  if upsert:
    upsert_embeddings(list(embeddings.values()), metadata)
  return True

def detect_and_fetch(dir , directory = False , min_confidence = 0):
  files = []
  files.append(dir)
  for file in files:
    try:
      embs , boxes = multiple_faces(file)
      labels = []
      probs = []
      for emb in embs:
        name , confidence = fetch_embeddings(emb)
        if confidence > min_confidence:
          labels.append(name)
          probs.append(confidence)
        else:
          labels.append('Unknown')
          probs.append('unk')
      img = draw_boxes(file, boxes, labels, probs)
    except Exception as e:
      print(f"Failed to detect faces in {file} due to {e}")
  return img , labels 


