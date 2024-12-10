
# Face Recognition Using MTCNN, VGG-Face2 and Pinecone DB

An implementation of siamease neural networks on one shot learning tasks for face recognition tasks utilising MTCNN, FaceNet and Pinecone DB for building an interactive and easy to use application to store and detect faces from images as well as camera inputs accurately.


## Deployment

To use this project application run

```bash
  https://face-recognition-tf-1.onrender.com
```
Note : Loading may take some time due to limited  server processing capability.



## Outputs
<img width="956" alt="356581473-3f98e226-d9a5-4036-9c35-06481f292c8a" src="https://github.com/user-attachments/assets/60c7ea7f-1604-421b-8265-5b171da0edaa">
<img width="956" alt="image" src="https://github.com/user-attachments/assets/1bc6a9c0-2698-483c-b478-80bf8d29f442">


## Key Features

__1)__ __Face detection using MTCNN__ :     
Used MTCNN for accurate face detection on images.          

__2)__ __Embedding extraction using FacNet__ :  
created embeddings vectors of shape (512,1) using a Tensorflow pretrained model on face datasets.  

__3)__  __Pinecone Database for efficent storage and retreival__:   
Created a pinecone DB index for storing embdeeings by creating relevent metadata and upserting it to the index with cosine similarity as serach parameter.    

__4)__ __Siamease Network Architehture__:   
Implemented a Siamease Network like architehture to acheive one shot learning for face recognition using a combination of face detection and face recognition.  

__5)__ __Streamlit application__:   
Created and deployed an interactive streamlit application to interact with the project. Deployment was done on render.  

__6)__ __Multiface Detection capabilities__:    
The model is able to detect and recognise multiple faces in an image however for creating a new entry in the database for a person, an induidual image is required to ensure integrity of the data in the database. 

__7)__ __Modularity of code__:  
The project is cretated so that the induvidual blocks can be changed to suit the detection needs, eg. MTCNN can be replace with YOLO detection for faster results for applications such as ANPR after replacing FaceNet with a suitably trained model to generate embeddings.   


## Pinecone API Reference

##### Refer to Pinecone Documentation at : [pc docs](https://docs.pinecone.io/reference/api/introduction)

#### Pinecone env variables used :

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `Pinecone key` | `string` | <Your API key>|
| `Pinecone index`      | `string` | Name of Index |


#### Installing pinecone client
```bash
pip install pinecone --upgrade pinecone-client
```
#### Connecting to index and upserting

```bash
from pinecone import Pinecone
pc = Pinecone(os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_IDX"])
```
```bash
vectors = 'Create Vectors'
index.upsert(vectors)
```

#### Query top k vectors
```bash
out = index.query(
      vector = vectors.tolist(),
      top_k = k,
      include_metadata = True
  )
```


## Dependencies
- OpenCV
- Tensorflow
- MTCNN
- FaceNet
- Pinecone
- streamlit
- os
- dotenv
- numpy
#### Installation :
After pulling this repo, run:
```bash
pip install requirements
```
#### env file setup :
setup an env file to store api keys and index information as the template :
```bash
pinecone_key = "your_api_key"
pinecone_index = "your_index_name"
```


