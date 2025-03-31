
# Face Recognition Using MTCNN, VGG-Face2 and Pinecone DB

An implementation of siamease neural networks on one shot learning tasks for face recognition tasks utilising MTCNN, FaceNet and Pinecone DB for building an interactive and easy to use application to store and detect faces from images as well as camera inputs accurately.


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
