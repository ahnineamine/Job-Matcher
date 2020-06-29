# Job-Matcher
Job Matching using document embedding and similarity

## Intro and backgroud :
this project invloves the decision of whether a job seeker is relevent to a job vacancy, given the profile required for the job.

Achieving job matching that serves the expectation of job seekers and recruiters has been a challenging task in recruitment industry. 
This is due to limitations in job seeker and vacancy data that pose difficulties in processing and understanding 
job seekers, vacancies, or matching process (Furtmueller et al., 2011).

## Problem Statement :
The process of picking and filtering candidates for a job vacancy, can be time and effort consuming due to the overwhelming
number of applicants and job offers. Furthermore, HR managers tend to overlook applicants that are in fact more fitting for the vacancy, due to lack of deep knowledge in the sector of application. All of this can cause tremendous loss for the company, in terms of time and money.

statistics showed that companies suffer from severe employment mismatch, and end up firing a good portion of applicants just few months after hiring them. The latter induces a high level of unemployment and a large set of unfilled job vacancies.

## Goals and objectives
Artificial Intelligence is heavily influencing interviewing and assessment of candidates, not only 
resulting in better recruitment automation but also making the hiring process more error free.
I have chosen to build an intelligent solution, which has the ability to automatically recommend the most suitable CVs for 
a company's job offer.
this solution will optimize the work of the HR department. it examines more content, efficiently and in much less time, presenting 
only the most fitting applicants for the vacancy.  

!! I have chosen the IT department as context for my project. The project is only relevent for IT job matching ( for now ) !!

## Technologies and tools :
To build this AI-based application, I have picked a set of efficient natural language processing (NLP) and deep learning 
technologies due to their wide use in the state-of-the-art NLP softwares.   
The tools and requirments : ( details in requirements.txt )
- Programming language: Python
- Python Libraries : Numpy,Scipy
- NLP Library: Gensim
- Neural network model: Word2vec
- API server: Flask 
- Cloud Deployment: Heroku

## Project steps and Code :
### train_word2vec.ipynb :
 - Data Collection from indeed.com ( over 50 000 CVs )
 - Data preocessing using gensim
 - Word2vec model training.
### DocSim.py :
 - Inference using TF-IDF weighted vectors
 - Matching & document similarity calculation using TS-SS method (details in research paper attached to this repo)
### app.py :
 - implementation of Flask api 
### deployment of git repo :
 - deployment of api using heroku
## API endpoint :
here is the endpoint In order to be able to consume the API:  
https://job-matcher.herokuapp.com/processjson
## how to use :
the format of json file to be sent to the API is the following:   
![input](https://user-images.githubusercontent.com/38895133/57663354-32b10a00-75e3-11e9-9424-7e0503ca7142.PNG)   
    
the format of json file that will be sent back from the API is the following:   
![output](https://user-images.githubusercontent.com/38895133/57663356-32b10a00-75e3-11e9-8c88-d5fc6b13a98c.PNG)   
   
the score goes from 0 to "infinity", 0 being completly similar !   
the resumes will be order from the most to the least adequat.



## References :
- A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering (Arash Heidarian & Michael J.Dinneen)
- Bidirectional Job Matching through Unsupervised Feature Learning ( M.Sc. Sisay Adugna Chala, Universität Siegen )

