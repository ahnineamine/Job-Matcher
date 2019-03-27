# Job-Matcher
Job Matching using document embedding and similarity

## Intro and backgroud :
this project invloves the decision of whether a job seeker is relevent to a job vacancy, given the profile needed for the job.

Achieving job matching that serves the expectation of job seekers and recruiters has been a challenging task in recruitment industry. 
This is due to limitations in job seeker and vacancy data that pose difficulties in processing and understanding 
job seekers, vacancies, or matching process (Furtmueller et al., 2011).

## Problem Statement :
The job of choosing the most fitting candidates for a certain job, can be time and effort consuming due to the overwhelming
number of applicants. Furthermore, HR managers may overlook some applicants that can actually be more fitting for a certain job, because of 
lack of deep knowledge in the sector of application. All of this can cause tremendous loss for the entreprise, in terms of time and money.

statistics showed that companies suffer from employment missmatch, and they end firing a good portion of the applicants just few months 
after hiring. which cause a high level of unemployment and at the same time alot of unfilled job vacancies.

## Goals and objectives
Artificial Intelligence, like everything else, are also heavily influencing interviewing and assessment of candidates, not only 
resulting in better recruitment automation but also making the hiring process more error free.
I have chosen to build an intelligent solution, which has the ability to automatically recommend the most suitable CVs for 
a company's job offers.
this solution will optimize the work of the HR department. it examines more content, efficiently and in much less time, presenting 
only the most fitting applicants for the vacancy.
!! I have chosen the IT department as context for my project. The project is only relevent for IT job matching ( for now ) !!

## Technologies and tools :
To build this AI-based application, I have chosen to adapt recent and efficient natural language processing (NLP) and deep learning 
technologies for their wide use in the state-of-the-art NLP softwares. 
The tools and requirments : ( details in requirements.txt )
- Programming language: Python
- Python Libraries : Numpy,Scipy
- NLP Library: Gensim
- Neural network model: Word2vec
- MicroFramework: Flask 
- Deployment: Heroku

## Project steps and Code :
train_word2vec.ipynb :
 1- Data Collection for indeed.com ( over 50 000 CVs )
 2- Data preocessing using gensim
 3- using gensim's word2vec implementation.
DocSim.py :
 4- Inference using TF-IDF weighted vectors
 5- Matching and recommandation using TS-SS method
app.py :
 6- implementation of Flask api ( input json + output json )
deployment of git repo :
 7- deployment of api using heroku

## References :
A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering ( Arash Heidarian & Michael J. Dinneen )
Bidirectional Job Matching through Unsupervised Feature Learning ( M.Sc. Sisay Adugna Chala, Universität Siegen )

