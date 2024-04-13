
from flask import Flask, request, jsonify,render_template,Response,g
import json
import spacy
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
import tempfile
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
from gradientai import Gradient
from gradientai import ExtractParamsSchemaValueType, Gradient
from dotenv import dotenv_values
from hugchat import hugchat
from hugchat.login import Login


app = Flask(__name__)

CORS(app)
gradient = Gradient(access_token="eVRUTHaz9a3lo2UgfPQEVJh5xmeFQXjq",workspace_id="85645f4c-5ba3-47c0-9798-ad329732435b_workspace")
COOKIE_FILE = 'cookies.json'
Q_FILE = 'question.json'
A_FILE = 'answer.json'

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_similarity(model, source_text, target_texts):
    # Tokenize source and target texts
    source_input = tokenizer(source_text, padding=True, truncation=True, return_tensors='pt')
    target_inputs = tokenizer(target_texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        source_output = model(**source_input)
        target_outputs = model(**target_inputs)

    # Perform pooling for source and target embeddings
    source_embedding = mean_pooling(source_output, source_input['attention_mask'])
    target_embeddings = mean_pooling(target_outputs, target_inputs['attention_mask'])

    # Calculate cosine similarity using sentence-transformers
    cosine_scores = torch.nn.functional.cosine_similarity(source_embedding, target_embeddings).cpu().numpy()

    return cosine_scores


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def preprocess_text(text_to_clean):
    # Remove URLs
    cleaned_text = re.sub('http\S+\s*', ' ', text_to_clean)
    # Remove RT and cc
    cleaned_text = re.sub('RT|cc', ' ', cleaned_text)
    # Remove hashtags
    cleaned_text = re.sub('#\S+', '', cleaned_text)
    # Remove mentions
    cleaned_text = re.sub('@\S+', ' ', cleaned_text)
    # Remove punctuations
    cleaned_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleaned_text)
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7f]', r' ', cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub('\s+', ' ', cleaned_text)
    # Remove leading and trailing whitespaces
    cleaned_text = cleaned_text.strip()

    return cleaned_text




def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()

    # Remove extra spaces and new lines
    text = ' '.join(text.split())
    
    return text

def jobClassification(resume):
    
        input_resume = preprocess_text(resume)
        # Load the saved model
        with open('./RankingAssets/nb_classifier_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('./RankingAssets/word_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

        new_data_vectorized = loaded_vectorizer.transform([input_resume])

        # Make predictions on the vectorized new data using the loaded model
        new_predictions = loaded_model.predict(new_data_vectorized)

        # The variable 'new_predictions' now contains the model predictions for the new data
        df = pd.read_csv('./RankingAssets/cv_data.csv')
        label = LabelEncoder()
        df['Category_but_Encoded'] = label.fit_transform(df['Category'])
        predicted_label = label.inverse_transform(new_predictions)

        return predicted_label[0]

def extract_questions(response):
    questions = []
    lines = response.split('\n')
    
    for line in lines:
        # Strip whitespace and ignore empty lines
        line = line.strip()
        if line.startswith(('\n', '\r', '\r\n', '-')):
            continue
        
        # Check if the line starts with a number followed by a dot (e.g., "1.", "2.")
        if line.lstrip().startswith(tuple(f"{i}." for i in range(1, 11))):
            question = line.split('.', 1)[1].strip()  # Extract the question after the number
            questions.append(question)
    
    return questions

    
@app.route('/')
def hello_world():
    secrets = dotenv_values('hf.env')
    hf_email = secrets['EMAIL']
    hf_pass = secrets['PASS']
    sign = Login(hf_email, hf_pass)
    cookies = sign.login()
    # Write cookies to file (assuming COOKIE_FILE is defined)
    with open(COOKIE_FILE, 'w') as f:
        json.dump(cookies.get_dict(), f)
    return 


@app.route('/question_generation/<string:domain>', methods=['GET'])
def question_generation(domain):
    try:
        # Read cookies from file
        with open(COOKIE_FILE, 'r') as f:
            cookies = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Cookies not found. Please log in first."}), 401
    try:
        # Create ChatBot using stored cookies
        chatbot = hugchat.ChatBot(cookies=cookies)
        # Generate interview questions for the specified domain
        questions_query = f"Create {domain} interview questions"
        response = chatbot.chat(questions_query)
        # Return the generated questions in a JSON response format
        return jsonify({"questions": extract_questions(str(response))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/answer_generation',methods=['POST'])
def answer_generation():

    try:
        with open(COOKIE_FILE, 'r') as f:
            cookies = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Cookies not found. Please log in first."}), 401
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies)

    try:
        with open(Q_FILE, 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "questions not found. Please generate questions in first."}), 401
    
    questions = json.dumps(questions)

    answers = "Give answers for the following questions " + questions + " and do not write anything except answers especially Sure, I can help you create some data science interview questions! Here are a few examples: or questions themselves "
    response2 = chatbot.chat(answers)
    response_answers_multiline = '"""' + response2 + '"""'
    response_anslist = response_answers_multiline.strip().split('\n')
    with open(A_FILE, 'w') as f:
        json.dump(response_anslist, f)
    
    result = {
        "response": response_anslist
    }
    return jsonify(result)

@app.route('/answer_comparison', methods=['POST'])
def answer_comparison():
    # Get user input from request body
    user_answer = request.json.get("user_answer")
    question = request.json.get("question")

    try:
        # Read cookies from file
        with open(COOKIE_FILE, 'r') as f:
            cookies = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Cookies not found. Please log in first."}), 401

    # Create ChatBot using stored cookies
    chatbot = hugchat.ChatBot(cookies=cookies)

    # Construct the evaluation prompt dynamically
    evaluation_prompt = f"Evaluate the correctness of the following answer for the given question:\n\nQuestion: {question}\nAnswer: {user_answer}\n\nReturn the percentage value of correctness for interview assessment. only return me the fix value. and give feedback how he/she can improve there answer "

    # Use the chatbot to evaluate the correctness percentage
    response = chatbot.chat(evaluation_prompt)
    print(response)
    # Parse the response and format the result
    evaluation_score = response
    result = {"response": str(response)}
    return result


@app.route('/resumeParser', methods=['POST'])
def parse_resume():
    try:
        if 'resume_pdf' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        resume_pdf = request.files['resume_pdf']

        if resume_pdf.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if resume_pdf and resume_pdf.filename.endswith('.pdf'):
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                file_path = tmp_file.name
                resume_pdf.save(file_path)

            # Extract text from the PDF file
            resume_text = extract_text_from_pdf(file_path)

            # Define schema for resume data extraction
            schema_ = {
                "Name": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Email": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Skills": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Education": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Description": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Experience": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
            }

            # Perform resume data extraction
            result = gradient.extract(
                document=resume_text,
                schema_=schema_,
            )

            # Further processing or classification of extracted data
            domain = jobClassification(resume_text)

            return jsonify({
                'Domain': domain,
                "ParsedData":result
            })

        else:
            return jsonify({'errors': 'Uploaded file is not a PDF'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/jobDescriptionParser', methods=['POST'])
def parse_jd():
    try:
        if 'job_pdf' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        job_pdf = request.files['job_pdf']

        if job_pdf.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if job_pdf and job_pdf.filename.endswith('.pdf'):
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                file_path = tmp_file.name
                job_pdf.save(file_path)

            # Extract text from the PDF file
            job_text = extract_text_from_pdf(file_path)

            # Define schema for job description data extraction
            schema_ = {
                "company": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Job Title": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Skills": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Education": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Responsibilities": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
                "Experience": {
                    "type": ExtractParamsSchemaValueType.STRING,
                    "required": True,
                },
            }

            # Perform job description data extraction
            result = gradient.extract(
                document=job_text,
                schema_=schema_,
            )

            return jsonify({
                'ParsedData': result
            })

        else:
            return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/jobClassifier', methods=['POST'])
def job_classifier():
    try:
        # Check if 'resume_pdf' is in request files
        if 'resume_pdf' not in request.files:
            return jsonify({'error': 'No resume file part in the request'}), 400
        resume_pdf = request.files['resume_pdf']
        # Check if a resume file was selected
        if resume_pdf.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Check if the uploaded file is a PDF
        if resume_pdf.filename.endswith('.pdf'):
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                file_path = tmp_file.name
                resume_pdf.save(file_path)
            # Extract text from the PDF resume
            resume_text = extract_text_from_pdf(file_path)
            # Preprocess the extracted resume text
            input_resume = preprocess_text(resume_text)
            # Load the saved model and vectorizer
            with open('./RankingAssets/nb_classifier_model.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            with open('./RankingAssets/word_vectorizer.pkl', 'rb') as vectorizer_file:
                loaded_vectorizer = pickle.load(vectorizer_file)
            # Vectorize the input resume text
            new_data_vectorized = loaded_vectorizer.transform([input_resume])
            # Make predictions using the loaded model
            new_predictions = loaded_model.predict(new_data_vectorized)
            # Decode predicted label using label encoder
            df = pd.read_csv('./RankingAssets/cv_data.csv')
            label_encoder = LabelEncoder()
            df['Category_but_Encoded'] = label_encoder.fit_transform(df['Category'])
            predicted_label = label_encoder.inverse_transform(new_predictions)
            # Return the predicted category
            return jsonify({'predictions': predicted_label[0]})
        else:
            return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/matchingscore_latest', methods=['POST'])
def jobMatchingScore_latest():
    try:
        # Extract data from the request JSON
        candidate_experience = request.json.get('candidate_experience')
        candidate_skills = request.json.get('candidate_skills')
        candidate_education = request.json.get('candidate_edu')
        candidate_description = request.json.get('candidate_desc')
        candidate_domain = request.json.get('candidate_domain')
        job_title = request.json.get('job_title')
        job_skills = request.json.get('job_skills')
        job_education = request.json.get('job_education')
        job_responsibilities = request.json.get('job_responsibilities')
        job_experience = request.json.get('job_experience')
        Domain_similarity_score = calculate_similarity(model , candidate_domain , job_title)
        Experience_similarity_score = calculate_similarity(model, candidate_experience, job_experience)
        Skills_similarity_score = calculate_similarity(model, candidate_skills, job_skills)
        Education_similarity_score = calculate_similarity(model, candidate_education, job_education)
        Description_similarity_score = calculate_similarity(model, candidate_description, job_responsibilities)
        avg_score = (Experience_similarity_score + Skills_similarity_score + Education_similarity_score + Description_similarity_score+Domain_similarity_score) / 5 * 100
        print(Experience_similarity_score,Skills_similarity_score,Education_similarity_score,Description_similarity_score)

        return jsonify({
            "experience_score": int(Experience_similarity_score[0]*100),
            "skills_score": int(Skills_similarity_score[0]*100),
            "education_score": int(Education_similarity_score[0]*100),
            "description_score": int(Description_similarity_score[0]*100),
            "domain_score": int(Domain_similarity_score[0]*100),
            "avg_score": int(avg_score[0])
        })

    except Exception as error:
        # Return JSON response in case of an exception
        return jsonify({"error": str(error)}), 500

   
   

# main driver function
if __name__ == '__main__':

	app.run()
