import os
import time
import requests
import base64
from flask import Flask, render_template, request, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load environment variables from .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        os.environ[parts[0].strip()] = parts[1].strip()
    except Exception as e:
        print("Error loading .env file:", e)

app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'truthshield-dev-secret-2026')

# ── Google OAuth via authlib ──────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', ''),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)
# ─────────────────────────────────────────────────────────────────────────────

# Load the full-text model and vectorizer
loaded_model_article = pickle.load(open("model.pkl", 'rb'))
vector_article = pickle.load(open("vector.pkl", 'rb'))

# Load the headline-specific model and vectorizer
loaded_model_headline = pickle.load(open("model_title.pkl", 'rb'))
vector_headline = pickle.load(open("vector_title.pkl", 'rb'))
# Set NLTK data directory to /tmp (Vercel has read-only filesystem except /tmp)
nltk_data_dir = os.path.join('/tmp', 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Ensure NLTK packages are downloaded
for package in ['stopwords', 'punkt', 'wordnet']:
    try:
        if package == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif package == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif package == 'wordnet':
            nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_dir, quiet=True)

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def gemini_fact_check(news, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are a professional fact-checker. Fact-check this news statement or claim: '{news}'\n\n"
        "Instructions:\n"
        "1. Determine whether the claim is TRUE, FALSE, or MISLEADING.\n"
        "2. Provide a clear 2-3 sentence explanation based on the most up-to-date factual evidence.\n"
        "3. Keep your output formatted exactly like this:\n"
        "VERDICT: [True / False / Misleading]\n"
        "EXPLANATION: [Your brief explanation here]"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "tools": [
            {"google_search": {}}
        ]
    }
    
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json['candidates'][0]['content']['parts'][0]['text']
                
                # Extract citations/sources
                sources = []
                try:
                    metadata = res_json['candidates'][0].get('groundingMetadata', {})
                    search_chunks = metadata.get('groundingChunks', [])
                    for chunk in search_chunks:
                        web = chunk.get('web', {})
                        title = web.get('title')
                        uri = web.get('uri')
                        if uri:
                            sources.append({"title": title or uri, "url": uri})
                except Exception:
                    pass
                
                verdict = "Unverified"
                explanation = text_response
                
                for line in text_response.split('\n'):
                    if line.upper().startswith("VERDICT:"):
                        verdict = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("EXPLANATION:"):
                        explanation = line.split(":", 1)[1].strip()
                        
                return {
                    "success": True,
                    "verdict": verdict,
                    "explanation": explanation,
                    "sources": sources[:5]
                }
            elif response.status_code in (500, 503, 429):
                # Transient server-side error — retry after delay
                last_error = f"API Error (Status {response.status_code}) — retrying..."
                time.sleep(2 * (attempt + 1))
                continue
            else:
                return {
                    "success": False,
                    "error": f"API Error (Status {response.status_code})"
                }
        except Exception as e:
            last_error = str(e)
            time.sleep(2 * (attempt + 1))
            continue
    return {
        "success": False,
        "error": f"API unavailable after 3 attempts. Please try again shortly. ({last_error})"
    }

def gemini_image_fact_check(image_bytes, mime_type, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = (
        "You are a professional fact-checker. Fact-check the news, claim, or context shown in this image.\n\n"
        "Instructions:\n"
        "1. Determine whether the claim or news depicted is TRUE, FALSE, or MISLEADING.\n"
        "2. Provide a clear 2-3 sentence explanation based on the most up-to-date factual evidence.\n"
        "3. Keep your output formatted exactly like this:\n"
        "VERDICT: [True / False / Misleading]\n"
        "EXPLANATION: [Your brief explanation here]"
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_image
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "tools": [
            {"google_search": {}}
        ]
    }
    
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json['candidates'][0]['content']['parts'][0]['text']
                
                sources = []
                try:
                    metadata = res_json['candidates'][0].get('groundingMetadata', {})
                    search_chunks = metadata.get('groundingChunks', [])
                    for chunk in search_chunks:
                        web = chunk.get('web', {})
                        title = web.get('title')
                        uri = web.get('uri')
                        if uri:
                            sources.append({"title": title or uri, "url": uri})
                except Exception:
                    pass
                
                verdict = "Unverified"
                explanation = text_response
                
                for line in text_response.split('\n'):
                    if line.upper().startswith("VERDICT:"):
                        verdict = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("EXPLANATION:"):
                        explanation = line.split(":", 1)[1].strip()
                        
                return {
                    "success": True,
                    "verdict": verdict,
                    "explanation": explanation,
                    "sources": sources[:5]
                }
            elif response.status_code in (500, 503, 429):
                last_error = f"API Error (Status {response.status_code}) — retrying..."
                time.sleep(2 * (attempt + 1))
                continue
            else:
                return {
                    "success": False,
                    "error": f"API Error (Status {response.status_code})"
                }
        except Exception as e:
            last_error = str(e)
            time.sleep(2 * (attempt + 1))
            continue
    return {
        "success": False,
        "error": f"API unavailable after 3 attempts. Please try again shortly. ({last_error})"
    }

def fake_news_det(news, is_headline=True):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review:
        if y not in stpwrds:
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    
    if is_headline:
        vectorized_input_data = vector_headline.transform(input_data)
        prediction = loaded_model_headline.predict(vectorized_input_data)
    else:
        vectorized_input_data = vector_article.transform(input_data)
        prediction = loaded_model_article.predict(vectorized_input_data)
     
    return prediction

# ── Auth helpers ─────────────────────────────────────────────────────────────
PUBLIC_ROUTES = {'login', 'google_login', 'google_callback', 'static'}

@app.before_request
def require_login():
    """Redirect unauthenticated users to /login for all protected routes."""
    if request.endpoint and request.endpoint not in PUBLIC_ROUTES:
        if 'user' not in session:
            return redirect(url_for('login'))

@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('home'))
    error = request.args.get('error')
    return render_template('login.html', error=error)

@app.route('/login/google')
def google_login():
    client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
    if not client_id:
        return redirect(url_for('login', error='Google OAuth is not configured yet. Please add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env'))
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        userinfo = token.get('userinfo') or google.userinfo()
        session['user'] = {
            'name': userinfo.get('name', 'User'),
            'email': userinfo.get('email', ''),
            'picture': userinfo.get('picture', ''),
        }
        return redirect(url_for('home'))
    except Exception as e:
        return redirect(url_for('login', error=f'Login failed: {str(e)}'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html', user=session.get('user'))

@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))

@app.route('/contact')
def contact():
    return render_template('contact.html', user=session.get('user'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        predict_type = request.form.get('predict_type', 'headline')
        gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()
        
        # Check if it's image mode
        image_file = request.files.get('news_image')
        if predict_type == 'image':
            fact_check = None
            if image_file and image_file.filename != '':
                image_bytes = image_file.read()
                mime_type = image_file.content_type or 'image/jpeg'
                if gemini_key:
                    fact_check = gemini_image_fact_check(image_bytes, mime_type, gemini_key)
                else:
                    fact_check = {
                        "success": False,
                        "error": "No Gemini API Key found in config to run dynamic image checks."
                    }
                return render_template("prediction.html", 
                                       fact_check=fact_check, 
                                       predict_type='image',
                                       user=session.get('user'))
            else:
                return render_template("prediction.html", 
                                       prediction_text="Please select an image file to verify.", 
                                       is_fake=False,
                                       predict_type='image',
                                       user=session.get('user'))
        
        # Text prediction path
        message = request.form.get('news', '')
        fact_check = None
        if gemini_key:
            fact_check = gemini_fact_check(message, gemini_key)
            
        if predict_type == 'auto':
            if len(message.strip().split()) < 25:
                is_headline = True
            else:
                is_headline = False
        else:
            is_headline = (predict_type == 'headline')
            
        pred = fake_news_det(message, is_headline=is_headline)
        
        # 1 = Fake (unreliable), 0 = Real (reliable)
        if pred[0] == 1:
            result = "Looking Fake News 📰"
            is_fake = True
        else:
            result = "Looking Real News 📰"
            is_fake = False
            
        return render_template("prediction.html", 
                               prediction_text=result, 
                               is_fake=is_fake, 
                               input_text=message, 
                               predict_type=predict_type,
                               fact_check=fact_check)
    else:
        return render_template('prediction.html', prediction_text="")

if __name__ == '__main__':
    app.run(debug=True)