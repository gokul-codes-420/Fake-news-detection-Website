import os
import time
import random
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

# Allow insecure transport for local OAuth testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

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
    
    last_error = None
    max_attempts = 5
    for attempt in range(max_attempts):
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        # On later attempts, remove Google Search grounding to bypass search-specific rate limits
        if attempt < 3:
            payload["tools"] = [{"google_search": {}}]
            
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
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
                # Transient error or rate limit — back off exponentially with jitter
                last_error = f"API Error (Status {response.status_code}) — retrying..."
                sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
                continue
            else:
                return {
                    "success": False,
                    "error": f"API Error (Status {response.status_code})"
                }
        except Exception as e:
            last_error = str(e)
            sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
            continue
    return {
        "success": False,
        "error": f"API unavailable after {max_attempts} attempts. Please try again shortly. ({last_error})"
    }



def gemini_phone_check(phone_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are an expert telecom security assistant. Analyze this phone number:\n"
        f"'{phone_text}'\n\n"
        "Return a JSON object with the following schema:\n"
        "{\n"
        "  \"country\": \"<country name with flag emoji, e.g. India 🇮🇳>\",\n"
        "  \"format_valid\": \"Valid\" | \"Invalid\",\n"
        "  \"risk_level\": \"Low 🟢\" | \"Medium 🟡\" | \"High 🔴\",\n"
        "  \"line_type\": \"Mobile\" | \"Landline\" | \"VoIP\" | \"Toll-Free\" | \"Unknown\",\n"
        "  \"spam_probability\": \"Low\" | \"Medium\" | \"High\",\n"
        "  \"suggestions\": [\"suggestion 1\", \"suggestion 2\"]\n"
        "}\n"
    )
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['candidates'][0]['content']['parts'][0]['text']
            import json
            data = json.loads(text_response)
            return {
                "success": True,
                "country": data.get("country", "Unknown"),
                "format_valid": data.get("format_valid", "Unknown"),
                "risk_level": data.get("risk_level", "Unknown"),
                "line_type": data.get("line_type", "Unknown"),
                "spam_probability": data.get("spam_probability", "Unknown"),
                "suggestions": data.get("suggestions", [])
            }
        elif response.status_code == 429:
            return {"success": False, "error": "Google Gemini API is currently receiving too many requests (Rate Limit). Please wait a few seconds and try again."}
        elif response.status_code == 503:
            return {"success": False, "error": "Google Gemini API servers are currently overloaded. Please try again in a few moments."}
        else:
            return {"success": False, "error": f"API Error (Status {response.status_code}). Please try again."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "The analysis took too long (Timeout). The AI server is busy. Please try again."}
    except Exception as e:
        return {"success": False, "error": "Connection failed. Please check your internet connection or try again."}

def gemini_sms_check(sms_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are an expert mobile security assistant. Analyze this SMS text message for scams, phishing, or spam:\\n"
        f"'{sms_text}'\\n\\n"
        "Analyze the message and return a JSON object with the following schema:\\n"
        "{\\n"
        "  \\\"verdict\\\": \\\"Safe\\\" | \\\"Suspicious\\\" | \\\"Scam\\\",\\n"
        "  \\\"scam_probability\\\": <integer between 0 and 100>,\\n"
        "  \\\"safe_probability\\\": <integer between 0 and 100>,\\n"
        "  \\\"platform\\\": \\\"<likely platform/brand, e.g. 'WhatsApp', 'Telegram', 'SBI Bank', 'Netflix', 'Amazon', 'Carrier SMS', etc.>\\\",\\n"
        "  \\\"explanation\\\": \\\"<brief explanation in 1-2 sentences>\\\"\\n"
        "}\\n"
        "Make sure scam_probability + safe_probability = 100."
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['candidates'][0]['content']['parts'][0]['text']
            import json
            data = json.loads(text_response)
            return {
                "success": True,
                "verdict": data.get("verdict", "Suspicious"),
                "scam_probability": data.get("scam_probability", 50),
                "safe_probability": data.get("safe_probability", 50),
                "platform": data.get("platform", "Unknown Sender"),
                "explanation": data.get("explanation", "Could not analyze fully.")
            }
        else:
            return {
                "success": False,
                "error": f"API Error (Status {response.status_code})"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_domain_info(url):
    import socket
    import urllib.parse
    import requests
    from bs4 import BeautifulSoup
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        scheme = parsed.scheme
        
        # 1. HTTPS Check
        https = "Yes" if scheme == "https" else "No"
        
        # 2. IP Address
        try:
            ip_address = socket.gethostbyname(domain)
        except Exception:
            ip_address = "Could not resolve"
            
        # 3. Country lookup via free IP API
        country = "Unknown"
        if ip_address != "Could not resolve":
            try:
                res = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=3)
                if res.status_code == 200:
                    country = res.json().get('country', 'Unknown')
            except Exception:
                pass
                
        # 4. Website Title
        title = "Unknown"
        try:
            res = requests.get(url, timeout=3, headers={"User-Agent": "Mozilla/5.0"})
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                title = soup.title.string.strip() if soup.title else "No Title Found"
        except Exception:
            pass
            
        return {
            "domain": domain,
            "title": title,
            "https": https,
            "ip": ip_address,
            "country": country
        }
    except Exception:
        return {
            "domain": "Invalid URL",
            "title": "Unknown",
            "https": "No",
            "ip": "Unknown",
            "country": "Unknown"
        }

def gemini_url_check(target_url, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are a cybersecurity expert. Analyze this URL for security risks, malware, phishing, or scams:\\n"
        f"'{target_url}'\\n\\n"
        "Analyze the URL and return a JSON object with the following schema:\\n"
        "{\\n"
        "  \\\"status\\\": \\\"Safe\\\" | \\\"Suspicious\\\" | \\\"Malicious\\\",\\n"
        "  \\\"safety_score\\\": <integer between 0 and 100>,\\n"
        "  \\\"explanation\\\": \\\"<brief explanation in 1-2 sentences>\\\"\\n"
        "}"
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['candidates'][0]['content']['parts'][0]['text']
            import json
            data = json.loads(text_response)
            return {
                "success": True,
                "status": data.get("status", "Suspicious"),
                "safety_score": data.get("safety_score", 50),
                "explanation": data.get("explanation", "Could not analyze safety fully.")
            }
        else:
            return {
                "success": False,
                "error": f"API Error (Status {response.status_code})"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def gemini_category_check(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are an AI news analyst. Analyze this news headline or article and categorize it:\\n"
        f"'{text}'\\n\\n"
        "Categorize the news into one of these standard categories: Politics, Technology, Business, Sports, Health, Entertainment, Science, World News, or Other.\\n"
        "Return a JSON object with the following schema:\\n"
        "{\\n"
        "  \\\"category\\\": \\\"<One of the categories listed above>\\\",\\n"
        "  \\\"confidence\\\": <integer between 0 and 100>,\\n"
        "  \\\"keywords\\\": [\\\"<key label 1>\\\", \\\"<key label 2>\\\", ...],\\n"
        "  \\\"summary\\\": \\\"<a short 1-sentence summary of why it fits this category>\\\"\\n"
        "}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['candidates'][0]['content']['parts'][0]['text']
            import json
            data = json.loads(text_response)
            return {
                "success": True,
                "category": data.get("category", "Other"),
                "confidence": data.get("confidence", 50),
                "keywords": data.get("keywords", []),
                "summary": data.get("summary", "Categorized successfully.")
            }
        return {"success": False, "error": f"API Error (Status {response.status_code})"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def gemini_ad_scam_check(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are an expert advertising fraud investigator. Analyze this advertisement copy or sponsor link:\\n"
        f"'{text}'\\n\\n"
        "Evaluate it for deceptive claims, bait-and-switch tactics, get-rich-quick, fake sweepstakes, or direct scams.\\n"
        "Return a JSON object with the following schema:\\n"
        "{\\n"
        "  \\\"verdict\\\": \\\"Safe\\\" | \\\"Suspicious\\\" | \\\"Scam\\\",\\n"
        "  \\\"scam_probability\\\": <integer between 0 and 100>,\\n"
        "  \\\"risk_flags\\\": [\\\"<risk flag 1>\\\", \\\"<risk flag 2>\\\", ...],\\n"
        "  \\\"explanation\\\": \\\"<brief explanation in 1-2 sentences of the fraud analysis>\\\"\\n"
        "}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['candidates'][0]['content']['parts'][0]['text']
            import json
            data = json.loads(text_response)
            return {
                "success": True,
                "verdict": data.get("verdict", "Suspicious"),
                "scam_probability": data.get("scam_probability", 50),
                "risk_flags": data.get("risk_flags", []),
                "explanation": data.get("explanation", "Could not analyze fully.")
            }
        return {"success": False, "error": f"API Error (Status {response.status_code})"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def gemini_shop_scam_check(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"You are a consumer protection agent specializing in e-commerce fraud. Analyze this online shopping offer, product deal, or website copy:\\n"
        f"'{text}'\\n\\n"
        "Evaluate it for fake stores, unreal discounts, counterfeit product warning, payment scams, or non-delivery risks.\\n"
        "Return a JSON object with the following schema:\\n"
        "{\\n"
        "  \\\"verdict\\\": \\\"Safe\\\" | \\\"Suspicious\\\" | \\\"Scam\\\",\\n"
        "  \\\"scam_probability\\\": <integer between 0 and 100>,\\n"
        "  \\\"risk_flags\\\": [\\\"<risk flag 1>\\\", \\\"<risk flag 2>\\\", ...],\\n"
        "  \\\"explanation\\\": \\\"<brief explanation in 1-2 sentences of the shopping risk assessment>\\\"\\n"
        "}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['candidates'][0]['content']['parts'][0]['text']
            import json
            data = json.loads(text_response)
            return {
                "success": True,
                "verdict": data.get("verdict", "Suspicious"),
                "scam_probability": data.get("scam_probability", 50),
                "risk_flags": data.get("risk_flags", []),
                "explanation": data.get("explanation", "Could not analyze fully.")
            }
        return {"success": False, "error": f"API Error (Status {response.status_code})"}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
    
    last_error = None
    max_attempts = 5
    for attempt in range(max_attempts):
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
            ]
        }
        # On later attempts, remove Google Search grounding to bypass search-specific rate limits
        if attempt < 3:
            payload["tools"] = [{"google_search": {}}]
            
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
                sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
                continue
            else:
                return {
                    "success": False,
                    "error": f"API Error (Status {response.status_code})"
                }
        except Exception as e:
            last_error = str(e)
            sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
            continue
    return {
        "success": False,
        "error": f"API unavailable after {max_attempts} attempts. Please try again shortly. ({last_error})"
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
PUBLIC_ROUTES = {'login', 'google_login', 'google_callback', 'developer_login', 'clerk_callback', 'static'}

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
    clerk_key = os.environ.get('CLERK_PUBLISHABLE_KEY', '')
    return render_template('login.html', error=error, clerk_publishable_key=clerk_key)

@app.route('/login/developer')
def developer_login():
    """Local developer bypass to sign in as guest without configuring Google OAuth."""
    session['user'] = {
        'name': 'Developer Guest',
        'email': 'developer@example.com',
        'picture': '',
    }
    return redirect(url_for('home'))

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

@app.route('/login/clerk', methods=['POST'])
def clerk_callback():
    data = request.get_json()
    if not data:
        return {"success": False, "error": "No data provided"}, 400
    
    session['user'] = {
        'name': data.get('name', 'User'),
        'email': data.get('email', ''),
        'picture': data.get('picture', ''),
    }
    return {"success": True}

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login?logout=true')

# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html', user=session.get('user'))

@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))

@app.route('/learn')
def learn():
    return render_template('learn.html', user=session.get('user'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Save to local JSON
        import json
        from datetime import datetime
        contact_entry = {
            "name": name,
            "email": email,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            contacts_file = 'contacts.json'
            contacts_data = []
            if os.path.exists(contacts_file):
                with open(contacts_file, 'r') as f:
                    contacts_data = json.load(f)
            contacts_data.append(contact_entry)
            with open(contacts_file, 'w') as f:
                json.dump(contacts_data, f, indent=4)
        except Exception as e:
            print(f"Error saving contact message: {e}")

        # Reload SMTP settings from .env file if it exists
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
                print("Error reloading .env:", e)

        # Check SMTP settings
        mail_server = os.environ.get('MAIL_SERVER')
        mail_port = os.environ.get('MAIL_PORT')
        mail_username = os.environ.get('MAIL_USERNAME')
        mail_password = os.environ.get('MAIL_PASSWORD')
        recipient = os.environ.get('MAIL_RECIPIENT', mail_username)
        
        email_sent = False
        error_msg = None
        
        if mail_server and mail_username and mail_password:
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                port = int(mail_port) if mail_port else 587
                server = smtplib.SMTP(mail_server, port)
                server.starttls()
                server.login(mail_username, mail_password)
                
                # 1. Send admin message
                msg_admin = MIMEMultipart()
                msg_admin['From'] = mail_username
                msg_admin['To'] = recipient
                msg_admin['Subject'] = f"TruthShield: New Contact Message from {name}"
                body_admin = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
                msg_admin.attach(MIMEText(body_admin, 'plain'))
                server.sendmail(mail_username, recipient, msg_admin.as_string())
                
                # 2. Send user copy confirmation (if email is valid)
                if email and '@' in email:
                    msg_user = MIMEMultipart()
                    msg_user['From'] = mail_username
                    msg_user['To'] = email
                    msg_user['Subject'] = "TruthShield Contact Form: Message Received"
                    body_user = f"Hi {name},\n\nThank you for reaching out! We have received your message:\n\n\"{message}\"\n\nWe will get back to you shortly.\n\nBest regards,\nThe TruthShield Team"
                    msg_user.attach(MIMEText(body_user, 'plain'))
                    server.sendmail(mail_username, email, msg_user.as_string())
                
                server.quit()
                email_sent = True
            except Exception as smtp_err:
                error_msg = str(smtp_err)
                print(f"SMTP Error: {smtp_err}")
                
        status = "success"
        if mail_server and mail_username and mail_password and not email_sent:
            status = "partial_error"
            
        return render_template('contact.html', 
                               status=status, 
                               email_sent=email_sent, 
                               error_msg=error_msg, 
                               user=session.get('user'))
                               
    return render_template('contact.html', user=session.get('user'))

def save_to_history(user_email, predict_type, input_text, results):
    if not user_email:
        user_email = 'guest'
    history_file = '/tmp/history.json' if os.environ.get('VERCEL') else 'history.json'
    history_data = []
    
    if os.path.exists(history_file):
        try:
            import json
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except Exception as e:
            print("Error reading history.json:", e)
            
    import uuid
    from datetime import datetime
    
    new_entry = {
        "id": f"hist_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "user_email": user_email,
        "predict_type": predict_type,
        "input_text": input_text,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    history_data.append(new_entry)
    
    try:
        import json
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4)
    except Exception as e:
        print("Error saving history.json:", e)

@app.route('/api/stats')
def api_stats():
    import json as _json
    from datetime import datetime, timezone
    history_file = '/tmp/history.json' if os.environ.get('VERCEL') else 'history.json'
    history_data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = _json.load(f)
        except Exception:
            pass

    today_str = datetime.now().strftime('%Y-%m-%d')

    total_all = len(history_data)
    total_today = 0
    fake_all = 0
    real_all = 0
    fake_today = 0
    real_today = 0
    mode_counts = {}

    for item in history_data:
        ts = item.get('timestamp', '')
        is_today = ts.startswith(today_str)
        mode = item.get('predict_type', 'headline')
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        results = item.get('results', {})
        # Determine verdict
        fact_check = results.get('fact_check', {})
        is_fake_flag = results.get('is_fake', None)

        if fact_check and fact_check.get('success') and fact_check.get('verdict'):
            verd = fact_check['verdict'].upper()
            is_fake_verdict = 'FALSE' in verd
        elif is_fake_flag is not None:
            is_fake_verdict = bool(is_fake_flag)
        else:
            # For sms/url/ad/shop — treat scam as fake
            sms = results.get('sms_result', {})
            url_r = results.get('url_result', {})
            ad = results.get('ad_result', {})
            shop = results.get('shop_result', {})
            any_scam = (
                (sms.get('verdict', '') in ['Scam', 'Suspicious']) or
                (url_r.get('verdict', '') in ['Phishing', 'Suspicious', 'Malicious']) or
                (ad.get('verdict', '') in ['Scam', 'Suspicious']) or
                (shop.get('verdict', '') in ['Scam', 'Suspicious'])
            )
            is_fake_verdict = any_scam

        if is_fake_verdict:
            fake_all += 1
            if is_today:
                fake_today += 1
        else:
            real_all += 1
            if is_today:
                real_today += 1

        if is_today:
            total_today += 1

    return {
        'success': True,
        'total_all': total_all,
        'total_today': total_today,
        'fake_all': fake_all,
        'real_all': real_all,
        'fake_today': fake_today,
        'real_today': real_today,
        'mode_counts': mode_counts,
    }

@app.route('/api/history')
def api_history():
    user = session.get('user')
    email = user.get('email', 'guest') if user else 'guest'
    
    history_file = '/tmp/history.json' if os.environ.get('VERCEL') else 'history.json'
    history_data = []
    
    if os.path.exists(history_file):
        try:
            import json
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except Exception as e:
            print("Error reading history.json:", e)
            
    user_history = [item for item in history_data if item.get('user_email') == email]
    user_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return {"success": True, "history": user_history[:50]}

@app.route('/api/history', methods=['DELETE'])
def api_history_clear():
    user = session.get('user')
    email = user.get('email', 'guest') if user else 'guest'
    
    history_file = '/tmp/history.json' if os.environ.get('VERCEL') else 'history.json'
    history_data = []
    
    if os.path.exists(history_file):
        try:
            import json
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except Exception as e:
            print("Error reading history.json:", e)
    
    # Remove only this user's entries, keep others
    filtered = [item for item in history_data if item.get('user_email') != email]
    
    try:
        import json
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=4)
    except Exception as e:
        print("Error saving history.json:", e)
        return {"success": False, "error": str(e)}
    
    return {"success": True, "message": "History cleared."}


@app.route('/phone', methods=['GET', 'POST'])
def phone_checker():
    gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()
    user_email = session.get('user', {}).get('email', 'guest')
    
    if request.method == 'POST':
        phone_text = request.form.get('phone', '').strip()
        phone_result = None
        if phone_text:
            if gemini_key:
                phone_result = gemini_phone_check(phone_text, gemini_key)
                if phone_result.get('success'):
                    save_to_history(user_email, 'phone', phone_text, {"phone_result": phone_result})
            else:
                phone_result = {"success": False, "error": "No Gemini API Key found."}
            return render_template("phone.html", 
                                   phone_result=phone_result, 
                                   input_text=phone_text,
                                   user=session.get('user'))
        else:
            return render_template("phone.html", 
                                   error="Please enter a phone number to analyze.", 
                                   user=session.get('user'))
    return render_template("phone.html", user=session.get('user'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        predict_type = request.form.get('predict_type', 'headline')
        gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()
        user_email = session.get('user', {}).get('email', 'guest')
        

        # Check if it's SMS mode
        if predict_type == 'sms':
            sms_text = request.form.get('news', '').strip()
            sms_result = None
            if sms_text:
                if gemini_key:
                    sms_result = gemini_sms_check(sms_text, gemini_key)
                    if sms_result.get('success'):
                        save_to_history(user_email, 'sms', sms_text, {"sms_result": sms_result})
                else:
                    sms_result = {
                        "success": False,
                        "error": "No Gemini API Key found in config to run SMS checks."
                    }
                return render_template("prediction.html", 
                                       sms_result=sms_result, 
                                       predict_type='sms',
                                       input_text=sms_text,
                                       user=session.get('user'))
            else:
                return render_template("prediction.html", 
                                       prediction_text="Please paste an SMS text to analyze.", 
                                       is_fake=False,
                                       predict_type='sms',
                                       user=session.get('user'))
        # Check if it's URL mode
        if predict_type == 'url':
            target_url = request.form.get('news', '').strip()
            url_result = None
            domain_info = None
            if target_url:
                domain_info = get_domain_info(target_url)
                if gemini_key:
                    url_result = gemini_url_check(target_url, gemini_key)
                    if url_result.get('success'):
                        save_to_history(user_email, 'url', target_url, {"url_result": url_result, "domain_info": domain_info})
                else:
                    url_result = {
                        "success": False,
                        "error": "No Gemini API Key found in config to run URL checks."
                    }
                return render_template("prediction.html", 
                                       url_result=url_result, 
                                       domain_info=domain_info,
                                       predict_type='url',
                                       input_text=target_url,
                                       user=session.get('user'))
            else:
                return render_template("prediction.html", 
                                       prediction_text="Please paste a URL to analyze.", 
                                       is_fake=False,
                                       predict_type='url',
                                       user=session.get('user'))
        
        # Check if it's News Category Detector mode
        if predict_type == 'category':
            input_text = request.form.get('news', '').strip()
            category_result = None
            if input_text:
                if gemini_key:
                    category_result = gemini_category_check(input_text, gemini_key)
                    if category_result.get('success'):
                        save_to_history(user_email, 'category', input_text, {"category_result": category_result})
                else:
                    category_result = {"success": False, "error": "No Gemini API Key found in config to run checks."}
                return render_template("prediction.html", 
                                       category_result=category_result, 
                                       predict_type='category',
                                       input_text=input_text,
                                       user=session.get('user'))
            else:
                return render_template("prediction.html", prediction_text="Please paste news headline or article to categorize.", is_fake=False, predict_type='category', user=session.get('user'))

        # Check if it's Advertisement Scam Checker mode
        if predict_type == 'ad_scam':
            input_text = request.form.get('news', '').strip()
            ad_scam_result = None
            if input_text:
                if gemini_key:
                    ad_scam_result = gemini_ad_scam_check(input_text, gemini_key)
                    if ad_scam_result.get('success'):
                        save_to_history(user_email, 'ad_scam', input_text, {"ad_scam_result": ad_scam_result})
                else:
                    ad_scam_result = {"success": False, "error": "No Gemini API Key found in config to run checks."}
                return render_template("prediction.html", 
                                       ad_scam_result=ad_scam_result, 
                                       predict_type='ad_scam',
                                       input_text=input_text,
                                       user=session.get('user'))
            else:
                return render_template("prediction.html", prediction_text="Please paste advertisement text to scan.", is_fake=False, predict_type='ad_scam', user=session.get('user'))

        # Check if it's Online Shopping Scam Checker mode
        if predict_type == 'shop_scam':
            input_text = request.form.get('news', '').strip()
            shop_scam_result = None
            if input_text:
                if gemini_key:
                    shop_scam_result = gemini_shop_scam_check(input_text, gemini_key)
                    if shop_scam_result.get('success'):
                        save_to_history(user_email, 'shop_scam', input_text, {"shop_scam_result": shop_scam_result})
                else:
                    shop_scam_result = {"success": False, "error": "No Gemini API Key found in config to run checks."}
                return render_template("prediction.html", 
                                       shop_scam_result=shop_scam_result, 
                                       predict_type='shop_scam',
                                       input_text=input_text,
                                       user=session.get('user'))
            else:
                return render_template("prediction.html", prediction_text="Please paste e-commerce offer or product deal text to scan.", is_fake=False, predict_type='shop_scam', user=session.get('user'))

        # Check if it's image mode
        image_file = request.files.get('news_image')
        if predict_type == 'image':
            fact_check = None
            if image_file and image_file.filename != '':
                image_bytes = image_file.read()
                mime_type = image_file.content_type or 'image/jpeg'
                if gemini_key:
                    fact_check = gemini_image_fact_check(image_bytes, mime_type, gemini_key)
                    if fact_check.get('success'):
                        save_to_history(user_email, 'image', image_file.filename, {"fact_check": fact_check})
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
            
        results = {
            "prediction_text": result,
            "is_fake": is_fake,
            "fact_check": fact_check
        }
        save_to_history(user_email, predict_type, message, results)
            
        return render_template("prediction.html", 
                               prediction_text=result, 
                               is_fake=is_fake, 
                               input_text=message, 
                               predict_type=predict_type,
                               fact_check=fact_check,
                               user=session.get('user'))
    else:
        return render_template('prediction.html', prediction_text="")

if __name__ == '__main__':
    # Use port 5001 by default to avoid conflicts on macOS
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on http://127.0.0.1:{port}...")
    app.run(host='127.0.0.1', port=port, debug=True)