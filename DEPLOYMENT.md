# 🚀 Deployment Guide - Free Hosting Options

## 🥇 **Option 1: Streamlit Community Cloud (Recommended)**

### **Perfect for: Quick demos, easiest setup**

**Step 1: Prepare Your Repository**

```bash
# 1. Copy the standalone version
cp streamlit_app_standalone.py streamlit_app.py
cp requirements_streamlit.txt requirements.txt

# 2. Create GitHub repository (must be public)
git init
git add .
git commit -m "Deploy to Streamlit Cloud"
git remote add origin https://github.com/YOUR_USERNAME/candidate-recommendation-engine.git
git push -u origin main
```

**Step 2: Deploy to Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `streamlit_app.py`
6. Click "Deploy!"

**Step 3: Configuration**

- Users will enter their OpenAI API key directly in the app
- No server configuration needed
- Automatic updates from GitHub

**Live in 2-3 minutes!** 🎉

---

## 🥈 **Option 2: Railway (Full Stack)**

### **Perfect for: Professional deployment with both frontend and backend**

**Step 1: Prepare for Railway**

```bash
# 1. Create railway.json
cat > railway.json << EOF
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run streamlit_app_standalone.py --server.address 0.0.0.0 --server.port \$PORT"
  }
}
EOF

# 2. Create .env.example for Railway
cat > .env.example << EOF
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

**Step 2: Deploy to Railway**

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Add environment variable: `OPENAI_API_KEY`
6. Deploy!

**Benefits:**

- Custom domain
- Environment variables
- $5/month free credit
- Professional features

---

## 🥉 **Option 3: Render**

### **Perfect for: Professional deployment with custom domains**

**Step 1: Create render.yaml**

```yaml
services:
  - type: web
    name: candidate-recommendation-engine
    env: python
    buildCommand: pip install -r requirements_streamlit.txt
    startCommand: streamlit run streamlit_app_standalone.py --server.address 0.0.0.0 --server.port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
```

**Step 2: Deploy to Render**

1. Go to [render.com](https://render.com)
2. Connect GitHub
3. "New Web Service"
4. Select repository
5. Add environment variables
6. Deploy!

**Note:** Free tier sleeps after 15 minutes of inactivity.

---

## 🛠️ **Option 4: Replit (Quickest)**

### **Perfect for: Instant sharing and testing**

**Step 1: Import to Replit**

1. Go to [replit.com](https://replit.com)
2. "Create Repl" → "Import from GitHub"
3. Paste your repository URL
4. Select Python as language

**Step 2: Configure**

```bash
# In Replit shell:
pip install -r requirements_streamlit.txt

# Create .replit file:
cat > .replit << EOF
run = "streamlit run streamlit_app_standalone.py --server.address 0.0.0.0 --server.port 8080"
modules = ["python-3.11"]
EOF
```

**Step 3: Run**

- Click "Run" button
- Share the generated URL
- Public by default

---

## 📱 **Usage Instructions for Recruiters**

### **Once Deployed, Share These Instructions:**

1. **Open the App** → `[Your deployed URL]`

2. **Enter OpenAI API Key**

   - Get free credits at [platform.openai.com](https://platform.openai.com)
   - Paste key in sidebar
   - One-time setup per session

3. **Add Job Description**

   - Enter job title and description
   - Add requirements (optional)

4. **Upload Candidate Resumes**

   - Drag & drop PDF/DOCX/TXT files
   - Or paste resume text manually
   - Up to 20 candidates

5. **Analyze & Review Results**
   - View similarity scores (0-100%)
   - Read AI-generated explanations
   - Export or screenshot results

---

## 🔧 **Deployment Configurations**

### **For Streamlit Cloud:**

```python
# streamlit_app_standalone.py is ready to go!
# Users enter API keys directly
# No server configuration needed
```

### **For Railway/Render:**

```bash
# Environment variables:
OPENAI_API_KEY=sk-...
PORT=8080  # Auto-assigned

# Health check endpoint:
GET /healthz  # Streamlit auto-provides
```

### **Domain Setup (Railway/Render):**

```bash
# Custom domain (optional):
1. Go to dashboard
2. Settings → Custom Domain
3. Add: candidates.yourcompany.com
4. Update DNS records
```

---

## 💰 **Cost Breakdown**

### **Hosting:**

- **Streamlit Cloud**: $0/month (forever)
- **Railway**: $0/month (with $5 credit)
- **Render**: $0/month (with sleep)
- **Replit**: $0/month (public)

### **OpenAI API:**

- **Embeddings**: ~$0.0001 per candidate
- **GPT Summaries**: ~$0.002 per candidate
- **Example**: 100 candidates = ~$0.21

### **Total Cost for 1000 candidates/month:**

- **Hosting**: $0
- **OpenAI**: ~$2.10
- **Total**: ~$2.10/month

---

## 🎯 **Recommended Deployment Strategy**

### **For Demo/Testing:**

1. **Streamlit Cloud** (easiest)
2. Share link with recruiters
3. They enter their own API keys

### **For Production:**

1. **Railway** (best features)
2. Set up company OpenAI account
3. Custom domain for branding
4. Monitor usage and costs

### **For Enterprise:**

1. **AWS/GCP/Azure** with Docker
2. Private deployment
3. Integrated with company SSO
4. Dedicated database

---

## 🔒 **Security Considerations**

### **API Key Security:**

```python
# For public demos:
- Users enter their own keys
- Keys not stored anywhere
- Session-based only

# For company deployment:
- Use environment variables
- Rotate keys regularly
- Monitor API usage
```

### **Data Privacy:**

- Resumes processed in memory only
- No data stored permanently
- OpenAI data retention policies apply
- Consider GDPR/privacy requirements

---

## 📞 **Support & Troubleshooting**

### **Common Issues:**

**"API Key Invalid"**

- Verify key is correct
- Check OpenAI account has credits
- Ensure key has embeddings permission

**"App is Slow"**

- Normal for first load (cold start)
- Processing time scales with candidates
- Consider reducing AI summary generation

**"File Upload Fails"**

- Check file size (max 10MB)
- Verify file format (PDF/DOCX/TXT)
- Try different browser

### **Getting Help:**

- Check deployment platform docs
- Monitor platform status pages
- Contact support via platform dashboards

---

## 🚀 **Ready to Deploy!**

Choose your preferred option and follow the steps above. **Streamlit Cloud** is recommended for the quickest start!

Your recruiters will have access to a professional AI-powered candidate recommendation system in minutes! 🎉
