#!/bin/bash

# Quick GitHub deployment script for Streamlit Cloud

echo "🚀 Preparing for Streamlit Cloud Deployment"
echo "==========================================="

# Copy standalone version as main app
echo "📦 Preparing files..."
cp streamlit_app_standalone.py streamlit_app.py
cp requirements_streamlit.txt requirements.txt

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << EOF
# Environment
.env
*.env
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Logs
*.log
api.log
streamlit.log
run.log

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Process files
*.pid

# Streamlit
.streamlit/secrets.toml
EOF
fi

# Initialize git if not already done
if [ ! -d .git ]; then
    echo "🔧 Initializing git repository..."
    git init
fi

# Stage files
echo "📁 Staging files..."
git add .

# Commit
echo "💾 Committing changes..."
git commit -m "Deploy Candidate Recommendation Engine to Streamlit Cloud

Features:
- AI-powered candidate matching
- PDF/DOCX/TXT resume processing  
- OpenAI embeddings and GPT summaries
- Interactive charts and analysis
- Standalone deployment ready"

echo ""
echo "🎉 Repository prepared for deployment!"
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository at: https://github.com/new"
echo "2. Copy and run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/candidate-recommendation-engine.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy to Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Select your repository"
echo "   - Main file: streamlit_app.py"
echo "   - Click 'Deploy!'"
echo ""
echo "🔗 Your app will be live at: https://YOUR_USERNAME-candidate-recommendation-engine-streamlit-app-xyz.streamlit.app"
echo ""
echo "💡 Users will need to enter their OpenAI API key in the sidebar"
