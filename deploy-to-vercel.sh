#!/bin/bash

# 🚀 Vercel Deployment Script for Space Station Safety Detector (Hugging Face Model)
# This script prepares and deploys your app to Vercel

echo "🛰️ Space Station Safety Detector - Vercel Deployment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Error: vercel.json not found. Please run this script from the project root."
    exit 1
fi

# Check if required files exist (model not needed locally)
echo "🔍 Checking required files..."
required_files=("vercel.json" "api/index.py" "public/index.html" "requirements-vercel.txt")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📝 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit for Vercel deployment"
    echo "✅ Git repository initialized"
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Vercel CLI. Please install manually: npm install -g vercel"
        exit 1
    fi
    echo "✅ Vercel CLI installed"
fi

# Check if user is logged in to Vercel
echo "🔐 Checking Vercel authentication..."
if ! vercel whoami &> /dev/null; then
    echo "🔑 Please log in to Vercel..."
    vercel login
    if [ $? -ne 0 ]; then
        echo "❌ Failed to log in to Vercel"
        exit 1
    fi
fi

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
echo "This should be quick — model will load from Hugging Face dynamically."

vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo "Your Space Station Safety Detector is now live on Vercel!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Test your deployment with sample images"
    echo "2. If model is private, ensure you’ve added your HF_TOKEN via:"
    echo "   vercel env add HF_TOKEN"
    echo "3. Monitor logs using 'vercel logs <deployment-url>'"
    echo ""
    echo "🛰️ Happy detecting!"
else
    echo "❌ Deployment failed. Check the error messages above."
    exit 1
fi
