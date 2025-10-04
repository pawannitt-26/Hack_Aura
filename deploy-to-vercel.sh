#!/bin/bash

# 🚀 Vercel Deployment Script for Space Station Safety Detector
# This script prepares and deploys your app to Vercel

echo "🛰️ Space Station Safety Detector - Vercel Deployment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Error: vercel.json not found. Please run this script from the project root."
    exit 1
fi

# Check if model file exists
if [ ! -f "safety_equipment_model.pt" ]; then
    echo "📦 Copying model file to root directory..."
    if [ -f "src/models/safety_equipment_model.pt" ]; then
        cp src/models/safety_equipment_model.pt safety_equipment_model.pt
        echo "✅ Model file copied successfully"
    else
        echo "❌ Error: Model file not found in src/models/"
        exit 1
    fi
fi

# Check if required files exist
echo "🔍 Checking required files..."
required_files=("vercel.json" "api/index.py" "public/index.html" "requirements-vercel.txt" "safety_equipment_model.pt")

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
echo "This may take 5-10 minutes due to model size..."

vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo "Your Space Station Safety Detector is now live on Vercel!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Test your deployment with sample images"
    echo "2. Set up a custom domain (optional)"
    echo "3. Monitor performance in Vercel dashboard"
    echo ""
    echo "🛰️ Happy detecting!"
else
    echo "❌ Deployment failed. Check the error messages above."
    exit 1
fi
