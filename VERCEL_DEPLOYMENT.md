# 🚀 Vercel Deployment Guide for Space Station Safety Detector

This guide will help you deploy your Streamlit app with the trained YOLO model to Vercel.

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Vercel CLI** (optional): Install with `npm i -g vercel`

## 🏗️ Project Structure

Your project should have this structure for Vercel deployment:

```
Hack_Aura/
├── api/
│   └── index.py              # Serverless API endpoint
├── public/
│   └── index.html            # Frontend web interface
├── vercel.json               # Vercel configuration
├── requirements-vercel.txt   # Python dependencies
├── safety_equipment_model.pt # Trained model (21MB)
└── src/                      # Original source code
```

## 🚀 Deployment Steps

### Method 1: Deploy via Vercel Dashboard (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add Vercel deployment files"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect the configuration

3. **Configure Build Settings**:
   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave as default)
   - **Build Command**: Leave empty (Vercel will auto-detect)
   - **Output Directory**: Leave empty
   - **Install Command**: Leave empty

4. **Environment Variables** (if needed):
   - No environment variables required for this deployment

5. **Deploy**:
   - Click "Deploy"
   - Wait for the build to complete (may take 5-10 minutes due to model size)

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel --prod
   ```

4. **Follow the prompts**:
   - Link to existing project or create new one
   - Confirm settings
   - Wait for deployment

## ⚙️ Configuration Details

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    },
    {
      "src": "public/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/index.py"
    },
    {
      "src": "/(.*)",
      "dest": "/public/$1"
    }
  ],
  "functions": {
    "api/index.py": {
      "maxDuration": 30
    }
  }
}
```

### Key Features:
- **Serverless API**: Handles image processing and model inference
- **Static Frontend**: Beautiful web interface for image upload
- **Model Caching**: Model is loaded once and cached for subsequent requests
- **CORS Support**: Allows frontend to communicate with API
- **Error Handling**: Comprehensive error handling and user feedback

## 🔧 Technical Considerations

### Model Size & Performance
- **Model Size**: 21MB (within Vercel's limits)
- **Cold Start**: First request may take 10-15 seconds to load model
- **Warm Requests**: Subsequent requests are much faster (~2-3 seconds)
- **Memory Usage**: Model requires ~200MB RAM when loaded

### Optimizations Made
1. **OpenCV Headless**: Using `opencv-python-headless` to reduce package size
2. **Model Caching**: Global variable caching prevents reloading
3. **Image Compression**: Base64 encoding with PNG compression
4. **Error Handling**: Graceful fallbacks for model loading failures

## 🧪 Testing Your Deployment

1. **Upload an Image**: Use the web interface to upload a test image
2. **Adjust Settings**: Try different confidence and IoU thresholds
3. **Check Results**: Verify detection accuracy and bounding boxes
4. **Performance**: Monitor response times and error rates

## 🐛 Troubleshooting

### Common Issues:

1. **Model Not Found Error**:
   - Ensure `safety_equipment_model.pt` is in the root directory
   - Check file size is ~21MB
   - Verify file is committed to Git

2. **Build Timeout**:
   - Vercel has a 15-minute build limit
   - Model download during build may cause timeouts
   - Consider using a smaller model or external storage

3. **Memory Issues**:
   - Vercel Pro plan has 1GB memory limit
   - Model + dependencies may exceed free tier limits
   - Consider upgrading to Pro plan

4. **Cold Start Delays**:
   - First request after inactivity takes longer
   - This is normal for serverless functions
   - Consider implementing a keep-alive ping

### Debug Steps:

1. **Check Vercel Logs**:
   ```bash
   vercel logs [deployment-url]
   ```

2. **Test API Directly**:
   ```bash
   curl -X POST https://your-app.vercel.app/api \
     -H "Content-Type: application/json" \
     -d '{"image": "data:image/png;base64,...", "confidence": 0.25}'
   ```

3. **Monitor Function Metrics**:
   - Check Vercel dashboard for function execution times
   - Monitor memory usage and error rates

## 📊 Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| Cold Start | 10-15 seconds |
| Warm Request | 2-3 seconds |
| Model Size | 21MB |
| Memory Usage | ~200MB |
| Max Duration | 30 seconds |

## 🔄 Updates and Maintenance

### Updating the Model:
1. Replace `safety_equipment_model.pt` with new model
2. Commit and push changes
3. Vercel will automatically redeploy

### Updating Dependencies:
1. Modify `requirements-vercel.txt`
2. Commit and push changes
3. Vercel will rebuild with new dependencies

## 💡 Tips for Production

1. **Monitor Usage**: Keep track of function invocations and costs
2. **Set Up Alerts**: Configure Vercel alerts for errors and performance
3. **Optimize Images**: Compress uploaded images to reduce processing time
4. **Cache Results**: Consider implementing result caching for repeated images
5. **Rate Limiting**: Implement rate limiting for production use

## 🎯 Next Steps

After successful deployment:

1. **Custom Domain**: Set up a custom domain in Vercel dashboard
2. **Analytics**: Enable Vercel Analytics for usage insights
3. **Monitoring**: Set up error tracking and performance monitoring
4. **Scaling**: Consider upgrading to Vercel Pro for better performance

## 📞 Support

If you encounter issues:

1. Check Vercel documentation: [vercel.com/docs](https://vercel.com/docs)
2. Review function logs in Vercel dashboard
3. Test locally with Vercel CLI: `vercel dev`
4. Check GitHub issues for similar problems

---

**Happy Deploying! 🚀**

Your Space Station Safety Equipment Detector is now ready to help astronauts stay safe! 🛰️
