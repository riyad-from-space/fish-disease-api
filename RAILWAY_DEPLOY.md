# 🚀 Railway Deployment Guide - Step by Step

## ✅ What's Ready

Your code is committed to git and ready to deploy!

## 📋 Deployment Steps

### Step 1: Create GitHub Repository

1. Go to: **https://github.com/new**
2. Repository name: `fish-disease-api`
3. Make it **Public** (required for Railway free tier)
4. **DO NOT** initialize with README (we already have files)
5. Click "Create repository"

### Step 2: Push Code to GitHub

Copy the commands GitHub shows you (under "push an existing repository"):

```bash
cd /Users/riyadafromspace/Documents/api_integration

git remote add origin https://github.com/YOUR-USERNAME/fish-disease-api.git
git branch -M main
git push -u origin main
```

Replace `YOUR-USERNAME` with your actual GitHub username.

### Step 3: Upload Model File to GitHub

Your model is too large for GitHub (130MB > 100MB limit).

**Option A: Use Git LFS (Recommended)**

```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track the model file
cd /Users/riyadafromspace/Documents/api_integration
git lfs track "*.h5"
git add .gitattributes

# Add and commit model
cp /Users/riyadafromspace/Documents/inceptionv3_fish_final.h5 .
git add inceptionv3_fish_final.h5
git commit -m "Add InceptionV3 model"
git push
```

**Option B: Deploy without model (then upload directly to Railway)**

Skip model upload to GitHub. We'll add it directly to Railway later.

### Step 4: Deploy to Railway

1. Go to: **https://railway.app/new**
2. Click "Deploy from GitHub repo"
3. Select your `fish-disease-api` repository
4. Railway will auto-detect Python and start deploying

### Step 5: Configure Railway

Once deployed:

1. Click on your service
2. Go to "Variables" tab
3. Add environment variable:
   ```
   MODEL_PATH=./inceptionv3_fish_final.h5
   ```

### Step 6: Upload Model (if not using Git LFS)

1. In Railway dashboard, click "Settings"
2. Find "Volumes" section
3. Or use Railway CLI to upload:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Upload model file (if needed)
# Railway will use the one from git if you used LFS
```

### Step 7: Get Your URL

1. Go to "Settings" → "Domains"
2. Click "Generate Domain"
3. You'll get: `https://your-app.up.railway.app`

**Use this URL in your Flutter app!**

---

## 🎯 Quick Summary

### What You Need to Do:

1. ✅ Sign up at railway.app (with GitHub)
2. ✅ Create GitHub repo at github.com/new
3. ✅ Run the git commands to push code
4. ✅ Handle model file (Git LFS or upload later)
5. ✅ Deploy on Railway from your GitHub repo
6. ✅ Get your permanent URL

---

## 💡 Alternative: Skip GitHub, Deploy Directly

If you don't want to use GitHub:

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Deploy: `railway up`

But this requires Node.js installed.

---

## ⚠️ Important Notes

**Free Tier Limits:**
- 500 hours/month runtime (≈17 days)
- $5 credit/month (usually enough)
- After credit exhausted, service pauses

**Model File:**
- 130MB is large but Railway supports it
- Git LFS is recommended for version control
- Or upload directly to Railway volumes

---

## 🆘 Need Help?

Tell me which step you're on and I'll guide you through it!

**Current Status:**
- ✅ Code ready in git
- ✅ All deployment files created
- ⏳ Waiting for you to create GitHub repo
- ⏳ Then we'll push and deploy to Railway

**Next:** Create GitHub repo and tell me your username!
