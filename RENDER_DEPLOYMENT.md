# Render Deployment Guide

## ✅ Code is Ready!

Your code has been updated to work on Render. Here's what changed:

### Changes Made:
1. **requirements.txt** — Updated to use Linux-compatible TensorFlow 2.15.0 (instead of 2.16.2 which doesn't exist for Linux)
2. **.gitignore** — Added model files to ignore (they're too large for git)
3. **render.yaml** — Added Render build configuration
4. **main.py** — Updated to handle missing model file gracefully

---

## 🚀 Step-by-Step Deployment

### Step 1: Connect GitHub to Render
1. Go to https://render.com
2. Sign up or log in
3. Click **"New +"** → **"Web Service"**
4. Select **"Connect a repository"**
5. Search for `fish-disease-api` and connect it

### Step 2: Configure the Service
Fill in these settings:
- **Name:** `fish-disease-api`
- **Environment:** `Python`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python main.py`
- **Instance Type:** Free tier is fine for testing

### Step 3: Add Environment Variables (Important!)
Click **"Environment"** and add:
```
PORT=8000
```

### Step 4: Deploy!
Click **"Create Web Service"** and wait for deployment (~3-5 minutes)

---

## ⚠️ IMPORTANT: Handle the Model File

Since the model file (`inceptionv3_fish_final.h5`) is too large for git, you have **two options:**

### Option A: Upload Model via Render's Disk (Recommended for Testing)
1. After deployment, go to your Render service
2. Go to **"Disks"** tab
3. Create a new disk with mount path `/models`
4. Connect via SSH:
   ```bash
   render connect <service-name>
   ```
5. Upload the model file:
   ```bash
   scp inceptionv3_fish_final.h5 render:/models/
   ```
6. Update `main.py` to use `/models/inceptionv3_fish_final.h5`

### Option B: Use Cloud Storage (Recommended for Production)
1. Upload your model to AWS S3, Google Cloud Storage, or similar
2. Set environment variable in Render:
   ```
   MODEL_URL=https://your-s3-bucket.s3.amazonaws.com/inceptionv3_fish_final.h5
   ```
3. Update `main.py` to download from `MODEL_URL` on startup

### Option C: Use Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes inceptionv3_fish_final.h5
git commit -m "Add model via LFS"
git push
```

---

## 🔗 After Deployment

Once deployed, you'll get a URL like:
```
https://fish-disease-api.onrender.com
```

### Update Your Flutter App:
```dart
static const String publicUrl = 'https://fish-disease-api.onrender.com';
```

### Test the API:
```bash
curl https://fish-disease-api.onrender.com/health
```

Expected response:
```json
{"status":"healthy","model_loaded":true,"num_classes":7}
```

---

## 🐛 Troubleshooting

### Deployment fails with "Module not found"
- Check `requirements.txt` has all dependencies
- Ensure Python version 3.11 or higher

### Model file not found error
- See **"IMPORTANT: Handle the Model File"** section above

### API returns 503 (Model not loaded)
- Model file is missing from the deployment
- See **"IMPORTANT: Handle the Model File"** section

### Timeout during build
- Model file may be too large; use cloud storage instead

---

## 💡 Pro Tips

- **Auto-deploy:** Render auto-deploys when you push to GitHub (already enabled in `render.yaml`)
- **Check logs:** Go to Render dashboard → Logs tab to see deployment issues
- **Monitor:** Render has free monitoring; use it to track requests
- **Scale:** If needed, upgrade from free tier to paid

---

## 🆘 Need Help?

If deployment fails:
1. Check the **Logs** tab in Render dashboard
2. Ensure model file is available (via Option A, B, or C above)
3. Verify `requirements.txt` has correct versions

Good luck! 🎉
