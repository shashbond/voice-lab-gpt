# 🚀 GitHub Deployment Steps

## Step 1: Add GitHub Remote
```bash
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/voice-lab-gpt.git
```

## Step 2: Verify Remote Connection
```bash
git remote -v
```
**Expected output:**
```
origin  https://github.com/YOUR_USERNAME/voice-lab-gpt.git (fetch)
origin  https://github.com/YOUR_USERNAME/voice-lab-gpt.git (push)
```

## Step 3: Set Main Branch and Push
```bash
git branch -M main
git push -u origin main
```

## Step 4: Verify Deployment
Go to your GitHub repository URL:
`https://github.com/YOUR_USERNAME/voice-lab-gpt`

You should see all 22 files uploaded successfully.

## Alternative: SSH Method (if you have SSH keys set up)
```bash
# Use SSH instead of HTTPS
git remote add origin git@github.com:YOUR_USERNAME/voice-lab-gpt.git
git push -u origin main
```

## Troubleshooting

### If you get authentication errors:
1. **GitHub Personal Access Token method:**
   ```bash
   # When prompted for password, use your Personal Access Token
   # Create token at: GitHub Settings → Developer settings → Personal access tokens
   ```

2. **GitHub CLI method:**
   ```bash
   # Install GitHub CLI first
   brew install gh  # macOS
   # or download from: https://cli.github.com/
   
   gh auth login
   git push -u origin main
   ```

### If remote already exists:
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/voice-lab-gpt.git
git push -u origin main
```

## Next Steps After Successful Push
1. Add repository topics (see GITHUB_CONFIGURATION.md)
2. Enable GitHub Pages (optional)
3. Set up issue templates
4. Configure branch protection rules