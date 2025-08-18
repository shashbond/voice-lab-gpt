# GitHub Setup Commands

After creating your repository on GitHub, run these commands:

```bash
# Add GitHub as remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/voice-lab-gpt.git

# Verify remote was added correctly
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: SSH Setup (if you have SSH keys configured)
```bash
# Use SSH instead of HTTPS
git remote add origin git@github.com:YOUR_USERNAME/voice-lab-gpt.git
git push -u origin main
```

## Verify Deployment
After pushing, your repository should contain:
- ✅ All source code files (21 files)
- ✅ Documentation (README.md, CONTRIBUTING.md)
- ✅ Configuration files (.gitignore, LICENSE)
- ✅ Docker support (Dockerfile, docker-compose.yml)
- ✅ Jupyter notebooks for Colab
- ✅ Complete test suite
- ✅ Examples and setup files

## Next Steps After GitHub Deployment

1. **Enable GitHub Pages** (optional):
   - Go to Settings → Pages
   - Choose source branch (usually main)
   - Your documentation will be available at: https://YOUR_USERNAME.github.io/voice-lab-gpt/

2. **Add Topics/Tags**:
   - In your repo, click the gear icon next to "About"
   - Add topics: `voice-analysis`, `speech-processing`, `clinical-tools`, `grbas`, `acoustic-analysis`, `python`

3. **Set up Issues Templates**:
   - Go to Settings → Features → Issues → Set up templates
   - Add templates for bug reports and feature requests

4. **Add Branch Protection** (for production):
   - Settings → Branches → Add rule
   - Require pull request reviews before merging

5. **Enable Discussions** (optional):
   - Settings → Features → Discussions
   - Great for community support and questions