# Deploying the PSAM Demo

Quick guide to deploy the interactive demo to GitHub Pages.

## Prerequisites

- Repository at `github.com/Foundation42/libpsam`
- Push access to the repository
- GitHub Pages enabled in repository settings

## Quick Deploy

### 1. Enable GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Under "Source", select **GitHub Actions**
3. Save

### 2. Push Demo Files

```bash
git add demo/ .github/workflows/
git commit -m "Add interactive PSAM demo"
git push origin main
```

### 3. Wait for Deployment

- Go to **Actions** tab
- Watch "Deploy Demo to GitHub Pages" workflow
- When complete, demo will be live at:
  **https://foundation42.github.io/libpsam/**

## Local Testing

Test before deploying:

```bash
cd demo

# Install dependencies
npm install

# Run development server
npm run dev
# Open http://localhost:5173

# Build for production
npm run build

# Preview production build
npm run preview
# Open http://localhost:4173
```

## Troubleshooting

### Workflow Fails

**Check Node version**:
- Workflow uses Node 18
- Update `.github/workflows/deploy-demo.yml` if needed

**Missing package-lock.json**:
```bash
cd demo
npm install
git add package-lock.json
git commit -m "Add package-lock.json"
git push
```

### Blank Page After Deploy

**Check base URL** in `demo/vite.config.ts`:
```typescript
base: '/libpsam/', // Must match repository name
```

If using custom domain:
```typescript
base: '/', // For custom domain
```

### Demo Not Updating

**Force rebuild**:
1. Go to Actions tab
2. Select latest "Deploy Demo" workflow
3. Click "Re-run all jobs"

**Clear GitHub Pages cache**:
- May take 5-10 minutes for changes to appear
- Hard refresh browser: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)

## Deployment Flow

```
1. Push to main
   ↓
2. GitHub Actions triggers
   ↓
3. Install dependencies
   ↓
4. Build Vite app (npm run build)
   ↓
5. Upload to GitHub Pages
   ↓
6. Deploy (usually < 2 minutes)
   ↓
7. Live at foundation42.github.io/libpsam
```

## Manual Deployment (Alternative)

If automatic deployment isn't working:

```bash
# Install gh-pages
npm install -g gh-pages

# Build and deploy
cd demo
npm run build
gh-pages -d dist -b gh-pages

# Or use npx
npx gh-pages -d dist -b gh-pages
```

Then configure GitHub Pages to use `gh-pages` branch in settings.

## Custom Domain (Optional)

### Setup

1. **Add CNAME file** to `demo/public/CNAME`:
   ```
   demo.libpsam.io
   ```

2. **Update vite.config.ts**:
   ```typescript
   base: '/', // Remove /libpsam/
   ```

3. **Configure DNS** (at your domain provider):
   - Type: CNAME
   - Name: demo (or @)
   - Value: foundation42.github.io
   - TTL: 3600

4. **Enable in GitHub**:
   - Settings → Pages → Custom domain
   - Enter: `demo.libpsam.io`
   - Wait for DNS check
   - Enable "Enforce HTTPS"

## Performance Tips

### Lighthouse Score

The demo is optimized for performance:
- ✅ Code splitting
- ✅ Asset optimization
- ✅ Lazy loading
- ✅ Gzip compression

Expected Lighthouse scores:
- Performance: 90+
- Accessibility: 95+
- Best Practices: 100
- SEO: 100

### Monitoring

Add analytics (optional):

```typescript
// demo/src/main.tsx
if (import.meta.env.PROD) {
  // Add Google Analytics or similar
}
```

## Rollback

If deployment has issues:

```bash
# Revert last commit
git revert HEAD
git push origin main

# Or restore specific commit
git reset --hard <previous-commit-hash>
git push --force origin main
```

## See Also

- [Demo README](./README.md)
- [GitHub Pages Guide](../docs/GITHUB_PAGES.md)
- [Vite Deployment](https://vitejs.dev/guide/static-deploy.html)

---

**Questions?** Open an issue at https://github.com/Foundation42/libpsam/issues
