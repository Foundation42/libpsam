# GitHub Pages Deployment Guide

This guide explains how to deploy the PSAM interactive demo to GitHub Pages.

## Automatic Deployment (Recommended)

The demo is automatically deployed when changes are pushed to the `main` branch.

### Setup Steps

1. **Enable GitHub Pages** in repository settings:
   - Go to Settings → Pages
   - Source: "GitHub Actions"
   - Save

2. **Push to main branch**:
   ```bash
   git add demo/
   git commit -m "Update demo"
   git push origin main
   ```

3. **Wait for deployment**:
   - Check Actions tab for deployment status
   - Demo will be available at: `https://foundation42.github.io/libpsam/`

### Workflow File

The deployment is configured in `.github/workflows/deploy-demo.yml`:
- Triggers on pushes to `demo/` directory
- Builds the Vite app
- Deploys to GitHub Pages

## Manual Deployment

If you need to deploy manually:

```bash
# 1. Build the demo
cd demo
npm install
npm run build

# 2. Deploy to gh-pages branch (if using that method)
# Install gh-pages
npm install -g gh-pages

# Deploy
gh-pages -d dist -b gh-pages
```

## Local Testing

Test the production build locally before deploying:

```bash
cd demo

# Build
npm run build

# Preview the build
npm run preview

# Open http://localhost:4173
```

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to `demo/public/`:
   ```
   demo.libpsam.io
   ```

2. Configure DNS:
   - Add CNAME record pointing to `foundation42.github.io`

3. Update `vite.config.ts`:
   ```typescript
   export default defineConfig({
     base: '/', // Remove /libpsam/ prefix
   });
   ```

## Troubleshooting

### Blank Page After Deployment

**Problem**: Page loads but shows blank screen

**Solution**: Check `base` in `vite.config.ts`:
```typescript
base: '/libpsam/', // Must match repository name
```

### 404 on Refresh

**Problem**: Direct URLs give 404 errors

**Solution**: Add `404.html` to `demo/public/` that redirects to index:
```html
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="refresh" content="0; URL='/libpsam/'" />
  </head>
</html>
```

### Assets Not Loading

**Problem**: CSS/JS files return 404

**Solution**: Ensure `base` is set correctly in `vite.config.ts`

### Build Fails in Actions

**Problem**: GitHub Actions build fails

**Solution**:
1. Check Node version in workflow (should be 18+)
2. Ensure `package-lock.json` is committed
3. Check build logs in Actions tab

## Performance Tips

### Optimize Build Size

```bash
# Analyze bundle size
cd demo
npm run build -- --mode analyze

# Use production mode
npm run build
```

### Enable Compression

GitHub Pages automatically serves gzipped files. Ensure your build includes:
- `.js.gz`
- `.css.gz`

Vite handles this automatically in production builds.

### Cache Busting

Vite automatically adds content hashes to filenames:
- `index-abc123.js`
- `style-def456.css`

This ensures browsers don't cache outdated assets.

## Monitoring

### Check Deployment Status

1. **Actions Tab**: See build and deploy logs
2. **Environment**: View deployment history in "Environments" → "github-pages"
3. **Status Badge**: Add to README:

```markdown
[![Demo](https://github.com/Foundation42/libpsam/actions/workflows/deploy-demo.yml/badge.svg)](https://foundation42.github.io/libpsam/)
```

### Analytics (Optional)

Add Google Analytics or similar:

1. Create account and get tracking ID

2. Add to `demo/index.html`:
```html
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

## Security

### Content Security Policy

Add CSP headers in `demo/public/_headers`:
```
/*
  Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';
```

### HTTPS

GitHub Pages automatically serves over HTTPS:
- `https://foundation42.github.io/libpsam/`
- Enforce HTTPS in repository settings

## Rollback

To rollback a deployment:

1. **Via Git**:
   ```bash
   git revert HEAD
   git push origin main
   ```

2. **Via Actions**:
   - Go to Actions tab
   - Select previous successful workflow
   - Click "Re-run all jobs"

## See Also

- [Vite Deployment Guide](https://vitejs.dev/guide/static-deploy.html)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Demo README](../demo/README.md)

---

**Questions?** Open an issue at https://github.com/Foundation42/libpsam/issues
