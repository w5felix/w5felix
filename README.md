# w5felix

Static site for an interactive xG pitch visualization. Ready for deployment on GitHub Pages.

## Deploying to GitHub Pages
This repository includes a GitHub Actions workflow that publishes the site to GitHub Pages on every push to the `main` (or `master`) branch.

Steps:
- In your GitHub repository, go to Settings → Pages and set Source to "GitHub Actions".
- Push or merge to `main` (or trigger the workflow manually via the Actions tab).
- The site will be available at: `https://<your-username>.github.io/<repo-name>/`.

Notes:
- All asset paths are relative (e.g., `style.css`, `model/xg_mlp_infer.js`, `model/xg_mlp_web.json`), so the site works under the `/REPO_NAME/` base path used by project pages.
- No build step is required; this is a pure static site.
