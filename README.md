# Maternal Health Risk Prediction — HTML/CSS/JavaScript

This is a static GitHub Pages version of the maternal health ANN project. It uses HTML, CSS and JavaScript only in the browser. Three ANN models were trained separately and their learned parameters are stored in `model/*.json` so GitHub Pages can perform predictions without Python or a server.

## Files
- `index.html` — website structure
- `style.css` — design
- `script.js` — prediction logic
- `model/*.json` — ANN parameters and performance
- `data/*.csv` — dataset copies

The public Mendeley CSV excludes `Patient ID` and `Name` because those are identifiers and are not model features.

## GitHub Pages
Create a GitHub repository, upload all files, then go to Settings → Pages → Build and deployment → Source: Deploy from a branch → `main` → `/ (root)` → Save. GitHub Pages will publish the site.
