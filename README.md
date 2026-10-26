# Cheminformatics Virtual Laboratory

A Streamlit educational app for molecular visualization and student molecular analysis reports.

## Features
- Home page and student information sidebar
- 2D molecular structure rendering with RDKit
- Interactive 3D viewer using 3Dmol.js
- Molecular descriptors calculated with RDKit
- Personalized PDF report generation with 2D/3D figures and property interpretations

## Included example molecules
- Paracetamol
- Ibuprofen
- Amoxicillin

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload these project files to a GitHub repository.
2. In Streamlit Community Cloud, create a new app and select that repository.
3. Set the main file path to `app.py`.
4. Deploy.

The interactive 3D viewer loads 3Dmol.js from its public CDN, so that feature requires an internet connection.

## Notes
This is an educational cheminformatics tool. Molecular descriptors are calculated computationally and are not a substitute for experimental measurements.
