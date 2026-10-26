import io
import json
import base64
import zipfile
import re
from xml.sax.saxutils import escape

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, Draw, rdMolDescriptors

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)

# ----------------------------------------------------------------------------
# DATA : only these 3 drugs are available in the whole app
# ----------------------------------------------------------------------------
DRUGS = {
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Amoxicillin": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C",
}

PAGES = ["🏠 Home", "🧬 Molecular Visualization", "📄 Final Project Report"]

CPK = {"H": "#FFFFFF", "C": "#404040", "O": "#E02020", "N": "#3050F8",
       "S": "#E6C300", "Cl": "#1FF01F", "F": "#90E050"}

# Greek letters for ReportLab (standard Helvetica has no sigma / pi glyphs)
SIG = '<font name="Symbol">s</font>'
PI = '<font name="Symbol">p</font>'


# ----------------------------------------------------------------------------
# CHEMISTRY HELPERS
# ----------------------------------------------------------------------------
def count_bonds(mol):
    """sigma = every bond (incl. H); pi = extra bond orders after kekulization."""
    mh = Chem.AddHs(mol)
    sigma = mh.GetNumBonds()
    k = Chem.Mol(mh)
    Chem.Kekulize(k, clearAromaticFlags=True)
    pi = sum(int(round(b.GetBondTypeAsDouble())) - 1 for b in k.GetBonds())
    return sigma, pi, sigma + pi


@st.cache_data(show_spinner=False)
def compute_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sigma, pi, total = count_bonds(mol)
    return dict(
        input_smiles=smiles,
        canonical=Chem.MolToSmiles(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        mw=Descriptors.MolWt(mol),
        exact=Descriptors.ExactMolWt(mol),
        logp=Crippen.MolLogP(mol),
        tpsa=rdMolDescriptors.CalcTPSA(mol),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        rot=Lipinski.NumRotatableBonds(mol),
        rings=rdMolDescriptors.CalcNumRings(mol),
        arom=rdMolDescriptors.CalcNumAromaticRings(mol),
        aliph=rdMolDescriptors.CalcNumAliphaticRings(mol),
        atoms=mol.GetNumAtoms(),
        heavy=mol.GetNumHeavyAtoms(),
        total_atoms=Chem.AddHs(mol).GetNumAtoms(),
        heavy_bonds=mol.GetNumBonds(),
        sigma=sigma, pi=pi, total_bonds=total,
        fsp3=rdMolDescriptors.CalcFractionCSP3(mol),
        mr=Crippen.MolMR(mol),
    )


@st.cache_data(show_spinner=False)
def get_molblock(smiles):
    """3D coordinates (with hydrogens) as an SDF/mol block."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


def image_2d(smiles, size=(500, 400)):
    return Draw.MolToImage(Chem.MolFromSmiles(smiles), size=size)


def image_3d_png(smiles):
    """Matplotlib ball-and-stick picture used inside the PDF."""
    mh = Chem.MolFromMolBlock(get_molblock(smiles), removeHs=False)
    pos = mh.GetConformer().GetPositions()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    for b in mh.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax.plot(*zip(pos[i], pos[j]), color="gray", lw=1.5)
    for a in mh.GetAtoms():
        s = a.GetSymbol()
        x, y, z = pos[a.GetIdx()]
        ax.scatter(x, y, z, s=40 if s == "H" else 160,
                   c=CPK.get(s, "#FF69B4"), edgecolors="black",
                   linewidths=0.6, depthshade=False)
    c = pos.mean(axis=0)
    r = (pos.max(axis=0) - pos.min(axis=0)).max() / 2 + 0.5
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------------
# 3D VIEWER (3Dmol.js inside Streamlit)
# ----------------------------------------------------------------------------
STYLE_JS = {
    "Ball and Stick": "{stick:{radius:0.15}, sphere:{scale:0.3}}",
    "Stick": "{stick:{radius:0.2}}",
    "Space Filling (Sphere)": "{sphere:{scale:1.0}}",
    "Wireframe (Line)": "{line:{}}",
}


def viewer_html(molblock, style, surface):
    surf_js = ("v.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.6, color:'white'});"
               if surface else "")
    return f"""
    <div id="c" style="width:100%;height:500px;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      var v = $3Dmol.createViewer("c", {{backgroundColor: "white"}});
      v.addModel({json.dumps(molblock)}, "sdf");
      v.setStyle({{}}, {STYLE_JS[style]});
      {surf_js}
      v.zoomTo(); v.render();
    </script>
    """


# ----------------------------------------------------------------------------
# PDF REPORT
# ----------------------------------------------------------------------------
def interpretation_rows(p):
    """(property, value, interpretation) rows with dynamic text. Markup allowed."""
    logp = p["logp"]
    if logp < 0:
        lip = "Indicates a hydrophilic character, favouring polar (aqueous) environments."
    elif logp <= 3:
        lip = ("Indicates moderate lipophilicity, often reflecting a balance between "
               "hydrophilic and hydrophobic character.")
    else:
        lip = "Indicates high lipophilicity, favouring non-polar (lipid-like) environments."

    t = p["tpsa"]
    if t < 40:
        tp = "A relatively low polar surface area, indicating limited overall molecular polarity."
    elif t <= 90:
        tp = "A moderate polar surface area, indicating a balanced polarity of the molecule."
    else:
        tp = "A high polar surface area, indicating a strongly polar molecule."

    rot = ("No freely rotatable bonds are detected, suggesting a relatively rigid molecular framework."
           if p["rot"] == 0 else
           f'{p["rot"]} rotatable bond(s) give the molecule conformational flexibility.')

    fs = ("A low sp<super>3</super> fraction suggests a comparatively planar, unsaturated or aromatic "
          "structural character."
          if p["fsp3"] < 0.3 else
          "A higher sp<super>3</super> fraction suggests a more saturated, three-dimensional carbon framework.")

    return [
        ("Molecular Formula", p["formula"],
         "Shows the elemental composition of the molecule and is useful for understanding its "
         "basic chemical composition."),
        ("Molecular Weight", f'{p["mw"]:.3f}',
         "Represents the average molecular mass. Molecular size and mass can influence diffusion, "
         "formulation and many physicochemical behaviours."),
        ("Exact Molecular Weight", f'{p["exact"]:.3f}',
         "The monoisotopic mass calculated from exact isotopic masses; this is especially useful "
         "when comparing mass-spectrometry data."),
        ("LogP", f"{logp:.3f}", lip),
        ("TPSA", f"{t:.1f}", tp),
        ("Hydrogen Bond Donors", p["hbd"],
         f'The molecule has {p["hbd"]} donor site(s) capable of donating hydrogen in hydrogen-bond interactions.'),
        ("Hydrogen Bond Acceptors", p["hba"],
         f'The molecule has {p["hba"]} acceptor site(s) capable of accepting hydrogen in hydrogen-bond interactions.'),
        ("Rotatable Bonds", p["rot"], rot),
        ("Ring Count", p["rings"],
         f'The molecule contains {p["rings"]} ring(s), which contribute to its overall shape and structural rigidity.'),
        ("Aromatic Rings", p["arom"],
         f'{p["arom"]} aromatic ring(s) contribute to aromaticity, planarity and delocalized {PI}-electron character.'),
        ("Aliphatic Rings", p["aliph"],
         f'{p["aliph"]} non-aromatic ring(s) contribute to cyclic structure without aromatic {PI}-electron delocalization.'),
        ("Number of Atoms", p["atoms"],
         "Counts atoms explicitly present in the RDKit molecular representation; implicit hydrogens "
         "are not included here."),
        ("Heavy Atoms", p["heavy"],
         "Counts all non-hydrogen atoms and provides a useful measure of molecular structural size."),
        ("Heavy Atom Bonds", p["heavy_bonds"],
         "Counts connections between non-hydrogen atoms in the molecular graph."),
        (f"Sigma ({SIG}) Bonds", p["sigma"],
         f"Every bonded atom pair contains one {SIG} bond; this value includes bonds involving "
         f"explicit or added hydrogens."),
        (f"Pi ({PI}) Bonds", p["pi"],
         f"{PI} bonds arise from multiple bonds and aromatic systems and are associated with "
         f"electron delocalization."),
        (f"Total Bonds ({SIG} + {PI})", p["total_bonds"],
         f"Represents the total bond components obtained by adding the calculated {SIG} and {PI} bond counts."),
        ("Fraction Csp<super>3</super>", f'{p["fsp3"]:.3f}', fs),
        ("Molar Refractivity", f'{p["mr"]:.3f}',
         "Relates to molecular volume and electronic polarizability and reflects how the electron "
         "cloud responds to an electric field."),
    ]


def point_wise(p):
    lip = ("hydrophilic" if p["logp"] < 0 else
           "moderately lipophilic" if p["logp"] <= 3 else "highly lipophilic")
    return [
        f'<b>Molecular identity:</b> The submitted structure has molecular formula {p["formula"]} '
        f'and molecular weight {p["mw"]:.3f}.',
        f'<b>Lipophilicity:</b> LogP = {p["logp"]:.3f}, indicating that the molecule is {lip}. '
        f'This descriptor describes the balance between preference for non-polar and polar environments.',
        f'<b>Molecular polarity:</b> TPSA = {p["tpsa"]:.1f} &Aring;<super>2</super>. TPSA reflects the '
        f'contribution of polar atoms and functional groups to the overall surface polarity of the molecule.',
        f'<b>Hydrogen bonding:</b> The structure contains {p["hbd"]} hydrogen-bond donor(s) and '
        f'{p["hba"]} acceptor(s), which helps describe possible intermolecular interactions.',
        f'<b>Flexibility:</b> {p["rot"]} rotatable bond(s) indicate the degree of conformational '
        f'freedom available to the molecule.',
        f'<b>Cyclic and aromatic character:</b> The molecule contains {p["rings"]} ring(s), including '
        f'{p["arom"]} aromatic ring(s). Aromatic rings are associated with delocalized {PI}-electron systems.',
        f'<b>Bonding pattern:</b> The calculated structure contains {p["sigma"]} {SIG} bonds and '
        f'{p["pi"]} {PI} bonds. This helps distinguish the framework-forming {SIG} bonds from '
        f'multiple-bond and delocalized {PI} bonding.',
        f'<b>Three-dimensional character:</b> Fraction Csp<super>3</super> = {p["fsp3"]:.3f}. This '
        f'descriptor provides a simple indication of the balance between saturated 3D carbon '
        f'environments and planar/unsaturated environments.',
        "<b>Overall conclusion:</b> The calculated descriptors should be interpreted together with the "
        "2D connectivity and generated 3D geometry to understand the molecule's composition, bonding, "
        "polarity, flexibility and structural character.",
    ]


def build_single_pdf(drug_name, student_name, reg_no):
    """ONE separate PDF for ONE drug, in the 'Final Student Molecular Analysis Report' format:
       heading -> student/SMILES table -> 1. Structures -> 2. Properties -> 3. Point-wise -> 4. Learning Note."""
    ss = getSampleStyleSheet()
    navy = colors.HexColor("#1F3A5F")
    H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=navy, fontSize=20, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=navy, fontSize=13, spaceBefore=10)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    cell = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8, leading=10)
    cell_b = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    cell_wrap = ParagraphStyle("cellwrap", parent=cell, wordWrap="CJK")  # long SMILES must wrap
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    grid = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")]

    smiles = DRUGS[drug_name]
    p = compute_props(smiles)

    info_rows = [("Student Name", student_name),
                 ("Registration Number", reg_no),
                 ("Drug Name", drug_name),
                 ("Input SMILES", p["input_smiles"]),
                 ("Canonical SMILES", p["canonical"]),
                 ("Molecular Formula", p["formula"])]
    info = Table([[Paragraph(escape(str(a)), cell_b), Paragraph(escape(str(b)), cell_wrap)]
                  for a, b in info_rows], colWidths=[50 * mm, 120 * mm])
    info.setStyle(TableStyle(grid + [("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF7"))]))

    buf2d = io.BytesIO()
    image_2d(smiles).save(buf2d, format="PNG")
    buf2d.seek(0)
    figs = Table([[Image(buf2d, 80 * mm, 64 * mm), Image(image_3d_png(smiles), 85 * mm, 68 * mm)]])

    data = [[Paragraph(h, head) for h in ("Property", "Value", "Interpretation")]]
    for a, b, c in interpretation_rows(p):
        data.append([Paragraph(a, cell), Paragraph(str(b), cell), Paragraph(c, cell)])
    pt = Table(data, colWidths=[40 * mm, 22 * mm, 108 * mm], repeatRows=1)
    pt.setStyle(TableStyle(grid + [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597"))]))

    story = [Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", H1),
             Paragraph("FINAL STUDENT MOLECULAR ANALYSIS REPORT", H2),
             info,
             Paragraph("1. Molecular Structures", H2),
             figs,
             Paragraph("<b>Figure 1.</b> Left: 2D molecular connectivity. Right: generated 3D "
                       "ball-and-stick geometry showing the spatial arrangement of atoms.", body),
             Paragraph("2. Complete Molecular Properties and Interpretation", H2),
             pt,
             Paragraph("3. Point-wise Molecular Interpretation", H2),
             Paragraph("The following points summarize the most important structural and "
                       "physicochemical features calculated from the student's submitted SMILES.", body)]
    for s in point_wise(p):
        story.append(Paragraph("&bull; " + s, body))
    story += [Paragraph("4. Learning Note", H2),
              Paragraph("The numerical descriptors in this report are computationally calculated from the "
                        "submitted molecular structure. They are useful for molecular comparison and "
                        "cheminformatics learning, but experimental measurements may differ depending on "
                        "conditions and methodology.", body)]

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm,
                      title=f"Final Student Molecular Analysis Report - {drug_name}").build(story)
    return out.getvalue()


def build_all_pdfs(drug_names, student_name, reg_no):
    """Separate report for every selected drug -> {drug_name: pdf_bytes}. Never combined."""
    return {d: build_single_pdf(d, student_name, reg_no) for d in drug_names}


def pdf_filename(reg_no, drug_name):
    safe_reg = re.sub(r"[^A-Za-z0-9_-]+", "_", reg_no)
    return f"Final_Report_{safe_reg}_{drug_name}.pdf"


def zip_pdfs(pdfs, reg_no):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d, data in pdfs.items():
            z.writestr(pdf_filename(reg_no, d), data)
    return buf.getvalue()


def show_pdf(pdf):
    try:
        st.pdf(pdf, height=900)
    except Exception:
        b64 = base64.b64encode(pdf).decode()
        st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" '
                    f'height="900" type="application/pdf"></iframe>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# STREAMLIT PAGES
# ----------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🧪 Virtual Lab")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.divider()
    st.sidebar.subheader("🧑‍🎓 Student Information")
    st.sidebar.text_input("Student Name", key="student_name")
    st.sidebar.text_input("Registration Number", key="reg_no")
    return page


def page_home():
    st.markdown("<h1 style='text-align:center'>🧪 Cheminformatics Virtual Laboratory</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#1F4E8C'>SRM Institute of Science and Technology, "
                "Tiruchirappalli</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>Faculty of Engineering and Technology | School of Sciences | "
                "Division of Chemistry</p>", unsafe_allow_html=True)
    st.divider()
    st.header("Welcome to the Virtual Laboratory")
    st.write("This interactive virtual laboratory helps students understand important concepts in "
             "cheminformatics and computational chemistry using three common drugs.")
    st.subheader("🧪 What can students do?")
    for item in ["Select a drug: Paracetamol, Ibuprofen or Amoxicillin (SMILES loads automatically)",
                 "View complete molecular information",
                 "Generate the 2D structure",
                 "Generate an interactive 3D structure with different visualization styles",
                 "Download a separate personalised final report (PDF) for each selected drug"]:
        st.success(item, icon="✅")
    st.info("Enter your Student Name and Registration Number in the sidebar before generating the final report.")


def page_visualization():
    st.markdown("# 🧬 Molecular Visualization Laboratory")
    st.info("Choose a drug. Its SMILES and all molecular details are loaded automatically.")

    c1, c2 = st.columns([3, 1.4])
    with c2:
        drug = st.selectbox("Choose Example Molecule", list(DRUGS), key="drug")
    smiles = DRUGS[drug]
    with c1:
        st.text_input("SMILES (auto-filled)", value=smiles, disabled=True, key=f"smiles_box_{drug}")

    if Chem.MolFromSmiles(smiles) is None:
        st.error("Invalid SMILES")
        return
    st.success("Valid Molecular Structure!")
    p = compute_props(smiles)

    left, right = st.columns(2)
    with left:
        st.subheader("2D Molecular Structure")
        st.image(image_2d(smiles, (600, 450)))
    with right:
        st.subheader("Molecular Information")
        st.write(f"**Drug:** {drug}")
        st.write(f"**Input SMILES:** `{p['input_smiles']}`")
        st.write(f"**Canonical SMILES:** `{p['canonical']}`")
        st.write(f"**Formula:** {p['formula']}")
        st.write(f"**Molecular Weight:** {p['mw']:.3f} g/mol")
        st.write(f"**Heavy Atoms:** {p['heavy']}")
        st.write(f"**Total Atoms (including H):** {p['total_atoms']}")
        st.write(f"**Heavy-Atom Bonds:** {p['heavy_bonds']}")
        st.write(f"**Sigma (σ) Bonds:** {p['sigma']}")
        st.write(f"**Pi (π) Bonds:** {p['pi']}")
        st.write(f"**Total Bonds (σ + π):** {p['total_bonds']}")

    st.divider()
    st.subheader("🌐 Interactive 3D Molecular Structure")
    style = st.selectbox("Select Molecular Model", list(STYLE_JS))
    surf = st.checkbox("Show van der Waals Surface")
    if st.button("🧬 Generate 3D Structure"):
        st.session_state["view3d"] = dict(drug=drug, style=style, surf=surf)

    cfg = st.session_state.get("view3d")
    if cfg and cfg["drug"] == drug:
        st.caption(f"{drug} – {cfg['style']}" + (" + van der Waals surface" if cfg["surf"] else ""))
        components.html(viewer_html(get_molblock(smiles), cfg["style"], cfg["surf"]), height=520)

    st.divider()
    st.caption("Cheminformatics Virtual Laboratory | Developed using Python, Streamlit and RDKit")


def page_report():
    st.markdown("# 📄 Final Student Molecular Analysis Report")
    st.write("Select the drugs you want. A **separate** report (PDF) is generated for **each** selected "
             "drug – the drugs are never combined into one file. Each report contains: Molecular "
             "Structures, Complete Molecular Properties and Interpretation, Point-wise Molecular "
             "Interpretation and Learning Note.")

    name = st.session_state.get("student_name", "").strip()
    reg = st.session_state.get("reg_no", "").strip()

    selected = st.multiselect("Drugs for the final report", list(DRUGS), default=list(DRUGS))

    if selected:
        st.table({
            "Drug": selected,
            "SMILES": [DRUGS[d] for d in selected],
            "Formula": [compute_props(DRUGS[d])["formula"] for d in selected],
            "MW (g/mol)": [f"{compute_props(DRUGS[d])['mw']:.3f}" for d in selected],
            "σ bonds": [compute_props(DRUGS[d])["sigma"] for d in selected],
            "π bonds": [compute_props(DRUGS[d])["pi"] for d in selected],
        })

    label = "📑 Generate Separate Report for Each Selected Drug"
    if st.button(label):
        if not name or not reg:
            st.warning("Please enter Student Name and Registration Number in the sidebar first.")
        elif not selected:
            st.warning("Select at least one drug.")
        else:
            with st.spinner("Generating reports..."):
                st.session_state["pdfs"] = build_all_pdfs(selected, name, reg)
                st.session_state["pdf_reg"] = reg

    pdfs = st.session_state.get("pdfs")
    if pdfs:
        reg_used = st.session_state.get("pdf_reg", "report")
        st.success(f"{len(pdfs)} separate report(s) generated!")

        if len(pdfs) > 1:
            st.download_button("⬇️ Download all reports (ZIP of separate PDFs)",
                               zip_pdfs(pdfs, reg_used),
                               file_name=f"Final_Reports_{re.sub(r'[^A-Za-z0-9_-]+', '_', reg_used)}.zip",
                               mime="application/zip", key="dl_zip")

        tabs = st.tabs(list(pdfs))
        for tab, (drug, pdf) in zip(tabs, pdfs.items()):
            with tab:
                st.download_button(f"⬇️ Download {drug} Report (PDF)", pdf,
                                   file_name=pdf_filename(reg_used, drug),
                                   mime="application/pdf", key=f"dl_{drug}")
                show_pdf(pdf)


# ----------------------------------------------------------------------------
st.set_page_config(page_title="Cheminformatics Virtual Lab", page_icon="🧪", layout="wide")
current = sidebar()
if current == PAGES[0]:
    page_home()
elif current == PAGES[1]:
    page_visualization()
else:
    page_report()
