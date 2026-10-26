import io
import json
import base64
from xml.sax.saxutils import escape

import numpy as np
import io
import json
import base64
from xml.sax.saxutils import escape
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

    fs = ("A low sp<super>3</super> fraction suggests a comparatively planar, unsaturated or aromatic structural character."
          if p["fsp3"] < 0.3 else
          "A higher sp<super>3</super> fraction suggests a more saturated, three-dimensional carbon framework.")

    return [
        ("Molecular Formula", p["formula"], "Shows the elemental composition of the molecule."),
        ("Molecular Weight", f'{p["mw"]:.3f}',
         "Represents the average molecular mass; size and mass influence diffusion and formulation."),
        ("Exact Molecular Weight", f'{p["exact"]:.3f}',
         "The monoisotopic mass from exact isotopic masses; useful when comparing mass-spectrometry data."),
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
         "Counts atoms explicitly present in the RDKit representation; implicit hydrogens are not included."),
        ("Heavy Atoms", p["heavy"], "Counts all non-hydrogen atoms; a measure of structural size."),
        ("Heavy Atom Bonds", p["heavy_bonds"], "Counts connections between non-hydrogen atoms."),
        (f"Sigma ({SIG}) Bonds", p["sigma"],
         f"Every bonded atom pair contains one {SIG} bond; this value includes bonds to hydrogen."),
        (f"Pi ({PI}) Bonds", p["pi"],
         f"{PI} bonds arise from multiple bonds and aromatic systems and relate to electron delocalization."),
        (f"Total Bonds ({SIG} + {PI})", p["total_bonds"],
         f"Total bond components obtained by adding the calculated {SIG} and {PI} bond counts."),
        ("Fraction Csp<super>3</super>", f'{p["fsp3"]:.3f}', fs),
        ("Molar Refractivity", f'{p["mr"]:.3f}',
         "Relates to molecular volume and electronic polarizability."),
    ]


def point_wise(p):
    lip = ("hydrophilic" if p["logp"] < 0 else
           "moderately lipophilic" if p["logp"] <= 3 else "highly lipophilic")
    return [
        f'<b>Molecular identity:</b> The submitted structure has molecular formula {p["formula"]} '
        f'and molecular weight {p["mw"]:.3f}.',
        f'<b>Lipophilicity:</b> LogP = {p["logp"]:.3f}, indicating that the molecule is {lip}.',
        f'<b>Molecular polarity:</b> TPSA = {p["tpsa"]:.1f} &Aring;<super>2</super>, reflecting the contribution '
        f'of polar atoms and functional groups to the surface polarity.',
        f'<b>Hydrogen bonding:</b> {p["hbd"]} hydrogen-bond donor(s) and {p["hba"]} acceptor(s) describe '
        f'possible intermolecular interactions.',
        f'<b>Flexibility:</b> {p["rot"]} rotatable bond(s) indicate the conformational freedom of the molecule.',
        f'<b>Cyclic and aromatic character:</b> {p["rings"]} ring(s), including {p["arom"]} aromatic ring(s).',
        f'<b>Bonding pattern:</b> {p["sigma"]} {SIG} bonds and {p["pi"]} {PI} bonds.',
        f'<b>Three-dimensional character:</b> Fraction Csp<super>3</super> = {p["fsp3"]:.3f}.',
        "<b>Overall conclusion:</b> The descriptors should be interpreted together with the 2D connectivity "
        "and generated 3D geometry to understand composition, bonding, polarity, flexibility and structure.",
    ]


def build_pdf(drug_names, student_name, reg_no):
    ss = getSampleStyleSheet()
    navy = colors.HexColor("#1F3A5F")
    H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=navy, fontSize=20)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=navy, fontSize=13, spaceBefore=10)
    H3 = ParagraphStyle("H3", parent=ss["Heading2"], textColor=navy, fontSize=15, spaceBefore=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    cell = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8, leading=10)
    cell_b = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    grid = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")]

    def info_table(rows, widths):
        data = [[Paragraph(escape(str(a)), cell_b), Paragraph(escape(str(b)), cell)] for a, b in rows]
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle(grid + [("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF7"))]))
        return t

    props = {d: compute_props(DRUGS[d]) for d in drug_names}

    story = [Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", H1),
             Paragraph("FINAL STUDENT MOLECULAR ANALYSIS REPORT", H2),
             info_table([("Student Name", student_name),
                         ("Registration Number", reg_no),
                         ("Number of Drugs Analysed", len(drug_names))], [50 * mm, 120 * mm]),
             Spacer(1, 8), Paragraph("List of Drugs Analysed", H2)]

    rows = [[Paragraph(h, head) for h in ("No.", "Drug", "Input SMILES", "Formula", "MW")]]
    for i, d in enumerate(drug_names, 1):
        p = props[d]
        rows.append([Paragraph(str(i), cell), Paragraph(escape(d), cell),
                     Paragraph(escape(p["input_smiles"]), cell),
                     Paragraph(escape(p["formula"]), cell), Paragraph(f'{p["mw"]:.3f}', cell)])
    lt = Table(rows, colWidths=[10 * mm, 28 * mm, 100 * mm, 20 * mm, 17 * mm], repeatRows=1)
    lt.setStyle(TableStyle(grid + [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597"))]))
    story.append(lt)

    for idx, d in enumerate(drug_names, 1):
        p = props[d]
        smiles = DRUGS[d]
        story += [PageBreak(),
                  Paragraph(f"Drug {idx}: {escape(d)}", H3),
                  info_table([("Input SMILES", p["input_smiles"]),
                              ("Canonical SMILES", p["canonical"]),
                              ("Molecular Formula", p["formula"])], [50 * mm, 120 * mm]),
                  Paragraph("1. Molecular Structures", H2)]

        buf2d = io.BytesIO()
        image_2d(smiles).save(buf2d, format="PNG")
        buf2d.seek(0)
        figs = Table([[Image(buf2d, 80 * mm, 64 * mm), Image(image_3d_png(smiles), 85 * mm, 68 * mm)]])
        story += [figs,
                  Paragraph("<b>Figure.</b> Left: 2D molecular connectivity. Right: generated 3D "
                            "ball-and-stick geometry showing the spatial arrangement of atoms.", body),
                  Paragraph("2. Complete Molecular Properties and Interpretation", H2)]

        data = [[Paragraph(h, head) for h in ("Property", "Value", "Interpretation")]]
        for a, b, c in interpretation_rows(p):
            data.append([Paragraph(a, cell), Paragraph(str(b), cell), Paragraph(c, cell)])
        pt = Table(data, colWidths=[40 * mm, 22 * mm, 108 * mm], repeatRows=1)
        pt.setStyle(TableStyle(grid + [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597"))]))
        story += [pt, Paragraph("3. Point-wise Molecular Interpretation", H2),
                  Paragraph("The following points summarize the most important structural and "
                            "physicochemical features calculated from the submitted SMILES.", body)]
        for s in point_wise(p):
            story.append(Paragraph("&bull; " + s, body))

    story += [Spacer(1, 10), Paragraph("4. Learning Note", H2),
              Paragraph("The numerical descriptors in this report are computationally calculated from the "
                        "submitted molecular structures. They are useful for molecular comparison and "
                        "cheminformatics learning, but experimental measurements may differ depending on "
                        "conditions and methodology.", body)]

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm,
                      title="Final Student Molecular Analysis Report").build(story)
    return out.getvalue()


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
                 "Download a personalised final project report (PDF)"]:
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
    st.write("Select the drugs to include. The report lists every selected drug with its structures, "
             "properties and interpretation.")

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

    if st.button("📑 Generate Personalized Final Project Report"):
        if not name or not reg:
            st.warning("Please enter Student Name and Registration Number in the sidebar first.")
        elif not selected:
            st.warning("Select at least one drug.")
        else:
            with st.spinner("Generating report..."):
                st.session_state["pdf_bytes"] = build_pdf(selected, name, reg)
                st.session_state["pdf_name"] = f"Final_Report_{reg}.pdf"

    pdf = st.session_state.get("pdf_bytes")
    if pdf:
        st.success("Report generated!")
        st.download_button("⬇️ Download PDF Report", pdf,
                           file_name=st.session_state.get("pdf_name", "Final_Report.pdf"),
                           mime="application/pdf")
        try:
            st.pdf(pdf, height=900)
        except Exception:
            b64 = base64.b64encode(pdf).decode()
            st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" '
                        f'height="900" type="application/pdf"></iframe>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
st.set_page_config(page_title="Cheminformatics Virtual Lab", page_icon="🧪", layout="wide")
current = sidebar()
if current == PAGES[0]:
    page_home()
elif current == PAGES[1]:
    page_visualization()
else:
    page_report()
