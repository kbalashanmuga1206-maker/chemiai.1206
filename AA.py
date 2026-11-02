import base64
from io import BytesIO
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="ChemiScope Pro - Molecular Analyzer",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 ChemiScope Pro: Molecular Structure & Property Suite")
st.markdown(
    "Enter SMILES notations to validate, visualize in 2D/3D, calculate properties, compare molecules, and test your knowledge!"
)

# Sidebar for Navigation / Input
st.sidebar.header("Navigation & Input")
app_mode = st.sidebar.radio(
    "Select Mode",
    [
        "Single Molecule Analysis",
        "Compare Multiple Molecules",
        "📝 Molecular Property Assessment",
    ],
)

if app_mode != "📝 Molecular Property Assessment":
  smiles_input = st.sidebar.text_area(
      "Enter SMILES (one per line for comparison):",
      value="CC(=O)OC1=CC=CC=C1C(=O)O\nCCO",  # Default: Aspirin and Ethanol
  )
  smiles_list = [
      s.strip() for s in smiles_input.split("\n") if s.strip()
  ]


def calculate_properties(mol):
  """Calculates standard molecular descriptors."""
  if mol is None:
    return None
  return {
      "Molecular Formula": Chem.MolToSmiles(mol),
      "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 2),
      "LogP": round(Descriptors.MolLogP(mol), 2),
      "H-Bond Donors": Descriptors.NumHDonors(mol),
      "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
      "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
      "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
  }


if app_mode == "Single Molecule Analysis":
  if len(smiles_list) > 0:
    smi = smiles_list[0]
    st.subheader(f"Analyzing SMILES: `{smi}`")

    # Validation
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      st.error(
          "❌ Invalid SMILES notation. Please check your syntax and try again."
      )
    else:
      st.success("✅ Molecular structure successfully validated!")

      # Tabs for features
      tab1, tab2, tab3, tab4 = st.tabs(
          [
              "📐 2D Structure",
              "🌐 3D Structure",
              "📊 Properties",
              "📥 Reports",
          ]
      )

      with tab1:
        st.markdown("### 2D Molecular Visualization")
        img = Draw.MolToImage(mol, size=(450, 450))
        st.image(img, caption=f"2D Structure for {smi}")

      with tab2:
        st.markdown("### 3D Conformer Generation")
        st.info(
            "Generating 3D coordinates using MMFF force field optimization..."
        )
        try:
          mol_3d = Chem.AddHs(mol)
          AllChem.EmbedMolecule(mol_3d, randomSeed=42)
          AllChem.MMFFOptimizeMolecule(mol_3d)
          st.success("3D Conformer generated successfully!")
          st.write(f"Atom count (with hydrogens): {mol_3d.GetNumAtoms()}")
        except Exception as e:
          st.warning(f"Could not generate 3D conformer: {e}")

      with tab3:
        st.markdown("### Complete Molecular Properties")
        props = calculate_properties(mol)
        df_props = pd.DataFrame(
            list(props.items()), columns=["Property", "Value"]
        )
        st.table(df_props)

      with tab4:
        st.markdown("### Download Molecular Report")
        props = calculate_properties(mol)
        report_df = pd.DataFrame([props])
        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name=f"molecular_report_{smi}.csv",
            mime="text/csv",
        )

elif app_mode == "Compare Multiple Molecules":
  st.subheader("⚖️ Multi-Molecule Comparison")
  if len(smiles_list) < 2:
    st.warning("Please enter at least two SMILES strings in the sidebar.")
  else:
    comparison_data = []
    valid_mols = []
    valid_smiles = []

    for smi in smiles_list:
      mol = Chem.MolFromSmiles(smi)
      if mol:
        valid_mols.append(mol)
        valid_smiles.append(smi)
        props = calculate_properties(mol)
        props["SMILES"] = smi
        comparison_data.append(props)
      else:
        st.error(f"Invalid SMILES skipped: `{smi}`")

    if comparison_data:
      df_comp = pd.DataFrame(comparison_data)
      cols = ["SMILES"] + [c for c in df_comp.columns if c != "SMILES"]
      df_comp = df_comp[cols]

      st.markdown("### Property Comparison Table")
      st.dataframe(df_comp, use_container_width=True)

      st.markdown("### Side-by-Side 2D Structures")
      grid_img = Draw.MolsToGridImage(
          valid_mols,
          molsPerRow=min(3, len(valid_mols)),
          subImgSize=(300, 300),
          legends=valid_smiles,
      )
      st.image(grid_img)

      csv_comp = df_comp.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="Download Comparison Report (CSV)",
          data=csv_comp,
          file_name="molecular_comparison_report.csv",
          mime="text/csv",
      )

elif app_mode == "📝 Molecular Property Assessment":
  st.subheader("🧠 Chemical Properties Quiz & Assessment")
  st.markdown(
      "Test your understanding of molecular descriptors, lipophilicity, and"
      " structural properties!"
  )

  with st.form("quiz_form"):
    st.markdown(
        "**Question 1: What does a higher Octanol-Water Partition Coefficient"
        " ($LogP$) generally indicate about a molecule?**"
    )
    q1 = st.radio(
        "Select your answer for Q1:",
        [
            "A) Higher water solubility",
            "B) Higher lipophilicity (fat solubility)",
            "C) Lower molecular weight",
            "D) Higher polarity",
        ],
        key="q1",
    )

    st.markdown("---")
    st.markdown(
        "**Question 2: Which descriptor is commonly used to measure the polar"
        " surface area of a drug candidate for membrane permeability?**"
    )
    q2 = st.radio(
        "Select your answer for Q2:",
        [
            "A) TPSA (Topological Polar Surface Area)",
            "B) Molecular Weight",
            "C) Number of Rotatable Bonds",
            "D) Boiling Point",
        ],
        key="q2",
    )

    st.markdown("---")
    st.markdown(
        "**Question 3: According to Lipinski's Rule of Five, what is the maximum"
        " recommended number of Hydrogen Bond Donors for a drug-like"
        " molecule?**"
    )
    q3 = st.radio(
        "Select your answer for Q3:",
        ["A) 5", "B) 10", "C) 3", "D) 8"],
        key="q3",
    )

    submitted = st.form_submit_button("Submit Assessment")

  if submitted:
    score = 0
    if q1.startswith("B"):
      score += 1
    if q2.startswith("A"):
      score += 1
    if q3.startswith("A"):
      score += 1

    st.markdown("### 📊 Assessment Results")
    if score == 3:
      st.success(f"🎉 Excellent! You scored {score}/3 correct!")
    elif score == 2:
      st.info(f"👍 Good job! You scored {score}/3 correct.")
    else:
      st.warning(f"💡 Keep learning! You scored {score}/3 correct.")

    with st.expander("View Detailed Answer Key"):
      st.markdown(
          """
            - **Question 1:** **B) Higher lipophilicity (fat solubility)** — LogP measures the ratio of concentrations of a solute in an octanol-water two-phase system; higher values mean greater affinity for lipids.
            - **Question 2:** **A) TPSA** — TPSA correlates well with passive molecular transport through membranes.
            - **Question 3:** **A) 5** — Lipinski's Rule of Five states that, generally, an orally active drug has no more than 5 hydrogen bond donors.
            """
      )
