
from Bio.PDB import PDBParser
from pathlib import Path

# Find the Chapter 20 folder
folder = Path(__file__).resolve().parent

# Our protein structure files
experimental_file = folder / "1TRZ_chain_A_cleaned.pdb"

prediction_files = list(folder.glob("*unrelaxed_rank_001*.pdb"))

if not prediction_files:
    raise FileNotFoundError("AlphaFold rank 001 PDB not found")

predicted_file = prediction_files[0]

# Load the protein structures
parser = PDBParser(QUIET=True)

experimental = parser.get_structure("Experimental", experimental_file)
predicted = parser.get_structure("AlphaFold", predicted_file)

# Count the amino acids
exp_residues = list(experimental.get_residues())
pred_residues = list(predicted.get_residues())

print("Experimental structure:", len(exp_residues), "residues")
print("AlphaFold prediction:", len(pred_residues), "residues")
print("Both protein structures loaded successfully!")


from Bio.PDB import Superimposer
import numpy as np

# Match the alpha carbon atoms
exp_atoms = []
pred_atoms = []

for exp_res, pred_res in zip(exp_residues, pred_residues):
    if "CA" in exp_res and "CA" in pred_res:
        exp_atoms.append(exp_res["CA"])
        pred_atoms.append(pred_res["CA"])

# Align the structures
superimposer = Superimposer()
superimposer.set_atoms(exp_atoms, pred_atoms)

# Calculate RMSD
print(f"RMSD: {superimposer.rms:.3f} Å")

# Apply alignment to the predicted structure
superimposer.apply(predicted.get_atoms())

# Calculate distance for each amino acid
distances = []

for exp_atom, pred_atom in zip(exp_atoms, pred_atoms):
    distance = np.linalg.norm(
        exp_atom.coord - pred_atom.coord
    )
    distances.append(distance)

print("Per-residue distances calculated!")


import matplotlib.pyplot as plt

# Plot per-residue distances
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(distances) + 1),
    distances,
    marker="o",
    color="blue"
)

plt.xlabel("Amino Acid Position")
plt.ylabel("Distance (Å)")
plt.title("Experimental vs AlphaFold Structure")
plt.grid(True)

plt.tight_layout()
plt.savefig("residue_distances.png", dpi=300)
plt.show()


# Step 5: AlphaFold confidence scores (pLDDT)

plddt_scores = []

for residue in pred_residues:
    if "CA" in residue:
        plddt_scores.append(residue["CA"].get_bfactor())

# Plot confidence scores
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(plddt_scores) + 1),
    plddt_scores,
    marker="o",
    color="green"
)

plt.xlabel("Amino Acid Position")
plt.ylabel("pLDDT Score")
plt.title("AlphaFold Confidence Scores")
plt.ylim(0, 100)
plt.grid(True)

plt.tight_layout()
plt.savefig("plddt_scores.png", dpi=300)
plt.show()

print("Average pLDDT:", round(np.mean(plddt_scores), 2))


from Bio.PDB import PDBIO

# Save the aligned AlphaFold structure
io = PDBIO()
io.set_structure(predicted)
io.save("aligned_alphafold.pdb")

print("Aligned AlphaFold structure saved!")


# STEP 6: Create 3D protein visualization
from pathlib import Path
import json

folder = Path(__file__).resolve().parent

experimental_path = folder / "1TRZ_chain_A_cleaned.pdb"
alphafold_path = folder / "aligned_alphafold.pdb"

# Create PyMOL script
pymol_script = f"""
reinitialize
load "{experimental_path}", experimental
load "{alphafold_path}", alphafold
hide everything
show cartoon, experimental
show cartoon, alphafold
color blue, experimental
color red, alphafold
set cartoon_transparency, 0.35, alphafold
bg_color white
orient
"""

(folder / "visualize_alignment.pml").write_text(
    pymol_script
)

# Create interactive browser visualization
html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Experimental vs AlphaFold</title>
<script src="https://cdn.jsdelivr.net/npm/ngl@2.4.0/dist/ngl.js"></script>
<style>
body { margin: 0; font-family: Arial; }
h2 { text-align: center; }
#viewport { width: 100%; height: 80vh; }
p { text-align: center; }
</style>
</head>
<body>

<h2>Experimental vs AlphaFold Insulin Structure</h2>
<p>
<span style="color:blue">Blue: Experimental</span>
|
<span style="color:red">Red: AlphaFold</span>
</p>

<div id="viewport"></div>

<script>
window.addEventListener("load", async function () {
    const stage = new NGL.Stage("viewport", {
        backgroundColor: "white"
    });

    const experimental = __EXPERIMENTAL__;
    const alphafold = __ALPHAFOLD__;

    const structures = await Promise.all([
        stage.loadFile(
            new Blob([experimental], {type: "text/plain"}),
            {ext: "pdb"}
        ),
        stage.loadFile(
            new Blob([alphafold], {type: "text/plain"}),
            {ext: "pdb"}
        )
    ]);

    structures[0].addRepresentation("cartoon", {
        color: "blue"
    });

    structures[1].addRepresentation("cartoon", {
        color: "red",
        opacity: 0.65
    });

    stage.autoView();

    window.addEventListener("resize", function () {
        stage.handleResize();
    });
});
</script>
</body>
</html>
"""

html = html_template.replace(
    "__EXPERIMENTAL__",
    json.dumps(experimental_path.read_text())
).replace(
    "__ALPHAFOLD__",
    json.dumps(alphafold_path.read_text())
)

(folder / "protein_comparison.html").write_text(html)

print("PyMOL visualization script saved!")
print("Interactive HTML visualization saved!")


# ==========================================
# STEP 8: ADDITIONAL STRUCTURE ANALYSIS
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert existing results into arrays
residue_distances = np.asarray(distances, dtype=float)
confidence_scores = np.asarray(plddt_scores, dtype=float)

# Make sure the measurements match
if len(residue_distances) != len(confidence_scores):
    raise ValueError(
        "Residue distances and pLDDT scores "
        "must have the same number of values."
    )

positions = np.arange(1, len(residue_distances) + 1)

# ------------------------------------------
# 1. Correlation between pLDDT and distance
# ------------------------------------------

if (
    len(positions) > 1
    and np.std(confidence_scores) > 0
    and np.std(residue_distances) > 0
):
    correlation = np.corrcoef(
        confidence_scores,
        residue_distances
    )[0, 1]
else:
    correlation = float("nan")

print("\nSTEP 8: Additional Analysis")
print("pLDDT vs Distance Correlation:",
      round(correlation, 4))

plt.figure(figsize=(9, 6))

plt.scatter(
    confidence_scores,
    residue_distances,
    color="purple",
    s=75
)

# Label amino acid positions
for position, x, y in zip(
    positions, confidence_scores, residue_distances
):
    plt.annotate(
        str(position),
        (x, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("AlphaFold pLDDT Confidence")
plt.ylabel("Structural Difference (Angstroms)")
plt.title("AlphaFold Confidence vs Structural Difference")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    folder / "plddt_distance_correlation.png",
    dpi=300
)
plt.close()

print("Correlation graph saved!")

# ------------------------------------------
# 2. Distance matrix heatmap
# ------------------------------------------

# Compare all experimental and predicted
# C-alpha atom coordinates

exp_coords = np.array([
    atom.get_coord() for atom in exp_atoms
])

pred_coords = np.array([
    atom.get_coord() for atom in pred_atoms
])

distance_matrix = np.linalg.norm(
    exp_coords[:, None, :] -
    pred_coords[None, :, :],
    axis=2
)

plt.figure(figsize=(9, 7))

plt.imshow(
    distance_matrix,
    cmap="viridis",
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="Distance (Angstroms)")

plt.xlabel("AlphaFold Residue Position")
plt.ylabel("Experimental Residue Position")
plt.title("Experimental vs AlphaFold Distance Matrix")

plt.tight_layout()
plt.savefig(
    folder / "distance_matrix_heatmap.png",
    dpi=300
)
plt.close()

print("Distance heatmap saved!")

# ------------------------------------------
# 3. Save numerical results
# ------------------------------------------

analysis = pd.DataFrame({
    "Residue_Position": positions,
    "pLDDT": confidence_scores,
    "Structural_Distance_A": residue_distances
})

analysis.to_csv(
    folder / "Chapter20_Residue_Analysis.csv",
    index=False
)

# Save a short numerical summary
with open(
    folder / "Chapter20_Numerical_Results.txt",
    "w"
) as file:

    file.write("Chapter 20 - AlphaFold Comparison\n")
    file.write("==============================\n")

    file.write(
        f"Residues compared: {len(positions)}\n"
    )

    file.write(
        f"Average pLDDT: "
        f"{np.mean(confidence_scores):.2f}\n"
    )

    file.write(
        f"Average residue distance: "
        f"{np.mean(residue_distances):.3f} A\n"
    )

    file.write(
        f"Maximum residue distance: "
        f"{np.max(residue_distances):.3f} A\n"
    )

    file.write(
        f"pLDDT-distance correlation: "
        f"{correlation:.4f}\n"
    )

print("Numerical results saved!")
print("\nChapter 20 coding completed!")
