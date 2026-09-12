from rdkit import Chem
from rdkit.Chem import Descriptors

smiles = "CCO"
molecule = Chem.MolFromSmiles(smiles)
print("Exact Molecular Weight:", Descriptors.ExactMolWt(molecule))
print("Hydrogen Bond Donors:", Descriptors.NumHDonors(molecule))
print("TPSA:", Descriptors.TPSA(molecule))

