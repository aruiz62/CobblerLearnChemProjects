#This script will search and display a compounds molecular properties according
# to PubChemPy


import pubchempy as pcp
name = input("Enter Chemical Name: ")
compounds = pcp.get_compounds(name, "name")
compound = compounds[0]
print("Molecular Weight:", compound.molecular_weight)
print("Molecular Formula:", compound.molecular_formula)
print("SMILES:", compound.connectivity_smiles)
print("Name:", name)
print("PubChem CID:", compound.cid)
print("Charge:", compound.charge)
print("XLogP:", compound.xlogp)
print("Hydrogen Bond Donors:", compound.h_bond_donor_count)
print("Hydrogen Bond Acceptors:", compound.h_bond_acceptor_count)



