
reinitialize
load "/Users/alex/PycharmProjects/CobblerLearnChemProjects/Chapter 20/1TRZ_chain_A_cleaned.pdb", experimental
load "/Users/alex/PycharmProjects/CobblerLearnChemProjects/Chapter 20/aligned_alphafold.pdb", alphafold
hide everything
show cartoon, experimental
show cartoon, alphafold
color blue, experimental
color red, alphafold
set cartoon_transparency, 0.35, alphafold
bg_color white
orient
