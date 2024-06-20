from tokenizer.aa import mass

H2O = 18.01056
NH3 = 17.02655
CO = 27.99491
PROTON = 1.00728

def calculate_ions_and_fragments(peptide):
    b_ions = []
    y_ions = []
    a_ions = []
    c_ions = []
    z_ions = []
    x_ions = []
    prefix_fragments = []
    suffix_fragments = []
    modifications = ["", "-H2O", "-NH3", "-H2O-NH3", "+2H"]

    for i in range(1, len(peptide)):
        b_ion_mass = sum(mass(aa) for aa in peptide[:i])
        y_ion_mass = sum(mass(aa) for aa in peptide[i:])
        
        # Calculate a-ions, c-ions, z-ions, and x-ions
        a_ion_mass = b_ion_mass - CO
        c_ion_mass = b_ion_mass + NH3
        z_ion_mass = y_ion_mass - NH3
        x_ion_mass = y_ion_mass + CO
        
        b_ions.append((b_ion_mass, b_ion_mass - H2O, b_ion_mass - NH3, b_ion_mass - H2O - NH3, (b_ion_mass + PROTON) / 2))
        y_ions.append((y_ion_mass, y_ion_mass - H2O, y_ion_mass - NH3, y_ion_mass - H2O - NH3, (y_ion_mass + PROTON) / 2))
        a_ions.append((a_ion_mass, a_ion_mass - H2O, a_ion_mass - NH3, a_ion_mass - H2O - NH3, (a_ion_mass + PROTON) / 2))
        c_ions.append((c_ion_mass, c_ion_mass - H2O, c_ion_mass - NH3, c_ion_mass - H2O - NH3, (c_ion_mass + PROTON) / 2))
        z_ions.append((z_ion_mass, z_ion_mass - H2O, z_ion_mass - NH3, z_ion_mass - H2O - NH3, (z_ion_mass + PROTON) / 2))
        x_ions.append((x_ion_mass, x_ion_mass - H2O, x_ion_mass - NH3, x_ion_mass - H2O - NH3, (x_ion_mass + PROTON) / 2))
        
        prefix_fragments.append(peptide[:i])
        suffix_fragments.append(peptide[i:])
        
    return b_ions, y_ions, a_ions, c_ions, z_ions, x_ions, prefix_fragments, suffix_fragments, modifications

# Example peptide sequence
peptide = "PEPTIDE"

# Calculate the ions and fragments
b_ions, y_ions, a_ions, c_ions, z_ions, x_ions, prefix_fragments, suffix_fragments, modifications = calculate_ions_and_fragments(peptide)

# Print the results
print(f"Peptide: 0peptide0")
print("b-ions, y-ions, a-ions, c-ions, z-ions, x-ions with modifications:")
for i, (b_ion, y_ion, a_ion, c_ion, z_ion, x_ion, prefix, suffix) in enumerate(zip(b_ions, y_ions, a_ions, c_ions, z_ions, x_ions, prefix_fragments, suffix_fragments), 1):
    print(f"Fragment {i}:")
    for mod, b_ion_mod, y_ion_mod, a_ion_mod, c_ion_mod, z_ion_mod, x_ion_mod in zip(modifications, b_ion, y_ion, a_ion, c_ion, z_ion, x_ion):
        print(f"  b{i}-ion{mod}: {b_ion_mod:.4f} | Prefix fragment: {prefix}")
        print(f"  y{i}-ion{mod}: {y_ion_mod:.4f} | Suffix fragment: {suffix}")
        print(f"  a{i}-ion{mod}: {a_ion_mod:.4f} | Prefix fragment: {prefix}")
        print(f"  c{i}-ion{mod}: {c_ion_mod:.4f} | Prefix fragment: {prefix}")
        print(f"  z{i}-ion{mod}: {z_ion_mod:.4f} | Suffix fragment: {suffix}")
        print(f"  x{i}-ion{mod}: {x_ion_mod:.4f} | Suffix fragment: {suffix}")
