from rdkit import Chem
from rdkit.Chem import QED

def calculate_QED_quantities(mol):
    donor_pattern = Chem.MolFromSmarts('[$([#8,#7,#16;!H0])]')
    acceptor_pattern = Chem.MolFromSmarts('[$([N;^1,^2&!H2,^3&!X4]),$([#8;H0,H1]),$([n;H0;X2]),F]')
    num_hydrogen_bond_donor = len(mol.GetSubstructMatches(donor_pattern))
    num_hydrogen_bond_acceptor = len(mol.GetSubstructMatches(acceptor_pattern))
    QED_score = QED.qed(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    prop = QED.properties(mol)
    QED_dict = {'formal_charge': formal_charge,
                'molecular_weights': prop.MW,
                'AlogP': prop.ALOGP,
                'num_hydrogen_bond_acceptor': num_hydrogen_bond_acceptor,
                'num_hydrogen_bond_donor': num_hydrogen_bond_donor,
                'polar_surface_area': prop.PSA,
                'num_rotatable_bonds': prop.ROTB,
                'num_aromatic_rings': prop.AROM,
                'num_structural_alerts': prop.ALERTS,
                'QED_score': QED_score}

    return QED_dict
