from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Zn not in the list as we have some Zn containing compounds in ChEMBL
# most of them are simple salts
METAL_LIST = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ga", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Cd", "In", "La", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Ac",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr", "Ge", "Sb", "Sn",
]


def exclude_flag(mol, includeRDKitSanitization=True):
    """
    Rules to exclude structures.
    - Metallic or non metallic with more than 7 boron atoms will be excluded
      due to problems when depicting borane compounds.
    """
    rdkit_fails = False
    exclude = False
    metallic = False
    isotope = False
    radicalE = False
    boron_count = 0

    for atom in mol.GetAtoms():
        a_type = atom.GetSymbol()
        if a_type in METAL_LIST:
            metallic = True
        if a_type == "B":
            boron_count += 1
        if atom.GetIsotope() != 0:
            isotope = True
        if atom.GetNumRadicalElectrons() != 0:
            radicalE = True

    if metallic or (not metallic and boron_count > 7) or rdkit_fails or isotope or radicalE:
        exclude = True

    return exclude

def kekulize_mol(m):
    Chem.Kekulize(m)
    return m

def update_mol_valences(m):
    m = Chem.Mol(m)
    m.UpdatePropertyCache(strict=False)
    return m

# derived from the MolVS set, with ChEMBL-specific additions
_normalization_transforms = """
//	Name	SMIRKS
Nitro to N+(O-)=O	[N;X3:1](=[O:2])=[O:3]>>[*+1:1]([*-1:2])=[*:3]
Diazonium N	[*:1]-[N;X2:2]#[N;X1:3]>>[*:1]-[*+1:2]#[*:3]
Quaternary N	[N;X4;v4;+0:1]>>[*+1:1]
Trivalent O	[*:1]=[O;X2;v3;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]
Sulfoxide to -S+(O-)	[!O:1][S+0;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
// this form addresses a pathological case that came up a few times in testing:
Sulfoxide to -S+(O-) 2	[!O:1][SH1+1;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
Trivalent S	[O:1]=[S;D2;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]
// Note that the next one doesn't work propertly because repeated appplications
// don't carry the cations from the previous rounds through. This should be
// fixed by implementing single-molecule transformations, but that's a longer-term
// project
//Alkaline oxide to ions	[Li,Na,K;+0:1]-[O+0:2]>>([*+1:1].[O-:2])
Bad amide tautomer1	[C:1]([OH1;D1:2])=;!@[NH1:3]>>[C:1](=[OH0:2])-[NH2:3]
Bad amide tautomer2	[C:1]([OH1;D1:2])=;!@[NH0:3]>>[C:1](=[OH0:2])-[NH1:3]
Halogen with no neighbors	[F,Cl,Br,I;X0;+0:1]>>[*-1:1]
Odd pyridine/pyridazine oxide structure	[C,N;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[*-0:1]=[*+1:2]-[*-:3]
"""
_normalizer_params = rdMolStandardize.CleanupParameters()
_normalizer = rdMolStandardize.NormalizerFromData(_normalization_transforms,
                                                  _normalizer_params)

_alkoxide_pattern = Chem.MolFromSmarts('[Li,Na,K;+0]-[#7,#8;+0]')


def normalize_mol(m):
    """
    normalize and desalt
    """
    Chem.FastFindRings(m)
    if m.HasSubstructMatch(_alkoxide_pattern):
        m = Chem.RWMol(m)
        for match in m.GetSubstructMatches(_alkoxide_pattern):
            m.RemoveBond(match[0], match[1])
            m.GetAtomWithIdx(match[0]).SetFormalCharge(1)
            m.GetAtomWithIdx(match[1]).SetFormalCharge(-1)
    
    fc = rdMolStandardize.LargestFragmentChooser()
    m = fc.choose(m)

    res = _normalizer.normalize(m)
    return res

def unassigned_chirality(m):
    centers = AllChem.FindMolChiralCenters(m, includeUnassigned=True)
    for c in centers:
        if c[1] == '?':
            return True
    return False

def standardize_mol(m, check_exclusion=True):
    if check_exclusion:
        exclude = exclude_flag(m, includeRDKitSanitization=False)
    else:
        exclude = False
    
    if not exclude:
        try:
            m = update_mol_valences(m)
            m = kekulize_mol(m)
            m = normalize_mol(m)
            m = uncharge_mol(m)
            Chem.SanitizeMol(m)

            if not unassigned_chirality(m):
                return m

            return None

        except:
            return None

    else:
        return None

def uncharge_mol(m):
    uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)
    res = uncharger.uncharge(m)
    res.UpdatePropertyCache(strict=False)
    return res

def standardize_mol_smiles(smiles, check_exclusion=True):
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if not m:
            return None
    except:
        return None

    if check_exclusion:
        if exclude_flag(m, includeRDKitSanitization=False):
            return None

    m = standardize_mol(m, check_exclusion=False)
    if m:
        return Chem.MolToSmiles(m)
    else:
        return None