from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

#def standardize_smi(smiles,basicClean=True,clearCharge=True, clearFrag=True, canonTautomer=True, isomeric=False):
def Normalizer(clean_mol):
    try:
        # clean_mol = Chem.MolFromSmiles(smiles)
        # ��ȥ�⡢����ԭ�ӡ���׼������
         if basicClean:
            clean_mol = rdMolStandardize.Cleanup(clean_mol) 
         if clearFrag:
        #  ��������ҪƬ����Ϊ����
            clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        # �������Ի��������
        #if clearCharge:
        #    uncharger = rdMolStandardize.Uncharger() 
        #    clean_mol = uncharger.uncharge(clean_mol)
        # �������칹���Σ���һ����ĳЩ����¿��ܲ�������
         if canonTautomer:
            te = rdMolStandardize.TautomerEnumerator() # idem
            clean_mol = te.Canonicalize(clean_mol)
        #set to True ����������Ϣ��set to False �Ƴ�������Ϣ���������Ӵ�Ϊ��׼�����SMILES��ʽ
        #stan_smiles=Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
    except Exception as e:
        print (e, smiles)
        return None
    return clean_mol
