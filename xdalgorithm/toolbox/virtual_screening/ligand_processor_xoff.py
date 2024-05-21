import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, TorsionFingerprints
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Cluster import Butina

from xdalgorithm.toolbox.database_builder.rand_id import get_rand_id
from xoff.topology.confgrid import ConfGrid

class LigandProcessor(object):
    def __init__(self, smiles_string, sdf_path, n_cpu=1, enum_isomer=True, max_conformer_per_isomer=500, max_attempts=1000, remove_twisted6ring=True, ignore=3, prot=False, verbose=False):
        self.sdf_path = sdf_path
        self.smiles = rdMolStandardize.StandardizeSmiles(smiles_string)
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.mol_set = []
        self.mol_name = self.smiles
        self.mol_name_set = []

        self.n_cpu = n_cpu
        self.enum_isomer = enum_isomer
        self.max_conformer_per_isomer = max_conformer_per_isomer
        self.max_attempts = max_attempts
        self.remove_twisted6ring = remove_twisted6ring
        self.ignore = ignore
        self.prot = prot
        self.verbose = verbose

    def __enumerate_stereoisomer__(self, mol_list):
        """maximum number of isomers is 1024 by default"""
        enumerated_mol_list = []
        opts = StereoEnumerationOptions(tryEmbedding=True, unique=False)

        for mol in mol_list:
            enumerated_mol_list += list(EnumerateStereoisomers(mol, options=opts))

        return enumerated_mol_list

    def __gen_conformers__(self, mol, n_cpu, numConfs=500, maxAttempts=1000, pruneRmsThresh=0.5, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True, useRandomCoords=True, useSmallRingTorsions=True):
        conformation_indices = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, useRandomCoords=useRandomCoords, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, useSmallRingTorsions=useSmallRingTorsions, enforceChirality=enforceChirality, numThreads=n_cpu)
        return list(conformation_indices)

    def __remove_cis_amide__(self, mol, conformation_indices):
        pattern = Chem.MolFromSmarts('[*][NH1]-;!@C=O')
        matches = mol.GetSubstructMatches(pattern)

        matches_noH = [(a1, a2, a3, a4) for a1, a2, a3, a4 in matches if not mol.GetAtomWithIdx(a1).GetSymbol() == 'H']
        matches_H = [(a1, a2, a3, a4) for a1, a2, a3, a4 in matches if mol.GetAtomWithIdx(a1).GetSymbol() == 'H']

        remove_indices = set()
        for conformation_idx in conformation_indices:
            angles = np.array(TorsionFingerprints.CalculateTorsionAngles(mol, [((matches_noH), 0)], [], confId=conformation_idx)[0][0])
            if np.any(angles > 20) and np.any(np.abs(angles - 360) > 20):
                remove_indices.add(conformation_idx)

            angles = np.array(TorsionFingerprints.CalculateTorsionAngles(mol, [((matches_H), 0)], [], confId=conformation_idx)[0][0])
            if np.any(np.abs(angles - 180) > 20):
                remove_indices.add(conformation_idx)

        return remove_indices

    def __eliminate_twisted_six_membered_ring__(self, mol):
        """ Eliminates twisted aliphatic six membered rings
            Bridged rings are not considered
        """
        ring_info = mol.GetRingInfo()
        six_membered_ring_indices = [atom_rings_idx for atom_rings_idx, atom_rings in enumerate(ring_info.AtomRings()) if len(atom_rings) == 6]
        sp3_simple_six_membered_ring_indices = []

        for six_membered_ring_idx in six_membered_ring_indices:
            save = True
            six_membered_atom_ring = ring_info.AtomRings()[six_membered_ring_idx]
            sp2_atom_indices = []

            for atom_idx in six_membered_atom_ring:
                current_atom = mol.GetAtomWithIdx(atom_idx)
                if current_atom.GetHybridization() != Chem.rdchem.HybridizationType.SP3:
                    for other_atom_idx in sp2_atom_indices:
                        current_bond = mol.GetBondBetweenAtoms(atom_idx, other_atom_idx)
                        if current_bond:
                            save = False
                            break
                    sp2_atom_indices.append(atom_idx)

            if save and len(sp2_atom_indices) <= 2:
                # remove bridged ring atoms
                bridged_ring = False
                for sp3_simple_six_membered_atom_ring in sp3_simple_six_membered_ring_indices:
                    if len(set(six_membered_atom_ring).intersection(set(sp3_simple_six_membered_atom_ring))) == 5:
                        bridged_ring = True
                        sp3_simple_six_membered_ring_indices.remove(sp3_simple_six_membered_atom_ring)
                        break

                if not bridged_ring:
                    sp3_simple_six_membered_ring_indices.append(six_membered_atom_ring)

        conformers = mol.GetConformers()
        removed_conf_indices = []
        for conf_idx, conf in enumerate(conformers):
            for sp3_simple_six_membered_atom_ring in sp3_simple_six_membered_ring_indices:
                dihedral_1 = rdMolTransforms.GetDihedralDeg(conf, sp3_simple_six_membered_atom_ring[0], sp3_simple_six_membered_atom_ring[1], sp3_simple_six_membered_atom_ring[2], sp3_simple_six_membered_atom_ring[3])
                dihedral_2 = rdMolTransforms.GetDihedralDeg(conf, sp3_simple_six_membered_atom_ring[3], sp3_simple_six_membered_atom_ring[4], sp3_simple_six_membered_atom_ring[5], sp3_simple_six_membered_atom_ring[0])
                if (dihedral_1 > 40 and dihedral_2 < -40) or (dihedral_1 < -40 and dihedral_2 > 40):
                    continue
                else:
                    removed_conf_indices.append(conf_idx)

        if mol.GetNumConformers() == 0:
            removed_conf_indices.remove(0)

        return removed_conf_indices

    def __get_nonplanar_ring_with_substituents__(self, mol):
        torsion_list, torsion_list_rings = TorsionFingerprints.CalculateTorsionLists(mol)
        ring_torsions = TorsionFingerprints.CalculateTorsionAngles(mol, [], torsion_list_rings)

        non_planar_rings = []
        for ring_torsion, ring_atoms in zip(ring_torsions, torsion_list_rings):
            if ring_torsion[0][0] > 5:
                all_atoms = set(np.concatenate(ring_atoms[0]))
                non_planar_rings.append(all_atoms)

        rings_by_bond_indices = []
        rings_by_atom_indices = []

        for ring_atom_indices in non_planar_rings:
            bond_indices = []
            atom_indices = []
            for ring_atm_idx in ring_atom_indices:
                current_atom = mol.GetAtomWithIdx(int(ring_atm_idx))
                bonds = current_atom.GetBonds()
                for bond in bonds:
                    atom_idx_1 = bond.GetBeginAtomIdx()
                    atom_idx_2 = bond.GetEndAtomIdx()

                    if mol.GetAtomWithIdx(atom_idx_1).GetSymbol() == 'H' or mol.GetAtomWithIdx(atom_idx_2).GetSymbol() == 'H':
                        continue

                    bond_indices.append(bond.GetIdx())
                    atom_indices.append(atom_idx_1)
                    atom_indices.append(atom_idx_2)

            bond_indices = list(set(bond_indices))
            bond_indices.sort()
            atom_indices = list(set(atom_indices))
            atom_indices.sort()

            rings_by_bond_indices.append(bond_indices)
            rings_by_atom_indices.append(atom_indices)

        return rings_by_bond_indices, rings_by_atom_indices

    def __find_matched_ring__(self, items, query):
        for item_idx, item in enumerate(items):
            if set(query) == set(item):
                return item_idx

    def __merge_rings__(self, mol, bds, ats):
        new_ats = []
        new_bds = []
        removed = set()
        for i in range(len(ats)):
            if i in removed:
                continue

            merged = False
            for j in range(i+1, len(ats)):
                inter = set(ats[i]).intersection(set(ats[j]))
                ring_atoms = set()
                for ai in inter:
                    if mol.GetAtomWithIdx(ai).IsInRing():
                        ring_atoms.add(ai)

                if len(ring_atoms) > 0:
                    new_ring_ats = set(ats[i]).union(set(ats[j]))
                    new_ring_ats = list(new_ring_ats)
                    new_ring_ats.sort()

                    new_ring_bds = set(bds[i]).union(set(bds[j]))
                    new_ring_bds = list(new_ring_bds)
                    new_ring_bds.sort()

                    new_ats.append(new_ring_ats)
                    new_bds.append(new_ring_bds)
                    removed.add(j)
                    merged = True
                    break

            if not merged:
                new_ats.append(ats[i])
                new_bds.append(bds[i])

        return new_bds, new_ats, len(removed)

    def __get_RMSD_by_nonplanar_rings__(self, mol, noH_mol, confIds):
        num_confs = len(confIds)
        matches = noH_mol.GetSubstructMatches(noH_mol, useChirality=True, uniquify=False)
        matches = np.array(matches)

        nonplanar_rings_by_bond_indexes, nonplanar_rings_by_atom_indexes = self.__get_nonplanar_ring_with_substituents__(mol)

        merging = 1
        while merging > 0:
            nonplanar_rings_by_bond_indexes, nonplanar_rings_by_atom_indexes, merging = self.__merge_rings__(mol, nonplanar_rings_by_bond_indexes, nonplanar_rings_by_atom_indexes)

        ring_mols = []
        atomMaps = []
        for bi in nonplanar_rings_by_bond_indexes:
            atomMap = {}
            ring_mols.append(Chem.PathToSubmol(mol, bi, atomMap=atomMap))
            atomMaps.append(atomMap)

        all_ring_rmsd = []

        for i in range(len(ring_mols)):
            this_ringMap = atomMaps[i]
            this_ring_atoms = [x for x in this_ringMap.keys()]

            match_options = []
            for match in matches:
                ring_match = np.array(match)[np.array(this_ring_atoms)]
                flag = True
                for mo in match_options:
                    if np.all(ring_match == mo):
                        flag = False
                        break
                if flag:
                    match_options.append(ring_match)


            ring_rmsd = []

            for u in range(num_confs):
                for v in range(u):
                    min_rmsd = 100
                    for option in match_options:
                        that_i = self.__find_matched_ring__(nonplanar_rings_by_atom_indexes, option)
                        that_ringMap = atomMaps[that_i]
#                        that_ring_atoms = [x for x in that_ringMap.keys()]

                        rmsd_atomMap = [(this_ringMap[_u], that_ringMap[_v]) for _u, _v in zip(this_ring_atoms, option)]

                        rmsd = AllChem.AlignMol(ring_mols[i], ring_mols[that_i], prbCid = confIds[u], refCid = confIds[v], atomMap = rmsd_atomMap)
                        min_rmsd = min(rmsd, min_rmsd)

                    ring_rmsd.append(min_rmsd * (len(this_ring_atoms) ** 0.5))

            all_ring_rmsd.append(ring_rmsd)

        if all_ring_rmsd:
            return np.max(all_ring_rmsd, axis=0)
        else:
            return []

    def __chiral_sulfonamide_N__(self, mol):
        inv = TorsionFingerprints._getAtomInvariantsWithRadius(mol, 2)
        pattern = Chem.MolFromSmarts('[$(NP=O),$(NS=O)]')
        sulfonamide_N = mol.GetSubstructMatches(pattern)
        chiral_N = []

        for i in sulfonamide_N:
            a = mol.GetAtomWithIdx(i[0])
            v = a.GetTotalValence()
            n_inv = set()
            neighbors = [i[0]]
            if v == 3:
                for n in a.GetNeighbors():
                    n_inv.add(inv[n.GetIdx()])
                    neighbors.append(n.GetIdx())

                if len(n_inv) == v:
                    chiral_N.append(neighbors)

        return chiral_N

    def __cluster_by_chiral_sulfonamide_N__(self, mol, confIds, chiral_N):
        if len(chiral_N) == 0:
            return [confIds]

        torsion_list = [([x], 0) for x in chiral_N]

        torsion_angles = [TorsionFingerprints.CalculateTorsionAngles(mol, torsion_list, [], i) for i in confIds]
        torsion_angles = [[x[0][0] for x in items] for items in torsion_angles]

        clusters = [[] for i in range(2**len(chiral_N))]
        for i, ci in enumerate(confIds):
            bits = ''.join([str(int(x > 180)) for x in torsion_angles[i]])
            index = int(bits, 2)
            clusters[index].append(ci)

        return clusters

    def __desalt__(self):
        """remove the smaller parts of disconnected mols"""
        c = rdMolStandardize.LargestFragmentChooser()
        self.mol = c.choose(self.mol)
        self.smiles = Chem.MolToSmiles(self.mol)

    def __calculate_pKa__(self, verbose=False):
        """calcualte pKa"""
        if verbose:
            print('Calculating pKa...')

        temp_prefix = 'temp'
        self.__write_smiles__(temp_prefix, self.smiles)
        os.system('/opt/chemaxon/marvinsuite/bin/cxcalc -i smiles %s.smi pKa majorms2 -H 7.4 > %s.tmp' %(temp_prefix, temp_prefix))
        with open(temp_prefix + '.tmp') as f:
            p_smiles = f.readlines()[-1].split()[-1]

        self.mol = Chem.MolFromSmiles(p_smiles)
        os.remove(temp_prefix + '.tmp')
        os.remove(temp_prefix + '.smi')

    def __write_smiles__(self, file_name_prefix, smiles):
        with open(file_name_prefix + '.smi', 'w') as f:
            f.write(smiles)

    def __write_conformers_to_sdf__(self, sdf_path, mol, conf_indices):
        saved_sdf_file_name = sdf_path + '/' + get_rand_id() + '.sdf'
        w = Chem.SDWriter(saved_sdf_file_name)
        for i, conf_idx in enumerate(conf_indices):
            for prop_name in mol.GetPropNames():
                mol.ClearProp(prop_name)

            mol.SetIntProp('conformer_id', i)
            w.write(mol, confId=conf_idx)

        w.flush()
        w.close()

        return saved_sdf_file_name

    def run_processing(self):
        self.__desalt__()
        chiral = Chem.FindMolChiralCenters(self.mol, includeUnassigned=True)

        undefined = 0
        for c, t in chiral:
            if t == '?':
                undefined += 1

        print('%d undefined chiral centers' %undefined)
        if undefined > self.ignore:
            if self.verbose:
                print('More than %d undefined chiral centers; Skip stereoisomer enumeration.' %self.ignore)
            self.enum_isomer = False

        if self.prot:
            self.__calculate_pKa__(self.verbose)

        self.mol = Chem.AddHs(self.mol)
        self.mol_set = [self.mol]

        if self.enum_isomer:
            if self.verbose:
                print('Enumerating stereoisomers...')
            self.mol_set = self.__enumerate_stereoisomer__(self.mol_set)

        # 3D preparation
        if self.verbose:
            print('Generating conformers and do clustering...')

        smiles_list = []
        sdf_file_name_list = []

        for mol in self.mol_set:
            smiles = Chem.MolToSmiles(mol, allHsExplicit=True)
            smiles_list.append(smiles)

            noH_mol = Chem.RemoveHs(mol)

            cg = ConfGrid([mol], 1, num_confs=self.max_conformer_per_isomer, max_attempts=self.max_attempts, cores=self.n_cpu)
            cg.build()
            mol = cg.minimize('./confgrids')[0]

            conformer_indices = range(mol.GetNumConformers())

            if self.remove_twisted6ring:
                removed_conformer_indices = self.__eliminate_twisted_six_membered_ring__(mol)

                conformer_indices = [x for x in conformer_indices if not x in set(removed_conformer_indices)]

            removed_conformer_indices = self.__remove_cis_amide__(mol, conformer_indices)
            conformer_indices = [x for x in conformer_indices if not x in removed_conformer_indices]

            rmsds = self.__get_RMSD_by_nonplanar_rings__(mol, noH_mol, conformer_indices)

            if len(rmsds) > 0:
                clusters = Butina.ClusterData(rmsds, len(conformer_indices), 0.5, isDistData=True)
            else:
                clusters= [list(range(len(conformer_indices)))]

            chiral_N = self.__chiral_sulfonamide_N__(mol)
            final_clusters = []

            for cluster in clusters:
                current_conformer_indices = [conformer_indices[i] for i in cluster]
                final_clusters.extend(self.__cluster_by_chiral_sulfonamide_N__(mol, current_conformer_indices, chiral_N))

            # save the representative conformer
            saved_conformer_indices = [x[0] for x in final_clusters if x]
            saved_sdf_file_name = self.__write_conformers_to_sdf__(self.sdf_path, mol, saved_conformer_indices)
            sdf_file_name_list.append(saved_sdf_file_name)

        return smiles_list, sdf_file_name_list
