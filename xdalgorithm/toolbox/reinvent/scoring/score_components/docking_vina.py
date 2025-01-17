from typing import List
import os
import random
import numpy as np
from collections import defaultdict
import subprocess
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary


from rdkit import Chem
from rdkit.Chem import AllChem


import time



class DockingVina(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super(DockingVina,self).__init__(parameters)
        self.center_x = self.parameters.specific_parameters["center_x"]
        self.center_y = self.parameters.specific_parameters["center_y"]
        self.center_z = self.parameters.specific_parameters["center_z"]
        self.pdbqt=self.parameters.specific_parameters["pdbqt"]
        self.protonation=self.parameters.specific_parameters.get('protonation', True)

    def calculate_score(self,molecules:List) -> ComponentSummary:
        scores = self._score_molecules(molecules)
        score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)

        return score_summary

    def _score_molecules(self,molecules):
        return np.array([self._score_molecule(molecule) for molecule in molecules])


    def _score_molecule(self,input_mol):
        # round0: valid check
        #input_mol = Chem.MolFromSmiles(smi)
        #if not input_mol:
        #    return 0.0
        # The valid has been checked before input. scoring functio of invalid smiles is 0.
        # round1: pharmacophore matched
        if not input_mol:
             return 0.0
        name_rad=random.random()
        name_time=time.time()

        if self.protonation :
             f=open('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.smi','w')
             smi=Chem.MolToSmiles(input_mol)
             f.write(smi)
             f.close()
             os.system('ligprep -epik -ph 7.4 -s 1  -ismi /backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.smi -osd /backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf -WAIT')
             os.remove(str(name_rad)+str(name_time)+'.log')
             os.remove('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.smi')
             #while not os.path.exists('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf'):
             #   time.sleep(10)
        else:
            input_mol=Chem.AddHs(input_mol) 
            AllChem.EmbedMoecule(input_mol)
            Chem.MolToMolFile(input_mol,'/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf')         
#             input_mol= [m for m in Chem.SDMolSupplier('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf')][0]
#Chem.MolFromMolFile('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf')
        try:
            os.system('obabel -isdf /backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf -omol2 -O '+str(name_rad)+str(name_time)+'.mol2')
            os.system('/home/wei/weilin/soft/mgltools_x86_64Linux2_1.5.7/bin/pythonsh /home/wei/weilin/soft/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l '+str(name_rad)+str(name_time)+'.mol2  -o /backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.pdbqt')
            subprocess.run('/home/wei/weilin/soft/autodock_vina_1_1_2_linux_x86/bin/vina --receptor '+ self.pdbqt +'  --ligand   /backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.pdbqt --center_x '+ str(self.center_x) +'  --center_y '+str(self.center_y)+'  --center_z '+str(self.center_z)+' --out /backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'_out.pdbqt --size_x 20 --size_y 20 --size_z 20  --cpu 1',shell=True,timeout=600)

            f=open('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'_out.pdbqt','r')
            lines=f.readlines()
            score= -0.2 * float(lines[1].split(':')[1].strip().split(' ')[0])
            f.close()
            os.remove('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'_out.pdbqt')
            os.remove('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.pdbqt')
            os.remove(str(name_rad)+str(name_time)+'.mol2')
            os.remove('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.sdf')
        #os.remove('/backup/wei/tmp/rnn/'+str(name_rad)+str(name_time)+'.log')

        except:
            score = 0.0

        return score

    def get_component_type(self):
        return "docking_vina"
