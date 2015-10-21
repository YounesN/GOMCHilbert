/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 1.0 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#include "DCOnSphere.h"
#include "TrialMol.h"
#include "DCData.h"
#include "../XYZArray.h"
#include "../PRNG.h"
#include "../Forcefield.h"
#include "../MolSetup.h"

namespace cbmc

{
   DCOnSphere::DCOnSphere(DCData* data, const mol_setup::MolKind kind,
			  uint atom, uint focus) :
     data(data), atom(atom),
     focus(focus)
   { 
      using namespace mol_setup;
      std::vector<Bond> bonds = AtomBonds(kind, atom);
      for(uint i = 0; i < bonds.size(); ++i)
      {
         if(bonds[i].a0 == focus || bonds[i].a1 == focus)
	 {
            bondLength = data->ff.bonds.Length(bonds[i].kind);
            break;
         }
      }
   }

   void DCOnSphere::BuildOld(TrialMol& oldMol, uint molIndex)
   {  //  printf("DCsphere old\n");
      XYZArray& positions = data->positions;
      uint nLJTrials = data->nLJTrialsNth;
      data->prng.FillWithRandomOnSphere(positions, nLJTrials, bondLength,
					oldMol.AtomPosition(focus));
      positions.Set(0, oldMol.AtomPosition(atom));

      double* inter = data->inter;
      double stepWeight = 0;
      data->axes.WrapPBC(positions, oldMol.GetBox());
      std::fill_n(inter, nLJTrials, 0);

	  

     // data->calc.ParticleInter(atom, positions, inter, molIndex, oldMol.GetBox());
	  //data->calc.ParticleInter(inter, positions, atom, molIndex,                               oldMol.GetBox(), nLJTrials);

      //data->calc.GetParticleEnergyGPU(oldMol.GetBox(),  inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex);
	   data->calc.GetParticleEnergy(oldMol.GetBox(),  inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex,nLJTrials);
	     //printf("DC of spherrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\n");
	  /*  for (int trial = 0; trial < nLJTrials; ++trial)
	   {
	   printf("serial Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }



	   printf("\n\n");
	  data->calc.GetParticleEnergyGPU(oldMol.GetBox(), nLJTrials, inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex);

	  for (int trial = 0; trial < nLJTrials; ++trial)
	   {
	   printf("GPU Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }


	  printf("===================\n\n");*/




      for (uint trial = 0; trial < nLJTrials; trial++)
      {
         stepWeight += exp(-1 * data->ff.beta * inter[trial]);
      }
      oldMol.MultWeight(stepWeight);
      oldMol.AddEnergy(Energy(0, 0, inter[0]));
      oldMol.ConfirmOldAtom(atom);
   }

   void DCOnSphere::BuildNew(TrialMol& newMol, uint molIndex)
   { //printf("DC of spherrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\n");
	   // printf("DCsphere new\n");

       XYZArray& positions = data->positions;
      uint nLJTrials = data->nLJTrialsNth;
      data->prng.FillWithRandomOnSphere(positions, nLJTrials, bondLength,
					newMol.AtomPosition(focus));

      double* inter = data->inter;
      double* ljWeights = data->ljWeights;
      double stepWeight = 0;
      data->axes.WrapPBC(positions, newMol.GetBox());
      std::fill_n(inter, nLJTrials, 0);

      //data->calc.ParticleInter(atom, positions, inter, molIndex, newMol.GetBox());
	  //data->calc.ParticleInter(inter, positions, atom, molIndex,                            newMol.GetBox(), nLJTrials);


	   //data->calc.GetParticleEnergyGPU(newMol.GetBox(),inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex);
	     data->calc.GetParticleEnergy(newMol.GetBox(),inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex,nLJTrials);
	   /* for (int trial = 0; trial < nLJTrials; ++trial)
	   {
	   printf("serial Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }



	   printf("\n\n");
	  data->calc.GetParticleEnergyGPU(newMol.GetBox(), nLJTrials, inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex);

	  for (int trial = 0; trial < nLJTrials; ++trial)
	   {
	   printf("GPU Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }


	  printf("===================\n\n");*/




      for (uint trial = 0; trial < nLJTrials; trial++)
      {
         ljWeights[trial] = exp(-1 * data->ff.beta * inter[trial]);
         stepWeight += ljWeights[trial];
      }
      uint winner = data->prng.PickWeighted(ljWeights, nLJTrials, stepWeight);
      newMol.MultWeight(stepWeight);
      newMol.AddEnergy(Energy(0, 0, inter[winner]));
      newMol.AddAtom(atom, positions[winner]);
   }   
}


