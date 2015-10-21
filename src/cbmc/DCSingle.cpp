/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 1.0 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#include "DCSingle.h"
#include "TrialMol.h"
#include "DCData.h"
#include "../PRNG.h"
#include "../CalculateEnergy.h"
#include "../XYZArray.h"
#include "../Forcefield.h"

namespace cbmc
{

   void DCSingle::BuildOld(TrialMol& oldMol, uint molIndex)
   {  // printf("DCsingle old\n");
      PRNG& prng = data->prng;
      XYZArray& positions = data->positions;
      uint nLJTrials = data->nLJTrialsFirst;

      prng.FillWithRandom(data->positions, nLJTrials,
			  data->axes.GetAxis(oldMol.GetBox()));
      positions.Set(0, oldMol.AtomPosition(atom));

      double* inter = data->inter;
      double stepWeight = 0;
      std::fill_n(inter, nLJTrials, 0);
     // data->calc.ParticleInter(atom, positions, inter, molIndex, oldMol.GetBox());
	  // data->calc.ParticleInter(inter, positions, atom, molIndex,                               oldMol.GetBox(), nLJTrials);

	 //  data->calc.GetParticleEnergyGPU(oldMol.GetBox(), inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex);
	    data->calc.GetParticleEnergy(oldMol.GetBox(), inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex,nLJTrials);
	      //  printf("DC singleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n");
	   /*for (int trial = 0; trial < nLJTrials; ++trial)
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

      for (uint trial = 0; trial < nLJTrials; ++trial)
      { // printf("GPU Trial %d energy=%f\n",trial,inter[trial] );
         stepWeight += exp(-1 * data->ff.beta * inter[trial]);
      }
      oldMol.MultWeight(stepWeight);
      oldMol.AddEnergy(Energy(0, 0, inter[0]));
      oldMol.ConfirmOldAtom(atom);
   }

   void DCSingle::BuildNew(TrialMol& newMol, uint molIndex)
   { //printf("DC singleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n");
	   // printf("DCsingle new\n");


      PRNG& prng = data->prng;
      XYZArray& positions = data->positions;
      uint nLJTrials = data->nLJTrialsFirst;
      double* inter = data->inter;
      double* ljWeights = data->ljWeights;

      prng.FillWithRandom(positions, nLJTrials,
			  data->axes.GetAxis(newMol.GetBox()));
      std::fill_n(inter, nLJTrials, 0);


	  //data->calc.ParticleInter(inter, positions, atom, molIndex,                               newMol.GetBox(), nLJTrials);
	 //   data->calc.GetParticleEnergyGPU(newMol.GetBox(),  inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex);
	  data->calc.GetParticleEnergy(newMol.GetBox(),  inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex,nLJTrials);


	  /*printf(" Energy for trials:\n==================================\n");
      data->calc.ParticleInter(atom, positions, inter, molIndex, newMol.GetBox());
	    printf("End Energy for trials:\n==================================\n");
	  for (int trial = 0; trial < nLJTrials; ++trial)
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




    double stepWeight = 0;
      for (uint trial = 0; trial < nLJTrials; ++trial)
      {  // printf("GPU Trial %d energy=%f\n",trial,inter[trial] );
         ljWeights[trial] = exp(-1 * data->ff.beta * inter[trial]);
         stepWeight += ljWeights[trial];
      }
      uint winner = prng.PickWeighted(ljWeights, nLJTrials, stepWeight);
      newMol.MultWeight(stepWeight);
      newMol.AddEnergy(Energy(0, 0, inter[winner]));

	 // printf("gpu trial winner=%f\n", inter[winner]);
      newMol.AddAtom(atom, positions[winner]);
	    //printf("DCsingle new done \n");

   }
}

