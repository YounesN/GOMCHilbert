/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 1.0 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#include "DCLinkNoDih.h"
#include "TrialMol.h"
#include "../Forcefield.h"
#include "../XYZArray.h"
#include "../MoleculeKind.h"
#include "../MolSetup.h"


namespace cbmc


{
   DCLinkNoDih::DCLinkNoDih(DCData* data, const mol_setup::MolKind kind,
			    uint atom, uint focus)
     : data(data), atom(atom), focus(focus)
   {
      using namespace mol_setup;
      std::vector<Bond> bonds = AtomBonds(kind, atom);
      for (uint i = 0; i < bonds.size(); ++i)
      {
         if (bonds[i].a0 == focus || bonds[i].a1 == focus)
	 {
            bondLength = data->ff.bonds.Length(bonds[i].kind);
            break;
         }
      }
      std::vector<Angle> angles = AtomEndAngles(kind, atom);
      for (uint i = 0; i < angles.size(); ++i)
      {
         if (angles[i].a1 == focus)
	 {
            prev = angles[i].a2;
            angleKind = angles[i].kind;
            break;
         }
      }
   }

   void DCLinkNoDih::PrepareNew()
   {
      double* angles = data->angles;
      double* angleEnergy = data->angleEnergy;
      double* angleWeights = data->angleWeights;
      PRNG& prng = data->prng;
      const Forcefield& ff = data->ff;
      uint count = data->nAngleTrials;
      bendWeight = 0;
      for (uint trial = 0; trial < count; trial++)
      {


         angles[trial] = prng.rand(M_PI);
         angleEnergy[trial] = ff.angles->Calc(angleKind, angles[trial]);
         angleWeights[trial] = exp(angleEnergy[trial] * -ff.beta);

         bendWeight += angleWeights[trial];
      }
      uint winner = prng.PickWeighted(angleWeights, count, bendWeight);
      theta = angles[winner];
      bendEnergy = angleEnergy[winner];
   }

   void DCLinkNoDih::PrepareOld()
   {
      PRNG& prng = data->prng;
      const Forcefield& ff = data->ff;
      uint count = data->nAngleTrials - 1;
      bendWeight = 0;
      for (uint trial = 0; trial < count; trial++)
      {



         double trialAngle = prng.rand(M_PI);
         double trialEn = ff.angles->Calc(angleKind, trialAngle);
         double trialWeight = exp(-ff.beta * trialEn);

         bendWeight += trialWeight;
      }
   }

   void DCLinkNoDih::IncorporateOld(TrialMol& oldMol)
   {
      double dummy;
      oldMol.OldThetaAndPhi(atom, focus, theta, dummy);
      const Forcefield& ff = data->ff;
      bendEnergy = ff.angles->Calc(angleKind, theta);
      bendWeight += exp(-ff.beta * bendEnergy);
   }

   void DCLinkNoDih::AlignBasis(TrialMol& mol)
   {
      mol.SetBasis(focus, prev);
   }

   void DCLinkNoDih::BuildOld(TrialMol& oldMol, uint molIndex)
   {//printf("DCLinkNoDeh old\n");
      AlignBasis(oldMol);
      IncorporateOld(oldMol);
	   double* nonbonded_1_4 = data->nonbonded_1_4;// v1
      double* inter = data->inter;
      uint nLJTrials = data->nLJTrialsNth;
      XYZArray& positions = data->positions;
      PRNG& prng = data->prng;
      positions.Set(0, oldMol.AtomPosition(atom));
      for (uint trial = 1, count = nLJTrials; trial < count; ++trial)
      {
         double phi = prng.rand(M_PI * 2);
         positions.Set(trial, oldMol.GetRectCoords(bondLength, theta, phi));
      }

      data->axes.WrapPBC(positions, oldMol.GetBox());
      std::fill_n(inter, nLJTrials, 0.0);
	   std::fill_n(nonbonded_1_4, nLJTrials, 0.0);//v1

      //data->calc.ParticleInter(inter, positions, atom, molIndex,                         oldMol.GetBox(), nLJTrials);

	   //data->calc.GetParticleEnergyGPU(oldMol.GetBox(),  inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex);
	    data->calc.GetParticleEnergy(oldMol.GetBox(),  inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex,nLJTrials);
	      //printf("DC Linked No dihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh\n");
	  /* for (int trial = 0; trial < data->nLJTrials; ++trial)
	   {
	   printf("serial Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }



	   printf("\n\n");
	  data->calc.GetParticleEnergyGPU(oldMol.GetBox(), data->nLJTrials, inter,positions, oldMol.molLength, oldMol.mOff, atom,oldMol.molKindIndex);

	  for (int trial = 0; trial < data->nLJTrials; ++trial)
	   {
	   printf("GPU Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }


	  printf("===================\n\n");*/




		data->calc.ParticleNonbonded_1_4(nonbonded_1_4, oldMol, positions, atom,
				   oldMol.GetBox(), nLJTrials);// v1

	     double stepWeight = 0;
      for (uint trial = 0, count = nLJTrials; trial < count; ++trial)
      {
        stepWeight += exp(-data->ff.beta * (inter[trial] +
					    nonbonded_1_4[trial]));//v1
      }
      oldMol.MultWeight(stepWeight * bendWeight);
      oldMol.ConfirmOldAtom(atom);
      oldMol.AddEnergy(Energy(bendEnergy, nonbonded_1_4[0], inter[0]));//v1

   }

   void DCLinkNoDih::BuildNew(TrialMol& newMol, uint molIndex)
   {//printf("DC Linked No dihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh\n");

	//  printf("DCLinkNoDeh new\n");

      AlignBasis(newMol);
      double* ljWeights = data->ljWeights;
	   double* nonbonded_1_4 = data->nonbonded_1_4;// v1

      double* inter = data->inter;
      uint nLJTrials = data->nLJTrialsNth;
      XYZArray& positions = data->positions;
      PRNG& prng = data->prng;

      for (uint trial = 0, count = nLJTrials; trial < count; ++trial)
      {
         double phi = prng.rand(M_PI * 2);
         positions.Set(trial, newMol.GetRectCoords(bondLength, theta, phi));
      }

      data->axes.WrapPBC(positions, newMol.GetBox());
      std::fill_n(inter, nLJTrials, 0);
	   std::fill_n(nonbonded_1_4, nLJTrials, 0);// v1


     // data->calc.ParticleInter(inter, positions, atom, molIndex,                         newMol.GetBox(), nLJTrials);
      
	 // data->calc.GetParticleEnergyGPU(newMol.GetBox(),  inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex);
	    data->calc.GetParticleEnergy(newMol.GetBox(),  inter,positions, newMol.molLength, newMol.mOff, atom,newMol.molKindIndex,nLJTrials);
	 /*   for (int trial = 0; trial < data->nLJTrials; ++trial)
	   {
	   printf("serial Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }



	   printf("\n\n");
	 

	  for (int trial = 0; trial < data->nLJTrials; ++trial)
	   {
	   printf("GPU Trial %d energy=%f\n",trial,inter[trial] );
	   
	   }


	  printf("===================\n\n");*/

		data->calc.ParticleNonbonded_1_4(nonbonded_1_4, newMol, positions, atom,
				   newMol.GetBox(), nLJTrials);// v1

      double stepWeight = 0;
      double beta = data->ff.beta;
      for (uint trial = 0, count = nLJTrials; trial < count; ++trial)
      {
       ljWeights[trial] = exp(-data->ff.beta * (inter[trial] +
						 nonbonded_1_4[trial]));// v1

         stepWeight += ljWeights[trial];
      }

      uint winner = prng.PickWeighted(ljWeights, nLJTrials, stepWeight);
      newMol.MultWeight(stepWeight * bendWeight);
      newMol.AddAtom(atom, positions[winner]);
       newMol.AddEnergy(Energy(bendEnergy, nonbonded_1_4[winner],
			      inter[winner]));// v1

   }

}

