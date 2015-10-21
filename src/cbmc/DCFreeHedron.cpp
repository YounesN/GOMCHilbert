/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 1.0 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#define _USE_MATH_DEFINES
#include <math.h>
#include "DCFreeHedron.h"
#include "DCData.h"
#include "TrialMol.h"
#include "../MolSetup.h"
#include "../Forcefield.h"
#include "../PRNG.h"

namespace cbmc {

 DCFreeHedron::DCFreeHedron(DCData* data, const mol_setup::MolKind& kind,uint focus, uint prev)
	  : data(data), seed(data, focus), hed(data, kind, focus, prev)
   {
         using namespace mol_setup;
         using namespace std;
         vector<Bond> onFocus = AtomBonds(kind, hed.Focus());
         for(uint i = 0; i < onFocus.size(); ++i) {
            if (onFocus[i].a1 == prev) {
               anchorBond = data->ff.bonds.Length(onFocus[i].kind);
               break;
            }
         }
   }


   void DCFreeHedron::PrepareNew()
   {
      hed.PrepareNew();
   }

   void DCFreeHedron::PrepareOld()
   {
      hed.PrepareOld();
   }


   void DCFreeHedron::BuildNew(TrialMol& newMol, uint molIndex)
   {  //printf("DCFREEHedron new\n");
      seed.BuildNew(newMol, molIndex);

      PRNG& prng = data->prng;
      const CalculateEnergy& calc = data->calc;
      const Forcefield& ff = data->ff;
       uint nLJTrials = data->nLJTrialsNth;
      double* ljWeights = data->ljWeights;
      double* inter = data->inter;

      //get info about existing geometry
      newMol.ShiftBasis(hed.Focus());
      const XYZ center = newMol.AtomPosition(hed.Focus());
      XYZArray* positions = data->multiPositions;

	  


	  //  printf("hedr=%d\n",hed.NumBond() );

      for (uint i = 0; i < hed.NumBond(); ++i)
      {
         positions[i].Set(0, newMol.RawRectCoords(hed.BondLength(i),hed.Theta(i), hed.Phi(i)));
		//  printf("%f,%f,%f\n",hed.BondLength(i),hed.Theta(i), hed.Phi(i));
      }

	 // exit(0);
      //add anchor atom
      positions[hed.NumBond()].Set(0, newMol.RawRectCoords(anchorBond, 0, 0));

	 

	   /*for (uint b = 0; b < hed.NumBond() + 1; ++b)
	   for (int i=0;i<positions[b].count;i++ )
		 printf("pos %d (%f,%f,%f)\n", b, positions[b].x[i],positions[b].y[i],positions[b].z[i] );

	   exit(0);*/

 /*for (uint b = 0; b < hed.NumBond()+1 ; ++b)
	   for (int i=0;i<positions[b].count;i++ )
		 printf("pos %d (%f,%f,%f)\n", b, positions[b].x[i],positions[b].y[i],positions[b].z[i] );

	   exit(0);*/


for (uint lj = nLJTrials; lj-- > 0;)
      {

		 /* double r1=prng();
			  double r2=prng();
			  double r3=prng();*/

			 // printf("%f,%f,%f\n", r1,r2,r3);
         //convert chosen torsion to 3D positions
         RotationMatrix spin =
            RotationMatrix::UniformRandom(prng(), prng(), prng());



         for (uint b = 0; b < hed.NumBond() + 1; ++b)
         {
               //find positions
               positions[b].Set(lj, spin.Apply(positions[b][0]));
               positions[b].Add(lj, center);

			  

         }
      }






      for (uint b = 0; b < hed.NumBond() + 1; ++b)
      { 
		
         data->axes.WrapPBC(positions[b], newMol.GetBox());
         /*for (int i=0;i<positions[b].count;i++ )
		 printf("pos %d (%f,%f,%f)\n", b, positions[b].x[i],positions[b].y[i],positions[b].z[i] );*/
		
      }
	//  exit(0);
//for (uint b = 0; b < hed.NumBond()+1 ; ++b)
//	   for (int i=0;i<positions[b].count;i++ )
//		 printf("pos %d (%f,%f,%f)\n", b, positions[b].x[i],positions[b].y[i],positions[b].z[i] );
//
// exit(0);
//
      std::fill_n(inter, nLJTrials, 0.0);
      for (uint b = 0; b < hed.NumBond(); ++b)
      {
         // calc.ParticleInter(inter, positions[b], hed.Bonded(b), 			    molIndex, newMol.GetBox(), nLJTrials);
		// data->calc.GetParticleEnergyGPU(newMol.GetBox(),  inter,positions[b], newMol.molLength, newMol.mOff, hed.Bonded(b),newMol.molKindIndex);// GPU call 
		  data->calc.GetParticleEnergy(newMol.GetBox(),  inter,positions[b], newMol.molLength, newMol.mOff, hed.Bonded(b),newMol.molKindIndex,nLJTrials);// GPU call cell list

		//   printf("energy of bond %d=%f\n", b, inter[b]);

		    //printf("DCFree Hedrooooooooooooooooooooooooooooooon\n");
		  // printf("box =%d, trials=%d, mol length=%d, mol kind index=%d\n",newMol.GetBox(), nLJTrials, newMol.molLength,newMol.molKindIndex);

		  // printf("bonded = %d, moff=%d\n",hed.Bonded(b), newMol.mOff);

      }

      //calc.ParticleInter(inter, positions[hed.NumBond()], hed.Prev(),           molIndex, newMol.GetBox(), nLJTrials);
 /*for (uint lj = 0; lj < nLJTrials; ++lj)
	  {   printf("GPU Trial %d energy=%f\n",lj,inter[lj] );

		 }*/
	   // data->calc.GetParticleEnergyGPU(newMol.GetBox(), inter,positions[hed.NumBond()], newMol.molLength, newMol.mOff,hed.Prev(),newMol.molKindIndex);// GPU call 
		data->calc.GetParticleEnergy(newMol.GetBox(), inter,positions[hed.NumBond()], newMol.molLength, newMol.mOff,hed.Prev(),newMol.molKindIndex,nLJTrials);// GPU call cell list
		


		// printf("box =%d, trials=%d, mol length=%d, mol kind index=%d\n",newMol.GetBox(), nLJTrials, newMol.molLength,newMol.molKindIndex);

		  //printf("bonded = %d, moff=%d\n",hed.Prev(), newMol.mOff);
	//	exit(0);



      double stepWeight = 0;
      for (uint lj = 0; lj < nLJTrials; ++lj)
	  {  // printf("GPU Trial %d energy=%f\n",lj,inter[lj] );
         ljWeights[lj] = exp(-ff.beta * inter[lj]);
         stepWeight += ljWeights[lj];
      }
       uint winner = prng.PickWeighted(ljWeights, nLJTrials, stepWeight);
      for(uint b = 0; b < hed.NumBond(); ++b){
         newMol.AddAtom(hed.Bonded(b), positions[b][winner]);
      }
      newMol.AddAtom(hed.Prev(), positions[hed.NumBond()][winner]);
      newMol.AddEnergy(Energy(hed.GetEnergy(), 0, inter[winner]));
      newMol.MultWeight(hed.GetWeight());
      newMol.MultWeight(stepWeight);
   }

   void DCFreeHedron::BuildOld(TrialMol& oldMol, uint molIndex)
   {   //printf("DCFREEHedron old\n");
      seed.BuildOld(oldMol, molIndex);
      PRNG& prng = data->prng;
      const CalculateEnergy& calc = data->calc;
      const Forcefield& ff = data->ff;
      uint nLJTrials = data->nLJTrialsNth;
      double* ljWeights = data->ljWeights;
      double* inter = data->inter;

      //get info about existing geometry
      oldMol.SetBasis(hed.Focus(), hed.Prev());
    //Calculate OldMol Bond Energy &
      //Calculate phi weight for nTrials using actual theta of OldMol
      hed.ConstrainedAnglesOld(data->nAngleTrials - 1, oldMol);
      const XYZ center = oldMol.AtomPosition(hed.Focus());
      XYZArray* positions = data->multiPositions;
      double prevPhi[MAX_BONDS];
       for (uint i = 0; i < hed.NumBond(); ++i) {
         //get position and shift to origin
         positions[i].Set(0, oldMol.AtomPosition(hed.Bonded(i)));
		  data->axes.UnwrapPBC(positions[i], 0, 1, oldMol.GetBox(), center);
         positions[i].Add(0, -center);
         
      }
      //add anchor atom
       positions[hed.NumBond()].Set(0, oldMol.AtomPosition(hed.Prev()));
      data->axes.UnwrapPBC(positions[hed.NumBond()], 0, 1,
			   oldMol.GetBox(), center);
      positions[hed.NumBond()].Add(0, -center);


      //counting backward to preserve prototype
      for (uint lj = nLJTrials; lj-- > 1;) {
         //convert chosen torsion to 3D positions
          //convert chosen torsion to 3D positions
         RotationMatrix spin =
            RotationMatrix::UniformRandom(prng(), prng(), prng());
         for (uint b = 0; b < hed.NumBond() + 1; ++b)
         {
            //find positions
            positions[b].Set(lj, spin.Apply(positions[b][0]));
            positions[b].Add(lj, center);
         }
      }

      for (uint b = 0; b < hed.NumBond() + 1; ++b)
      {
         positions[b].Add(0, center);
         data->axes.WrapPBC(positions[b], oldMol.GetBox());
      }

      std::fill_n(inter, nLJTrials, 0.0);
       for (uint b = 0; b < hed.NumBond(); ++b)
      {
         // calc.ParticleInter(inter, positions[b], hed.Bonded(b),              molIndex, oldMol.GetBox(), nLJTrials);
		   //data->calc.GetParticleEnergyGPU(oldMol.GetBox(),  inter,positions[b], oldMol.molLength, oldMol.mOff, hed.Bonded(b),oldMol.molKindIndex);// GPU call
		  data->calc.GetParticleEnergy(oldMol.GetBox(),  inter,positions[b], oldMol.molLength, oldMol.mOff, hed.Bonded(b),oldMol.molKindIndex,nLJTrials);// GPU call cell 
      }
      double stepWeight = 0;
      // calc.ParticleInter(inter, positions[hed.NumBond()], hed.Prev(),                 molIndex, oldMol.GetBox(), nLJTrials);
	 // data->calc.GetParticleEnergyGPU(oldMol.GetBox(),  inter,positions[hed.NumBond()], oldMol.molLength, oldMol.mOff,hed.Prev(),oldMol.molKindIndex);// GPU call 

	 data->calc.GetParticleEnergy(oldMol.GetBox(),  inter,positions[hed.NumBond()], oldMol.molLength, oldMol.mOff,hed.Prev(),oldMol.molKindIndex,nLJTrials);// GPU call cell
      for (uint lj = 0; lj < nLJTrials; ++lj) {
		  // printf("GPU Trial %d energy=%f, box=%d\n",lj,inter[lj], oldMol.GetBox() );

         stepWeight += exp(-ff.beta * inter[lj]);
      }
       for(uint b = 0; b < hed.NumBond(); ++b) {
         oldMol.ConfirmOldAtom(hed.Bonded(b));
      }
      oldMol.ConfirmOldAtom(hed.Prev());
      oldMol.AddEnergy(Energy(hed.GetEnergy(), 0, inter[0]));
      oldMol.MultWeight(hed.GetWeight());
      oldMol.MultWeight(stepWeight);
   }


}

