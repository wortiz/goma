/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2022 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/
/* File containing the many of Goma's global variables, moved for -fno-common */

#include "ac_particles.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_node_const.h"
#include "rf_solver.h"

int CoordinateSystem; /* Indicates type of coordinate system (see fem_const.h)*/

/* FEM Interpolation Parameters (see fem_const.h) */

int VelocityPressure; /* Indicates which element type is used             */
/* for velocity and pressure interpolation.         */
int Velocity; /* Indicates which type of interpolation is used    */
/* for velocity. (set by value of VelocityPressure) */
int Pressure; /* Indicates which type of interpolation is used    */
/* for pressure. (set by value of VelocityPressure) */
int Temperature; /* Indicates which type of interpolation is used    */
/* for temperature.                                 */
int MeshDisplacement; /* Indicates which element type is used             */
/* for mesh displacement interpolation.             */
int MassFraction; /* Indicates which type of interpolation is used    */
/* for mass fraction and density.                   */
int nEQM; /* Indicates one or more element quality metrics    */
/* are to be performed.				    */
int Use_DG; /* Indicates when Discontinuous Galerkin	    */
/* inpterpolation is in use.			    */
int Do_Overlap; /* Indicates that Overlap AC's will be used         */

/* Parameters to select Problem type (see fem_const.h)*/

int ProblemType,     /* Select type of problem to be solved		    */
    ProblemCoupling, /* Select fully coupled or dilute solution method   */
    StateEq,         /* Select equation of state                         */
    Multicomponent;  /* Select fomulation for multicomponent transport   */

/* global variable to account for extra terms in axisymmetric or swirling
   flow problems: define in setup_pd */
int VIM;
int WIM;

/* Boundary Condition information */

int Num_BC; /* number of boundary conditions which are defined  */

int Num_Interface_Srcs; /* number *_D interfaces*/
int IntSrc_BCID[MAX_INTERFACE];
/* Rotation information */
int Num_ROT; /* number of rotations which are defined  */

/* Import & Export field counts (used in coupled mode only) */
int Num_Import_NV;                    /* number of nodal vars to import */
int Num_Import_EV;                    /* number of nodal vars to import */
int Num_Export_XS;                    /* number of solution vars to export */
int Num_Export_XP;                    /* number of post-proc vars to export */
int Export_XS_ID[MAX_EXTERNAL_FIELD]; /* ID's of solution vars to export */
int Export_XP_ID[MAX_EXTERNAL_FIELD]; /* ID's of post proc vars to export */

/*
 * How many unique kinds of basis functions do we need to set up?
 * Some will be needed as Galerkin weights, some are used to interpolate
 * variables, and the elemental Jacobian matrix transforming the global
 * integral into local coordinates will use some kind of basis function.
 * For moving mesh problems, this will be the same as the interpolation
 * function for mesh displacement unknowns...
 *
 * At any rate, to economize the memory allocation, try to count up
 * from the input file specification exactly how many different basis
 * functions will be used.
 *
 * eg.; 2D fluid flow w/ mesh displacement
 *           v - Q2 (for momentum weight and for velocity interpolation)
 *           P - P1 (for continuity weight and for pressure interpolation)
 *           d - Q2 (for mesh stress weight and for displacement interpolation)
 *
 * Total: 2 unique_kinds of basis_functions
 */

int Num_Basis_Functions;
int Unique_Basis_Functions[MAX_BASIS_FUNCTIONS];

/*
 * Count up unique element types read in from the EXODUS II database, where
 * each element block has an element type associated with it.
 */

int Num_Element_Types;
int Unique_Element_Types[MAX_ELEMENT_TYPES];

/*
 * This is all very confusing...but...there really are basis function shapes
 * and interpolations. Together, they form a distinct basis function type.
 *
 *
 */

int Num_Interpolations;
int Unique_Interpolations[MAX_INTERPOLATIONS];
int Highest_Interpolation;

int Num_Shapes;
int Unique_Shapes[MAX_ELEMENT_SHAPES];

/* Parameters to select time integration technique                           */
int TimeIntegration; /* Select time integration method                    */
int Use_Level_Set;   /* Global switch to turn on level set computations   */
int Use_Phase_Field; /* Global switch to turn on phase-field computations   */

/* double  theta;  */ /* Time step parameter: theta = 0. => Backward Euler
                                              theta = 1. => Forward  Euler */

double eps; /* Time step error                                   */
int print_freq;
double print_delt;
double print_delt2_time, print_delt2;

/* Parameters for continuation */
int Continuation;
int ContType;
int BdyCondID;
int DataFltID;
int MatID;
int MatPropID;
int MatPropSIID;
int MaxPathSteps;
double Delta_s0;
double Delta_s_min;
double Delta_s_max;
double PathMax;

double print_delt2_path;
double BegParameterValue;
double EndParameterValue;

/* Parameters for augmenting conditions */
int nAC;

/* Parameters for multiple continuation conditions */
int nCC, nTC, nUC, nUTC;

int nnz_own; /* number of nonzeroes in the matrix that
              * belong to this processor
              */

int nnz_total; /* total number of nonzeroes in the matrix
                * that this processor sees, including the
                * entries for external unknowns that will
                * not be updated by this processor  */

/* Parameters for hunting conditions */
int nHC;

int PRS_mat_ielem;

struct Boundary_Condition *BC_Types;
struct Rotation_Specs *ROT_Types;

NODE_INFO_STRUCT **Nodes = NULL;

struct elem_side_bc_struct ***First_Elem_Side_BC_Array;
struct elem_edge_bc_struct ***First_Elem_Edge_BC_Array;
