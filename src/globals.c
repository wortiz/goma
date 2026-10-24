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

#include <limits.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "ac_particles.h"
#include "dp_types.h"
#include "dpi.h"
#include "exo_struct.h"
#include "goma_reset.h"
#include "mm_as.h"
#include "mm_elem_block_structs.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_io_const.h"
#include "rf_mp.h"
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

/* Parameters for hunting conditions */
String_line Matrix_Solver;

String_line Matrix_Format;

String_line Matrix_Scaling;

String_line Matrix_Preconditioner;

String_line Matrix_Subdomain_Solver; /* new Aztec 2.x option */

String_line Matrix_Residual_Norm_Type;

String_line Matrix_Output_Type;

String_line Matrix_Factorization_Reuse;

String_line Matrix_Graph_Fillin; /* new Aztec 2.x option */

String_line Matrix_Maximum_Iterations;

String_line Matrix_Polynomial_Order;

String_line Matrix_Factor_Overlap;

String_line Matrix_Overlap_Type;

String_line Matrix_Krylov_Subspace;

String_line Matrix_Orthogonalization;

String_line Matrix_Auxiliary_Vector;

String_line Matrix_Convergence_Tolerance;

String_line Matrix_Drop_Tolerance;

String_line Matrix_Factorization_Save; /* Aztec 2 */

String_line Matrix_ILUT_Fill_Factor; /* Aztec 2 */

String_line Matrix_RILU_Relax_Factor; /* Aztec 2 */

String_line Matrix_BILU_Threshold; /* Trilinos 1 */

String_line Matrix_Relative_Threshold; /* Trilinos 2 */

String_line Matrix_Absolute_Threshold; /* Trilinos 2 */

String_line Amesos_Package;

String_line Amesos2_Package;

String_line AztecOO_Solver;

String_line Stratimikos_File[MAX_NUM_MATRICES];

String_line Amesos2_File[MAX_NUM_MATRICES];

/*
 * A new Aztec 2.0 option. There are more and difft options and our
 * previous options probably ought to be revised to reflect the newer
 * Aztec 2.0 capability specifications.
 */

String_line Matrix_Reorder;

int Linear_Solver; /* Aztec, Sparse, MA28, UMFPACK */

int UMFPACK_IDIM; /* UMFPACK STORAGE CONSTANT */
int UMFPACK_XDIM; /* UMFPACK STORAGE CONSTANT */
int LOCA_UMF_ID;  /* UMFPACK SYSTEM ID */

int Max_Newton_Steps;  /* Maximum number of Newton steps to take.     */
int Guess_Flag;        /* Indicates the type of initial guess         */
int Conformation_Flag; /* Indicates mapping from stress to log-conformation tensor */
int Print3DBCDup;

double damp_factor;
double damp_factor1; /* Relaxation factor for Newton iteration */
/* damp_factor1 = 1.0 is full Newton */
/* damp_factor1 = 0.0 is not updating our */
/*                   solution estimate */
double damp_factor2, /* Additional damping factors for custom */
    damp_factor3,    /* schemes for automatic control with    */
    custom_tol1,     /* NORM(0,0) tolerances                  */
    custom_tol2, custom_tol3;
double var_damp[MAX_VARIABLE_TYPES]; /* variable specific damp factors */

int Newt_Jacobian_Reformation_stride; /*Stride for reformation of jacobian for
                                   modified newton scheme               */
int Time_Jacobian_Reformation_stride;
int Newton_Line_Search_Type;
double Line_Search_Minimum_Damping;
int modified_newton;               /*boolean flag for modified Newton */
int save_old_A;                    /*boolean flag for saving old A matrix
                                    for resolve reasons with AZTEC.   There
                                    are at least four reasons, that you
                                    can see in sl_util.c */
double convergence_rate_tolerance; /* tolerance for jacobian reformation
                                       based on convergence rate */
double modified_newt_norm_tol;     /* tolerance for jacobian reformation
                                           based on residual norm */

double Epsilon[MAX_NUM_MATRICES][3]; /* Used for determining stopping criteria.     */
int Solver_Output_Format;            /* Bitmap for Solver Output Format     */
int Output_Variable_Stats;           /* Toggle for Variable Stats Output    */
int Output_Variable_Regression;      /* Toggle for Variable Regression    */

int NZeros; /* Number of nonzeros in this procs matrix     */

int nnz_own; /* number of nonzeroes in the matrix that
              * belong to this processor
              */

int nnz_total; /* total number of nonzeroes in the matrix
                * that this processor sees, including the
                * entries for external unknowns that will
                * not be updated by this processor  */

int GNZeros; /* Number of nonzeros in global matrix         */

int fill_zeros; /* number of nonzeros in fill matrix for this
                  processor */

int Gfill_zeros; /* number of nonzeros in fill matrix for the
                   global problem */

int PSPG;          /* 1 means pressure stabilized Petrov-Galerkin is used */
int PSPP;          /* 1 means pressure stabilized polynomial projection is used */
double PS_scaling; /* This term is a constant scaling for the PSPG or PSPP term */
int Cont_GLS;      /* 1 means continuity stabilization is used */

int Filter_Species, filter_species_material_number;
double c_min, c_max;

int Include_Visc_Sens, Visc_Sens_Copy, Visc_Sens_Factor;
/* 1 means to include the sensitivities of the
viscosity functions in the jacobian matrix.
0 indicates that these sensitivities are not to
be included.  The latter case is useful for highly
shear thinning viscosity models.  By disabling
the viscosity sensititives convergence can be
achieved albeit at a less than quadratic rate
      */
int nHC;

int PRS_mat_ielem;

struct Boundary_Condition *BC_Types;
struct Rotation_Specs *ROT_Types;

NODE_INFO_STRUCT **Nodes = NULL;

struct elem_side_bc_struct ***First_Elem_Side_BC_Array;
struct elem_edge_bc_struct ***First_Elem_Edge_BC_Array;

/* Global variables extern declared in ac_particles.h. */
int Particle_Dynamics;                /* global toggle indicating particles are present. */
enum Particle_Model_t Particle_Model; /* What flavor of particle<->continuum stuff... */
dbl Particle_Model_Data[MAX_PARTICLE_MODEL_DATA_VALUES]; /* Real values for this model. */
int Particle_Number;                                     /* number of discrete particles. */
particle_filename_s Particle_Restart_Filename;           /* restart filename */
int Particle_Output_Stride;    /* How often to output particle information. */
dbl Particle_Output_Time_Step; /* Output every these units. */
int Particle_Max_Time_Steps;   /* max number of particle time steps if steady solution. */
enum Particle_Output_Format_t Particle_Output_Format; /* What kind of output file? */
int Particle_Full_Output_Stride;                   /* > 0 => full output every that many steps. */
particle_filename_s Particle_Full_Output_Filename; /* where to put them. */
int Particle_Number_Sample_Types;                  /* How many datasets to output? */
int *Particle_Number_Samples_Existing;             /* How many are tagged for each sample type?*/
int *Particle_Number_Samples;          /* How many particles to output for dataset #n? */
int *Particle_Number_Output_Variables; /* How many output vars for each sample. */
particle_variable_s *
    *Particle_Output_Variables; /* List of variable indices to output for dataset #n */
particle_filename_s *Particle_Filename_Template; /* Template of where to put the data... */

dbl Particle_Density;         /* Density of particle in problem units */
dbl Particle_Radius;          /* Radius of particle in problem units. */
dbl Particle_Ratio;           /* Real/computational particle ratio. */
int Particle_Show_Debug_Info; /* Show particle debug info. */
enum Particle_Domain_t Particle_Creation_Domain;
enum Particle_Domain_t Particle_Move_Domain;
particle_filename_s Particle_Creation_Domain_Filename;
particle_s Particle_Creation_Domain_Name;
particle_filename_s Particle_Move_Domain_Filename;
particle_s Particle_Move_Domain_Name;
dbl Particle_Creation_Domain_Reals[MAX_DOMAIN_REAL_VALUES];
dbl Particle_Move_Domain_Reals[MAX_DOMAIN_REAL_VALUES];
dbl xi_boundary_tolerances[3] = {XI_BOUNDARY_TOLERANCE0, XI_BOUNDARY_TOLERANCE1,
                                 XI_BOUNDARY_TOLERANCE2};

int Particle_Number_PBCs; /* Number of particle-related sideset BC's. */
PBC_t *PBCs;              /* Particle boundary condition structures. */

element_particle_info_t *element_particle_info;

/*
 * Globals formerly defined in main.c, moved here so that the goma library is
 * self-contained and can be driven through the goma.h API.
 */
/*
 * Global variables defined here.
 */

Dpi *DPI_ptr = NULL;

Exo_DB *EXO_ptr = NULL;

char Input_File[MAX_FNL] = "\0"; /* input EXODUS II database w/ problem defn */

char ExoFile[MAX_FNL] = "\0"; /* input EXODUS II database w/ problem defn */

char Exo_LB_File[MAX_FNL] = "\0"; /* EXODUS II load balance info for mesh */

char ExoFileOut[MAX_FNL] = "\0"; /* output EXODUS II database w/ results */

char ExoFileOutMono[MAX_FNL] = "\0"; /* output EXODUS II database without per proc identifier */

char ExoAuxFile[MAX_FNL] = "\0"; /* auxiliary EXODUS II database for initguess */

int ExoTimePlane = INT_MAX; /* Time plane # or continuation # of soln to use as an initguess */

char Echo_Input_File[MAX_FNL] = "\0"; /* echo of problem def file  */

int Decompose_Flag = 1;
int Decompose_Type = 0;
int Skip_Fix = 0;

char *GomaPetscOptions = NULL;
int GomaPetscOptionsStrLen = 0;

char DomainMappingFile[MAX_FNL] = "\0"; /* Domain Mapping file. Maps the materials
                                      and names of material boundaries
                                      specified in this file into this file
                                      into chemkin domains and chemkin
                                      surface and volumetric domains. */

int CPU_word_size;
int IO_word_size; /*Precision variables for exodus II files*/

char Init_GuessFile[MAX_FNL]; /* ASCII file holding initial guess */

char Soln_OutFile[MAX_FNL]; /* ASCII file holding solution vector, */
                            /* same format as Init_GuessFile      */

int Debug_Flag; /* Flag to specify debug info is to be     */
                /* printed out. The value of this flag     */
                /* determines the level of diagnostic info */
                /* which is printed to stdout              */
                /* Debug_Flag == 0 	No output          */
                /*	 1	minimun output             */
                /*	 2	medium  output		   */
                /*	 3	maximum output             */
                /*	 -1	check jacobian             */
                /*	 -2	check jacobian with scaling*/

int New_Parser_Flag; /* New_Parser_Flag = 0	Parse with old parser */
                     /* New_Parser_Flag = 1  Parse with new flex/bison parser */

#ifdef MATRIX_DUMP
int Number_Jac_Dump = 0; /* Number of jacobians to dump out
                          * If the value is negative, then the one
                          * jacobian, the -n'th jacobian, is dumped
                          * out */
#endif
int Iout; /* Flag to specify level of diagnostic output */
          /* which is to be printed out for the program */

int Write_Intermediate_Solutions = FALSE; /* Flag specifies whether to */
                                          /* write out solution data at each */
                                          /* Newton iteration. */
int Write_Initial_Solution = FALSE;
/* Flag to indicate whether to write the
 * initial solution to the ascii and exodus
 * output files */
int Num_Var_Init;         /* number of variables to overwrite with
                           * global initialization */
int Num_Var_Bound;        /* number of variables to bound  */
int Num_Var_LS_Init;      /* number of variables to overwirte with
                           * level set index initialization */
int Num_Var_External;     /* number of total external variables (exoII or pixel)*/
int Num_Var_External_pix; /* number of external variables (pixel only)*/
int Anneal_Mesh;          /* flag specifying creation of a special exodus
                           *  file with coordinates adjusted to the
                           * deformed coordinates (i.e. new displacements
                           * are set to zero and mesh is deemed stress-free */

double Porous_liq_inventory;     /*global variable for finite-insult boundary condition*/
double **Spec_source_inventory;  /*global variable for cumulative reacted source */
double *Spec_source_lumped_mass; /*global variable for species lumped mass */

const char anneal_file[] = ANNEAL_FILE_NAME;

/*
 * Benner's frontal solver wants to strcat() onto this directory name,
 * so leave enough space for dirname/lu.123456.0, for example.
 *
 * Here it is defined and initialized.
 */

char front_scratch_directory[MAX_FNL] = FRONT_SCRATCH_DIRECTORY;

/*
 * Variables defined here but everywhere else are extern via "rf_mp.h".
 */

int parallel_err = FALSE;
int parallel_err_global = FALSE;

int Unlimited_Output = TRUE; /* print limit flag */

int Num_Proc = 1;
int ProcID = 0;
int Dim = -1;

int unlerr;

MPI_Request *Request = NULL;
MPI_Status *Status = NULL;
MPI_Aint type_extent;
int Num_Requests = 6;

/*
 * Data structures for transporting input data to other processors...
 */

DDD Noahs_Raven; /* Stage 1 - preliminary sizing information */
DDD Noahs_Ark;   /* Stage 2 - big boatload - main body*/
DDD Noahs_Dove;  /* Stage 3 - doubly dynamic sized stuff */

Comm_Ex **cx = NULL; /* communications info for ea neighbor proc */

double time_goma_started; /* Save it here... */

/*
 * Declare these copies so that other routines may access them as global
 * external variables instead of passing them through the argument lists.
 */

char **Argv;

int Argc;

ELEM_BLK_STRUCT *Element_Blocks = NULL; /* Pointer to array of global
                                           element block information. */

ELEM_BLK_STRUCT *Current_EB_ptr = NULL; /* Pointer to the current element block
                                           structure. This is calculated in the
                                           fill */

/* ddd_free() without MPI_Type_free() */
static void free_ddd_members(DDD p) {
  if (p == NULL)
    return;
  free(p->block_count);
  free(p->data_type);
  free(p->address);
  free(p);
}

/*
 * Reset the globals defined in this file to their load time values, freeing
 * arrays owned by them that are not freed elsewhere.  Used by goma_destroy so
 * that Goma can be run again in the same process.  Keep in sync with the
 * definitions above.
 */
void globals_free_globals(void) {
  int i;

  /* not freed elsewhere, arrays inside BC_Types are not yet freed */
  free(BC_Types);
  if (cx != NULL && upd != NULL) {
    for (i = 0; i < upd->Total_Num_Matrices; i++) {
      free(cx[i]);
    }
  }
  free(cx);
  free(GomaPetscOptions);
  /* the MPI types were already freed after their broadcasts in goma_create */
  free_ddd_members(Noahs_Raven);
  free_ddd_members(Noahs_Ark);
  free_ddd_members(Noahs_Dove);

  CoordinateSystem = 0;
  VelocityPressure = 0;
  Velocity = 0;
  Pressure = 0;
  Temperature = 0;
  MeshDisplacement = 0;
  MassFraction = 0;
  nEQM = 0;
  Use_DG = 0;
  Do_Overlap = 0;
  ProblemType = 0;
  ProblemCoupling = 0;
  StateEq = 0;
  Multicomponent = 0;
  VIM = 0;
  WIM = 0;
  Num_BC = 0;
  Num_Interface_Srcs = 0;
  memset(IntSrc_BCID, 0, sizeof(IntSrc_BCID));
  Num_ROT = 0;
  Num_Import_NV = 0;
  Num_Import_EV = 0;
  Num_Export_XS = 0;
  Num_Export_XP = 0;
  memset(Export_XS_ID, 0, sizeof(Export_XS_ID));
  memset(Export_XP_ID, 0, sizeof(Export_XP_ID));
  Num_Basis_Functions = 0;
  memset(Unique_Basis_Functions, 0, sizeof(Unique_Basis_Functions));
  Num_Element_Types = 0;
  memset(Unique_Element_Types, 0, sizeof(Unique_Element_Types));
  Num_Interpolations = 0;
  memset(Unique_Interpolations, 0, sizeof(Unique_Interpolations));
  Highest_Interpolation = 0;
  Num_Shapes = 0;
  memset(Unique_Shapes, 0, sizeof(Unique_Shapes));
  TimeIntegration = 0;
  Use_Level_Set = 0;
  Use_Phase_Field = 0;
  eps = 0.0;
  print_freq = 0;
  print_delt = 0.0;
  print_delt2_time = 0.0;
  print_delt2 = 0.0;
  Continuation = 0;
  ContType = 0;
  BdyCondID = 0;
  DataFltID = 0;
  MatID = 0;
  MatPropID = 0;
  MatPropSIID = 0;
  MaxPathSteps = 0;
  Delta_s0 = 0.0;
  Delta_s_min = 0.0;
  Delta_s_max = 0.0;
  PathMax = 0.0;
  print_delt2_path = 0.0;
  BegParameterValue = 0.0;
  EndParameterValue = 0.0;
  nAC = 0;
  nCC = 0;
  nTC = 0;
  nUC = 0;
  nUTC = 0;
  memset(Matrix_Solver, 0, sizeof(String_line));
  memset(Matrix_Format, 0, sizeof(String_line));
  memset(Matrix_Scaling, 0, sizeof(String_line));
  memset(Matrix_Preconditioner, 0, sizeof(String_line));
  memset(Matrix_Subdomain_Solver, 0, sizeof(String_line));
  memset(Matrix_Residual_Norm_Type, 0, sizeof(String_line));
  memset(Matrix_Output_Type, 0, sizeof(String_line));
  memset(Matrix_Factorization_Reuse, 0, sizeof(String_line));
  memset(Matrix_Graph_Fillin, 0, sizeof(String_line));
  memset(Matrix_Maximum_Iterations, 0, sizeof(String_line));
  memset(Matrix_Polynomial_Order, 0, sizeof(String_line));
  memset(Matrix_Factor_Overlap, 0, sizeof(String_line));
  memset(Matrix_Overlap_Type, 0, sizeof(String_line));
  memset(Matrix_Krylov_Subspace, 0, sizeof(String_line));
  memset(Matrix_Orthogonalization, 0, sizeof(String_line));
  memset(Matrix_Auxiliary_Vector, 0, sizeof(String_line));
  memset(Matrix_Convergence_Tolerance, 0, sizeof(String_line));
  memset(Matrix_Drop_Tolerance, 0, sizeof(String_line));
  memset(Matrix_Factorization_Save, 0, sizeof(String_line));
  memset(Matrix_ILUT_Fill_Factor, 0, sizeof(String_line));
  memset(Matrix_RILU_Relax_Factor, 0, sizeof(String_line));
  memset(Matrix_BILU_Threshold, 0, sizeof(String_line));
  memset(Matrix_Relative_Threshold, 0, sizeof(String_line));
  memset(Matrix_Absolute_Threshold, 0, sizeof(String_line));
  memset(Amesos_Package, 0, sizeof(String_line));
  memset(Amesos2_Package, 0, sizeof(String_line));
  memset(AztecOO_Solver, 0, sizeof(String_line));
  memset(Stratimikos_File, 0, sizeof(Stratimikos_File));
  memset(Amesos2_File, 0, sizeof(Amesos2_File));
  memset(Matrix_Reorder, 0, sizeof(String_line));
  Linear_Solver = 0;
  UMFPACK_IDIM = 0;
  UMFPACK_XDIM = 0;
  LOCA_UMF_ID = 0;
  Max_Newton_Steps = 0;
  Guess_Flag = 0;
  Conformation_Flag = 0;
  Print3DBCDup = 0;
  damp_factor = 0.0;
  damp_factor1 = 0.0;
  damp_factor2 = 0.0;
  damp_factor3 = 0.0;
  custom_tol1 = 0.0;
  custom_tol2 = 0.0;
  custom_tol3 = 0.0;
  memset(var_damp, 0, sizeof(var_damp));
  Newt_Jacobian_Reformation_stride = 0;
  Time_Jacobian_Reformation_stride = 0;
  Newton_Line_Search_Type = 0;
  Line_Search_Minimum_Damping = 0.0;
  modified_newton = 0;
  save_old_A = 0;
  convergence_rate_tolerance = 0.0;
  modified_newt_norm_tol = 0.0;
  memset(Epsilon, 0, sizeof(Epsilon));
  Solver_Output_Format = 0;
  Output_Variable_Stats = 0;
  Output_Variable_Regression = 0;
  NZeros = 0;
  nnz_own = 0;
  nnz_total = 0;
  GNZeros = 0;
  fill_zeros = 0;
  Gfill_zeros = 0;
  PSPG = 0;
  PSPP = 0;
  PS_scaling = 0.0;
  Cont_GLS = 0;
  Filter_Species = 0;
  filter_species_material_number = 0;
  c_min = 0.0;
  c_max = 0.0;
  Include_Visc_Sens = 0;
  Visc_Sens_Copy = 0;
  Visc_Sens_Factor = 0;
  nHC = 0;
  PRS_mat_ielem = 0;
  BC_Types = NULL;
  ROT_Types = NULL;
  Nodes = NULL;
  First_Elem_Side_BC_Array = NULL;
  First_Elem_Edge_BC_Array = NULL;

  /* particle arrays are freed by the particle code */
  Particle_Dynamics = 0;
  Particle_Model = 0;
  memset(Particle_Model_Data, 0, sizeof(Particle_Model_Data));
  Particle_Number = 0;
  memset(Particle_Restart_Filename, 0, sizeof(particle_filename_s));
  Particle_Output_Stride = 0;
  Particle_Output_Time_Step = 0.0;
  Particle_Max_Time_Steps = 0;
  Particle_Output_Format = 0;
  Particle_Full_Output_Stride = 0;
  memset(Particle_Full_Output_Filename, 0, sizeof(particle_filename_s));
  Particle_Number_Sample_Types = 0;
  Particle_Number_Samples_Existing = NULL;
  Particle_Number_Samples = NULL;
  Particle_Number_Output_Variables = NULL;
  Particle_Output_Variables = NULL;
  Particle_Filename_Template = NULL;
  Particle_Density = 0.0;
  Particle_Radius = 0.0;
  Particle_Ratio = 0.0;
  Particle_Show_Debug_Info = 0;
  Particle_Creation_Domain = 0;
  Particle_Move_Domain = 0;
  memset(Particle_Creation_Domain_Filename, 0, sizeof(particle_filename_s));
  memset(Particle_Creation_Domain_Name, 0, sizeof(particle_s));
  memset(Particle_Move_Domain_Filename, 0, sizeof(particle_filename_s));
  memset(Particle_Move_Domain_Name, 0, sizeof(particle_s));
  memset(Particle_Creation_Domain_Reals, 0, sizeof(Particle_Creation_Domain_Reals));
  memset(Particle_Move_Domain_Reals, 0, sizeof(Particle_Move_Domain_Reals));
  xi_boundary_tolerances[0] = XI_BOUNDARY_TOLERANCE0;
  xi_boundary_tolerances[1] = XI_BOUNDARY_TOLERANCE1;
  xi_boundary_tolerances[2] = XI_BOUNDARY_TOLERANCE2;
  Particle_Number_PBCs = 0;
  PBCs = NULL;
  element_particle_info = NULL;

  DPI_ptr = NULL;
  EXO_ptr = NULL;
  memset(Input_File, 0, sizeof(Input_File));
  memset(ExoFile, 0, sizeof(ExoFile));
  memset(Exo_LB_File, 0, sizeof(Exo_LB_File));
  memset(ExoFileOut, 0, sizeof(ExoFileOut));
  memset(ExoFileOutMono, 0, sizeof(ExoFileOutMono));
  memset(ExoAuxFile, 0, sizeof(ExoAuxFile));
  ExoTimePlane = INT_MAX;
  memset(Echo_Input_File, 0, sizeof(Echo_Input_File));
  Decompose_Flag = 1;
  Decompose_Type = 0;
  Skip_Fix = 0;
  GomaPetscOptions = NULL;
  GomaPetscOptionsStrLen = 0;
  memset(DomainMappingFile, 0, sizeof(DomainMappingFile));
  CPU_word_size = 0;
  IO_word_size = 0;
  memset(Init_GuessFile, 0, sizeof(Init_GuessFile));
  memset(Soln_OutFile, 0, sizeof(Soln_OutFile));
  Debug_Flag = 0;
  New_Parser_Flag = 0;
#ifdef MATRIX_DUMP
  Number_Jac_Dump = 0;
#endif
  Iout = 0;
  Write_Intermediate_Solutions = FALSE;
  Write_Initial_Solution = FALSE;
  Num_Var_Init = 0;
  Num_Var_Bound = 0;
  Num_Var_LS_Init = 0;
  Num_Var_External = 0;
  Num_Var_External_pix = 0;
  Anneal_Mesh = 0;
  Porous_liq_inventory = 0.0;
  /* freed by the solvers */
  Spec_source_inventory = NULL;
  Spec_source_lumped_mass = NULL;
  memset(front_scratch_directory, 0, sizeof(front_scratch_directory));
  strcpy(front_scratch_directory, FRONT_SCRATCH_DIRECTORY);

  parallel_err = FALSE;
  parallel_err_global = FALSE;
  Unlimited_Output = TRUE;
  Num_Proc = 1;
  ProcID = 0;
  Dim = -1;
  unlerr = 0;
  /* Request and Status are freed by goma_destroy */
  Request = NULL;
  Status = NULL;
  type_extent = 0;
  Num_Requests = 6;
  Noahs_Raven = NULL;
  Noahs_Ark = NULL;
  Noahs_Dove = NULL;
  cx = NULL;
  time_goma_started = 0.0;
  Argv = NULL;
  Argc = 0;
  Element_Blocks = NULL;
  Current_EB_ptr = NULL;
}
