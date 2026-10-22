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
#include "goma_reset.h"

/* This removes the entire file if Amesos & Trilinos are not defined */
#if defined(GOMA_ENABLE_AMESOS) && defined(TRILINOS)
#include <stdlib.h>
#include <string>

#include "Epetra_ConfigDefs.h"
#include "Epetra_DataAccess.h"
#include "Epetra_Object.h"
#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#ifndef GOMA_SL_AMESOS_INTERFACE_CC
#define GOMA_SL_AMESOS_INTERFACE_CC
#endif

#if defined(PARALLEL) && !defined(EPETRA_MPI)
#define EPETRA_MPI
#endif

#ifndef __cplusplus
#define __cplusplus
#endif

#include <iostream>

#include "mpi.h"
#include "sl_amesos_interface.h"

#ifdef EPETRA_MPI
#define AZ_MPI
#define AZTEC_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "AztecOO.h"
#include "EpetraExt_MatrixMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Trilinos_Util.h"

static void bf_mat_to_Epetra(double *, int *, Epetra_CrsMatrix *, int, int, int);

static int ls_FirstRun = 1;
static int ls_err = 0;
static std::string ls_Pkg_Name;
static Epetra_CrsMatrix *ls_A = nullptr;
static Epetra_LinearProblem ls_Problem;

void trilinos_solve_ls(double *bf_mat_,
                       int *j_map_,
                       double *f_rhs_,
                       double *x_,
                       double *Atranspose_f_,
                       int txt_num_pts,
                       int nnz_per_row,
                       int num_cols,
                       int NewMatrix) {

  /* Initialize MPI communications */
#ifdef EPETRA_MPI
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif

  std::cout << comm << std::endl;

  int options[AZ_OPTIONS_SIZE];
  double params[AZ_PARAMS_SIZE];

  /* Define internal variables */
  bool transA = true;
  bool notransA = false;

  /* Convert to Epetra format */
  if (NewMatrix) {
    if (!ls_FirstRun)
      delete ls_A;
    ls_A = (Epetra_CrsMatrix *)construct_Epetra_CrsMatrix_ls(txt_num_pts, nnz_per_row, num_cols);
    bf_mat_to_Epetra(bf_mat_, j_map_, ls_A, txt_num_pts, nnz_per_row, num_cols);
  }

  const Epetra_Map &map = (*ls_A).RowMatrixRowMap();
  const Epetra_Map &Colmap = (*ls_A).RowMatrixColMap();
  Epetra_Vector f_rhs(Copy, map, f_rhs_);
  Epetra_Vector x(Copy, Colmap, x_);

  Epetra_CrsMatrix A_T_A(Copy, Colmap, num_cols);
  Epetra_Vector A_T_f(Copy, Colmap, Atranspose_f_);

  // Form Normal Equations

  ls_err = EpetraExt::MatrixMatrix::Multiply(*ls_A, transA, *ls_A, notransA, A_T_A);
  if (ls_err != 0) {
    std::cout << "MatrixMatrix" << ls_err << std::endl;
    abort();
  }
  (*ls_A).Multiply(transA, f_rhs, A_T_f);

  /* Assemble linear problem */
  // if (NewMatrix) Epetra_LinearProblem Problem(&A_T_A, &x, &A_T_f);
  if (NewMatrix) {
    ls_Problem.SetOperator(&A_T_A);
    ls_Problem.SetLHS(&x);
    ls_Problem.SetRHS(&A_T_f);
    ls_Problem.CheckInput();
  }

  AztecOO Solver(ls_Problem);
  /* Set aztec params to default */
  AZ_defaults(options, params);

  Solver.SetAllAztecOptions(options);
  Solver.SetAllAztecParams(params);

  // Solver.SetAztecOption( AZ_precond, AZ_Jacobi);
  Solver.SetAztecOption(AZ_precond, AZ_Neumann);
  Solver.SetAztecOption(AZ_solver, AZ_bicgstab);
  Solver.SetAztecOption(AZ_output, 4);

  Solver.Iterate(500, 6e-2);
  std::cout << "solver performed" << Solver.NumIters() << " iterations." << std::endl
            << "Norm of true residual = " << Solver.TrueResidual() << std::endl;

  /* Convert solution vector */
  for (int i = 0; i < num_cols; i++) {
    x_[i] = x[i];
  }

  /* Cleanup problem */
  ls_FirstRun = 0;
}

static void bf_mat_to_Epetra(
    double *bf_mat_, int *j_map_, Epetra_CrsMatrix *A, int txt_num_pts, int nnz_per_row, int n_cols)

{
#ifdef EPETRA_MPI
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif

  static int newmatrix = 1;

  int NumMyRows = txt_num_pts;
  int NumMyCols = n_cols;

  (*A).PutScalar(0.0);

  Epetra_Map Map(NumMyRows, 0, comm); // Not a square matrix, though,as this supposes
  int *MyGlobalElements = Map.MyGlobalElements();

  Epetra_Map DomainMap(NumMyCols, 0, comm); // Not a square matrix, though,as this supposes

  int *NumNz = new int[NumMyRows];
  for (int i = 0; i < NumMyRows; i++)
    NumNz[i] = nnz_per_row;

  int *Indices = new int[nnz_per_row];
  double *Values = new double[nnz_per_row];

  for (int i = 0; i < NumMyRows; i++) {
    for (int j = 0; j < NumNz[i]; j++) {
      Indices[j] = j_map_[i * nnz_per_row + j];
      Values[j] = bf_mat_[i * nnz_per_row + j];
    }

    if (newmatrix) {
      (*A).InsertGlobalValues(MyGlobalElements[i], NumNz[i], Values, Indices);
      //(*A).InsertGlobalValues( MyGlobalElements[i], 1, &(bf_mat_[0]), MyGlobalElements+i );
    } else {
      (*A).SumIntoGlobalValues(MyGlobalElements[i], NumNz[i], Values, Indices);
      //(*A).SumIntoGlobalValues( MyGlobalElements[i], 1, &(val[i]), MyGlobalElements+i );
    }
  }

  (*A).FillComplete(DomainMap, Map);

  delete[] Values;
  delete[] Indices;
  delete[] NumNz;
  return;
}

void *
construct_Epetra_CrsMatrix_ls(int const txt_num_pts, int const nnz_per_row, int const num_cols) {
#ifdef EPETRA_MPI
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif

  // if ( ams->EpetraMat != NULL ) return ( ams->EpetraMat );

  int NumMyRows = txt_num_pts;
  int NumMyCols = num_cols;

  Epetra_Map RowMap(-1, NumMyRows, 0, comm);
  Epetra_Map DomainMap(-1, NumMyCols, 0, comm);

  int *NumNz = new int[NumMyRows];

  for (int i = 0; i < NumMyRows; i++) {
    NumNz[i] = nnz_per_row;
  }

  Epetra_CrsMatrix *A = new Epetra_CrsMatrix(Copy, RowMap, DomainMap, NumNz);

  delete[] NumNz;

  return (A);
}

/*
 * Reset this file's globals to their load time values, see goma_reset.h
 *
 * ls_Problem is given its operator and vectors on every call.
 */
void leastsquares_free_globals(void) {
  if (!ls_FirstRun)
    delete ls_A;
  ls_A = nullptr;
  ls_FirstRun = 1;
  ls_err = 0;
  ls_Pkg_Name.clear();
}

#else

void leastsquares_free_globals(void) {}

#endif // if defined(GOMA_ENABLE_AMESOS) && defined(TRILINOS)
