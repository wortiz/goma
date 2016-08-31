// This defines useful macros like HAVE_MPI, which is defined if and
// only if Epetra was built with MPI enabled.
#include <Epetra_config.h>

#ifdef HAVE_MPI
#  include <mpi.h>
// Epetra's wrapper for MPI_Comm.  This header file only exists if
// Epetra was built with MPI enabled.
#  include <Epetra_MpiComm.h>
#else
#  include <Epetra_SerialComm.h>
#endif // HAVE_MPI
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_EpetraLinearOp.hpp"
#include "Thyra_get_Epetra_Operator.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_Version.h>

#include <sstream>
#include <stdexcept>

#include "ns_structs.h"
#include "Epetra_Vector.h"
#include "Epetra_LinearProblem.h"
#include "Amesos.h"
#include "Amesos_ConfigDefs.h"

typedef int global_ordinal_type;

// Create and return a pointer to an example CrsMatrix, with row
// distribution over the given Map.  The caller is responsible for
// freeing the result.
Epetra_CrsMatrix*
createCrsMatrix (const Epetra_Map& map)
{
  const Epetra_Comm& comm = map.Comm ();

  // Create an Epetra_CrsMatrix using the Map, with dynamic allocation.
  Epetra_CrsMatrix* A = new Epetra_CrsMatrix (Copy, map, 3);

  // The list of global indices owned by this MPI process.

  const global_ordinal_type* myGblElts = NULL;
  global_ordinal_type numGblElts = 0;
#ifdef EXAMPLE_USES_64BIT_GLOBAL_INDICES
  myGblElts = map.MyGlobalElements64 ();
  numGblElts = map.NumGlobalElements64 ();
#else
  myGblElts = map.MyGlobalElements ();
  numGblElts = map.NumGlobalElements ();
#endif // EXAMPLE_USES_64BIT_GLOBAL_INDICES

  // The number of global indices owned by this MPI process.
  const int numMyElts = map.NumMyElements ();

  // In general, tests like this really should synchronize across all
  // processes.  However, the likely cause for this case is a
  // misconfiguration of Epetra, so we expect it to happen on all
  // processes, if it happens at all.
  if (numMyElts > 0 && myGblElts == NULL) {
    throw std::logic_error ("Failed to get the list of global indices");
  }

  // Local error code for use below.
  int lclerr = 0;

  // Fill the sparse matrix, one row at a time.
  double tempVals[3];
  global_ordinal_type tempGblInds[3];
  for (int i = 0; i < numMyElts; ++i) {
    // A(0, 0:1) = [2, -1]
    if (myGblElts[i] == 0) {
      tempVals[0] = 2.0;
      tempVals[1] = -1.0;
      tempGblInds[0] = myGblElts[i];
      tempGblInds[1] = myGblElts[i] + 1;
      if (lclerr == 0) {
        lclerr = A->InsertGlobalValues (myGblElts[i], 2, tempVals, tempGblInds);
      }
      if (lclerr < 0) {
        break;
      }
    }
    // A(N-1, N-2:N-1) = [-1, 2]
    else if (myGblElts[i] == numGblElts - 1) {
      tempVals[0] = -1.0;
      tempVals[1] = 2.0;
      tempGblInds[0] = myGblElts[i] - 1;
      tempGblInds[1] = myGblElts[i];
      if (lclerr == 0) {
        lclerr = A->InsertGlobalValues (myGblElts[i], 2, tempVals, tempGblInds);
      }
      if (lclerr != 0) {
        break;
      }
    }
    // A(i, i-1:i+1) = [-1, 2, -1]
    else {
      tempVals[0] = -1.0;
      tempVals[1] = 2.0;
      tempVals[2] = -1.0;
      tempGblInds[0] = myGblElts[i] - 1;
      tempGblInds[1] = myGblElts[i];
      tempGblInds[2] = myGblElts[i] + 1;
      if (lclerr == 0) {
        lclerr = A->InsertGlobalValues (myGblElts[i], 3, tempVals, tempGblInds);
      }
      if (lclerr != 0) {
        break;
      }
    }
  }

  // If any process failed to insert at least one entry, throw.
  int gblerr = 0;
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    if (A != NULL) {
      delete A;
    }
    throw std::runtime_error ("Some process failed to insert an entry.");
  }

  // Tell the sparse matrix that we are done adding entries to it.
  gblerr = A->FillComplete ();
  if (gblerr != 0) {
    if (A != NULL) {
      delete A;
    }
    std::ostringstream os;
    os << "A->FillComplete() failed with error code " << gblerr << ".";
    throw std::runtime_error (os.str ());
  }

  return A;
}


void
example (const Epetra_Comm& comm)
{
  // The global number of rows in the matrix A to create.  We scale
  // this relative to the number of (MPI) processes, so that no matter
  // how many MPI processes you run, every process will have 10 rows.
  const global_ordinal_type numGblElts = 10 * comm.NumProc ();
  // The global min global index in all the Maps here.
  const global_ordinal_type indexBase = 0;

  // Local error code for use below.
  //
  // In the ideal case, we would use this to emulate behavior like
  // that of Haskell's Maybe in the context of MPI.  That is, if one
  // process experiences an error, we don't want to abort early and
  // cause the other processes to deadlock on MPI communication
  // operators.  Rather, we want to chain along the local error state,
  // until we reach a point where it's natural to pass along that
  // state with other processes.  For example, if one is doing an
  // MPI_Allreduce anyway, it makes sense to pass along one more bit
  // of information: whether the calling process is in a local error
  // state.  Epetra's interface doesn't let one chain the local error
  // state in this way, so we use extra collectives below to propagate
  // that state.  The code below uses very conservative error checks;
  // typical user code would not need to be so conservative and could
  // therefore avoid all the all-reduces.
  int lclerr = 0;

  // Construct a Map that is global (not locally replicated), but puts
  // all the equations on MPI Proc 0.
  const int procZeroMapNumLclElts = (comm.MyPID () == 0) ?
    numGblElts :
    static_cast<global_ordinal_type> (0);
  Epetra_Map procZeroMap (numGblElts, procZeroMapNumLclElts, indexBase, comm);

  // Construct a Map that puts approximately the same number of
  // equations on each processor.
  Epetra_Map globalMap (numGblElts, indexBase, comm);

  // Create a sparse matrix using procZeroMap.
  Epetra_CrsMatrix* A = createCrsMatrix (procZeroMap);
  if (A == NULL) {
    lclerr = 1;
  }

  // Make sure that sparse matrix creation succeeded.  Normally you
  // don't have to check this; we are being extra conservative because
  // this example is also a test.  Even though the matrix's rows live
  // entirely on Process 0, the matrix is nonnull on all processes in
  // its Map's communicator.
  int gblerr = 0;
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("createCrsMatrix returned NULL on at least one "
                              "process.");
  }

  //
  // We've created a sparse matrix whose rows live entirely on MPI
  // Process 0.  Now we want to distribute it over all the processes.
  //

  // Redistribute the matrix.  Since both the source and target Maps
  // are one-to-one, we could use either an Import or an Export.  If
  // only the source Map were one-to-one, we would have to use an
  // Import; if only the target Map were one-to-one, we would have to
  // use an Export.  We do not allow redistribution using Import or
  // Export if neither source nor target Map is one-to-one.

  // Make an export object with procZeroMap as the source Map, and
  // globalMap as the target Map.  The Export type has the same
  // template parameters as a Map.  Note that Export does not depend
  // on the Scalar template parameter of the objects it
  // redistributes.  You can reuse the same Export for different
  // Tpetra object types, or for Tpetra objects of the same type but
  // different Scalar template parameters (e.g., Scalar=float or
  // Scalar=double).
  Epetra_Export exporter (procZeroMap, globalMap);

  // Make a new sparse matrix whose row map is the global Map.
  Epetra_CrsMatrix B (Copy, globalMap, 0);

  // Redistribute the data, NOT in place, from matrix A (which lives
  // entirely on Proc 0) to matrix B (which is distributed evenly over
  // the processes).
  //
  // Export() has collective semantics, so we must always call it on
  // all processes collectively.  This is why we don't select on
  // lclerr, as we do for the local operations above.
  lclerr = B.Export (*A, exporter, Insert);

  // Make sure that the Export succeeded.  Normally you don't have to
  // check this; we are being extra conservative because this example
  // example is also a test.  We test both min and max, since lclerr
  // may be negative, zero, or positive.
  gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("Export() failed on at least one process.");
  }
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("Export() failed on at least one process.");
  }

  // FillComplete has collective semantics, so we must always call it
  // on all processes collectively.  This is why we don't select on
  // lclerr, as we do for the local operations above.
  lclerr = B.FillComplete ();

  // Make sure that FillComplete succeeded.  Normally you don't have
  // to check this; we are being extra conservative because this
  // example is also a test.  We test both min and max, since lclerr
  // may be negative, zero, or positive.
  gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("B.FillComplete() failed on at least one process.");
  }
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("B.FillComplete() failed on at least one process.");
  }

  if (A != NULL) {
    delete A;
  }
}

// Create and return a pointer to an example CrsMatrix, with row
// distribution over the given Map.  The caller is responsible for
// freeing the result.
Epetra_CrsMatrix*
copyCrsMatrix (const Epetra_Map& map, matrix_data *mat_system)
{
  const Epetra_Comm& comm = map.Comm ();

  // Create an Epetra_CrsMatrix using the Map, with dynamic allocation.
  Epetra_CrsMatrix* A = new Epetra_CrsMatrix (Copy, map, 10);

  // The list of global indices owned by this MPI process.

  const global_ordinal_type* myGblElts = NULL;
  global_ordinal_type numGblElts = 0;
#ifdef EXAMPLE_USES_64BIT_GLOBAL_INDICES
  myGblElts = map.MyGlobalElements64 ();
  numGblElts = map.NumGlobalElements64 ();
#else
  myGblElts = map.MyGlobalElements ();
  numGblElts = map.NumGlobalElements ();
#endif // EXAMPLE_USES_64BIT_GLOBAL_INDICES

  // The number of global indices owned by this MPI process.
  const int numMyElts = map.NumMyElements ();

  // In general, tests like this really should synchronize across all
  // processes.  However, the likely cause for this case is a
  // misconfiguration of Epetra, so we expect it to happen on all
  // processes, if it happens at all.
  if (numMyElts > 0 && myGblElts == NULL) {
    throw std::logic_error ("Failed to get the list of global indices");
  }

  // Local error code for use below.
  int lclerr = 0;

  // Fill the sparse matrix, one row at a time.
  double tempVals[3];
  global_ordinal_type tempGblInds[3];
  for (int i = 0; i < numMyElts; ++i) {
    int global_row = myGblElts[i];
    if (lclerr >= 0) {
      int count = mat_system->A->rowptr[global_row+1] - mat_system->A->rowptr[global_row];
      int idx = mat_system->A->rowptr[global_row];
      lclerr = A->InsertGlobalValues (global_row, count, mat_system->A->val + idx, mat_system->A->colind + idx);
    }
    if (lclerr < 0) {
      break;
    }
  }

  // If any process failed to insert at least one entry, throw.
  int gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr < 0) {
    if (A != NULL) {
      delete A;
    }
    throw std::runtime_error ("Some process failed to insert an entry.");
  }

  // Tell the sparse matrix that we are done adding entries to it.
  gblerr = A->FillComplete ();
  if (gblerr != 0) {
    if (A != NULL) {
      delete A;
    }
    std::ostringstream os;
    os << "A->FillComplete() failed with error code " << gblerr << ".";
    throw std::runtime_error (os.str ());
  }

  return A;
}

void trilinos_solve(const Epetra_Comm & comm, matrix_data *mat_system)
{

  int global_elements;
  int nnz = 0;

  if (comm.MyPID() == 0) global_elements = mat_system->A->m;
  if (comm.MyPID() == 0) nnz = mat_system->A->nnz;

  comm.Broadcast(&global_elements, 1, 0);
  comm.Broadcast(&nnz, 1, 0);

  comm.Barrier();
  
  global_ordinal_type numGblElts = global_elements;
  // The global min global index in all the Maps here.
  const global_ordinal_type indexBase = 0;

  // Local error code for use below.
  //
  // In the ideal case, we would use this to emulate behavior like
  // that of Haskell's Maybe in the context of MPI.  That is, if one
  // process experiences an error, we don't want to abort early and
  // cause the other processes to deadlock on MPI communication
  // operators.  Rather, we want to chain along the local error state,
  // until we reach a point where it's natural to pass along that
  // state with other processes.  For example, if one is doing an
  // MPI_Allreduce anyway, it makes sense to pass along one more bit
  // of information: whether the calling process is in a local error
  // state.  Epetra's interface doesn't let one chain the local error
  // state in this way, so we use extra collectives below to propagate
  // that state.  The code below uses very conservative error checks;
  // typical user code would not need to be so conservative and could
  // therefore avoid all the all-reduces.
  int lclerr = 0;

  // Construct a Map that is global (not locally replicated), but puts
  // all the equations on MPI Proc 0.
  const int procZeroMapNumLclElts = (comm.MyPID () == 0) ?
    numGblElts :
    static_cast<global_ordinal_type> (0);

  Epetra_Map procZeroMap (numGblElts, procZeroMapNumLclElts, indexBase, comm);

  // Construct a Map that puts approximately the same number of
  // equations on each processor.
  Epetra_Map globalMap (numGblElts, indexBase, comm);

  // Create a sparse matrix using procZeroMap.
  Epetra_CrsMatrix* A = copyCrsMatrix (procZeroMap, mat_system);
  if (A == NULL) {
    lclerr = 1;
  }

  // Make sure that sparse matrix creation succeeded.  Normally you
  // don't have to check this; we are being extra conservative because
  // this example is also a test.  Even though the matrix's rows live
  // entirely on Process 0, the matrix is nonnull on all processes in
  // its Map's communicator.
  int gblerr = 0;
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("createCrsMatrix returned NULL on at least one "
                              "process.");
  }


  //
  // We've created a sparse matrix whose rows live entirely on MPI
  // Process 0.  Now we want to distribute it over all the processes.
  //

  // Redistribute the matrix.  Since both the source and target Maps
  // are one-to-one, we could use either an Import or an Export.  If
  // only the source Map were one-to-one, we would have to use an
  // Import; if only the target Map were one-to-one, we would have to
  // use an Export.  We do not allow redistribution using Import or
  // Export if neither source nor target Map is one-to-one.

  // Make an export object with procZeroMap as the source Map, and
  // globalMap as the target Map.  The Export type has the same
  // template parameters as a Map.  Note that Export does not depend
  // on the Scalar template parameter of the objects it
  // redistributes.  You can reuse the same Export for different
  // Tpetra object types, or for Tpetra objects of the same type but
  // different Scalar template parameters (e.g., Scalar=float or
  // Scalar=double).
  Epetra_Export exporter (procZeroMap, globalMap);
  Epetra_Export exporter_reverse (globalMap, procZeroMap);

  // Make a new sparse matrix whose row map is the global Map.
  Epetra_CrsMatrix B (Copy, globalMap, 0);

  // Redistribute the data, NOT in place, from matrix A (which lives
  // entirely on Proc 0) to matrix B (which is distributed evenly over
  // the processes).
  //
  // Export() has collective semantics, so we must always call it on
  // all processes collectively.  This is why we don't select on
  // lclerr, as we do for the local operations above.
  lclerr = B.Export (*A, exporter, Insert);

  // Make sure that the Export succeeded.  Normally you don't have to
  // check this; we are being extra conservative because this example
  // example is also a test.  We test both min and max, since lclerr
  // may be negative, zero, or positive.
  gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("Export() failed on at least one process.");
  }
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("Export() failed on at least one process.");
  }

  // FillComplete has collective semantics, so we must always call it
  // on all processes collectively.  This is why we don't select on
  // lclerr, as we do for the local operations above.
  lclerr = B.FillComplete ();

  Epetra_Vector local_x(procZeroMap);
  Epetra_Vector local_b(procZeroMap);
  const int numMyElts = procZeroMap.NumMyElements ();
  const global_ordinal_type* myGblElts = NULL;
  myGblElts = procZeroMap.MyGlobalElements ();
  
  for (int i = 0; i < numMyElts; ++i) {
    int global_row = myGblElts[i];
    local_x.ReplaceGlobalValues(1, &mat_system->delta_x[global_row], &global_row);
    local_b.ReplaceGlobalValues(1, &mat_system->b[global_row], &global_row);
  }

  Epetra_Vector x(globalMap);
  Epetra_Vector b(globalMap);

  lclerr = x.Export(local_x, exporter, Insert);
  gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("Export() failed on at least one process.");
  }

  lclerr = b.Export(local_b, exporter, Insert);
  gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("Export() failed on at least one process.");
  }
  //  std::cout << "B nnz = " << B.NumGlobalNonzeros() << " , " << nnz << std::endl;
  assert(nnz == B.NumGlobalNonzeros());

  /*
  Epetra_LinearProblem Problem;

  Amesos_BaseSolver *A_Base;
  Amesos A_Factory;

  char * Pkg_Name = "Amesos_Mumps";

  Problem.SetOperator(&B);
  Problem.SetLHS(&x);
  Problem.SetRHS(&b);
  Problem.CheckInput();
  A_Base = A_Factory.Create( Pkg_Name, Problem );
  
  A_Base->SymbolicFactorization();
  A_Base->NumericFactorization();	
  A_Base->Solve();
  */

  using Teuchos::RCP;
  bool success = true;
  bool verbose = true;

  try {
    // Assign A with false so it doesn't get garbage collected.
    RCP<Epetra_CrsMatrix> epetra_A = Teuchos::rcp(&B, false);
    RCP<Epetra_Vector> epetra_x = Teuchos::rcp(
        new Epetra_Vector(x));
    RCP<Epetra_Vector> epetra_b = Teuchos::rcp(
        new Epetra_Vector(b));

    RCP<const Thyra::LinearOpBase<double> > tA = Thyra::epetraLinearOp(epetra_A);
    RCP<Thyra::VectorBase<double> > tx = Thyra::create_Vector(epetra_x,
							      tA->domain());
    RCP<const Thyra::VectorBase<double> > tb = Thyra::create_Vector(epetra_b,
								    tA->range());

    Teuchos::RCP<Teuchos::FancyOStream> outstream =
        Teuchos::VerboseObjectBase::getDefaultOStream();

    // Get parameters from file
    RCP<Teuchos::ParameterList> solverParams;
    solverParams = Teuchos::getParametersFromXmlFile("stratimikos.xml");

    // Set up base builder
    Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
    
#ifdef HAVE_TEKO
    Teko::addTekoToStratimikosBuilder(linearSolverBuilder);
#endif

    linearSolverBuilder.setParameterList(solverParams);

    // set up solver factory using base/params
    RCP<Thyra::LinearOpWithSolveFactoryBase<double> > solverFactory =
        linearSolverBuilder.createLinearSolveStrategy("");


    linearSolverBuilder.writeParamsFile(*solverFactory, "echo_stratimikos.xml");
    
    // set output stream
    solverFactory->setOStream(outstream);

    // set solver verbosity
    solverFactory->setDefaultVerbLevel(Teuchos::VERB_NONE);

    RCP<Thyra::LinearOpWithSolveBase<double> > solver =
        Thyra::linearOpWithSolve(*solverFactory, tA);

    Thyra::SolveStatus<double> status = Thyra::solve<double>(*solver,
        Thyra::NOTRANS, *tb, tx.ptr());

    tx = Teuchos::null;

    /* Convert solution vector */
    Epetra_Vector *raw_x = epetra_x.get();

    local_x.Export(*raw_x, exporter_reverse, Insert);
    for (int i = 0; i < numMyElts; ++i) {
      int global_row = myGblElts[i];
      mat_system->delta_x[global_row] = local_x[global_row];
    }
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success)

  // Make sure that FillComplete succeeded.  Normally you don't have
  // to check this; we are being extra conservative because this
  // example is also a test.  We test both min and max, since lclerr
  // may be negative, zero, or positive.
  gblerr = 0;
  (void) comm.MinAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("B.FillComplete() failed on at least one process.");
  }
  (void) comm.MaxAll (&lclerr, &gblerr, 1);
  if (gblerr != 0) {
    throw std::runtime_error ("B.FillComplete() failed on at least one process.");
  }

  if (A != NULL) {
    delete A;
  }
}

/*int
  main (int argc, char *argv[])
  {
  using std::cout;
  using std::endl;

  #ifdef HAVE_MPI
  MPI_Init (&argc, &argv);
  Epetra_MpiComm comm (MPI_COMM_WORLD);
  #else
  Epetra_SerialComm comm;
  #endif // HAVE_MPI

  const int myRank = comm.MyPID ();
  const int numProcs = comm.NumProc ();

  if (myRank == 0) {
  // Print out the Epetra software version.
  cout << Epetra_Version () << endl << endl
  << "Total number of processes: " << numProcs << endl;
  }

  example (comm); // Run the whole example.

  // This tells the Trilinos test framework that the test passed.
  if (myRank == 0) {
  cout << "End Result: TEST PASSED" << endl;
  }

  #ifdef HAVE_MPI
  (void) MPI_Finalize ();
  #endif // HAVE_MPI

  return 0;
  }
*/
