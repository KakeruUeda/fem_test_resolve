/**
 * @file main.pp
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sys/stat.h>

#include "ExampleHelper.hpp"
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

// Uses ReSolve data types
using namespace ReSolve;
using namespace ReSolve::examples;
using namespace ReSolve::memory;
using vector_type = ReSolve::vector::Vector;
using index_type = ReSolve::index_type;

/// Prints help message describing system usage
void printHelpInfo()
{
  std::cout << "\nsysGmres.exe loads a linear system from files and solves it using GMRES.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "sysGmres.exe -m <matrix file> -r <rhs file>\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-b <cpu|cuda|hip> \tSelects hardware backend.\n";
  std::cout << "\t-h \tPrints this message.\n";
  std::cout << "\t-i <iter method> \tIterative method: randgmres or fgmres (default 'randgmres').\n";
  std::cout << "\t-g <gs method> \tGram-Schmidt method: cgs1, cgs2, or mgs (default 'cgs2').\n";
  std::cout << "\t-s <sketching method> \tSketching method: count or fwht (default 'count')\n";
  std::cout << "\t-x <flexible> \tEnable flexible: yes or no (default 'yes')\n\n";
}

/// Prototype of the example function
template <class workspace_type>
static int sysGmres(int argc, char *argv[]);

/// Checks if inputs for GMRES are valid, otherwise sets defaults
static void processInputs(std::string &method,
                          std::string &gs,
                          std::string &sketch,
                          std::string &flexible);

/// Main function selects example to be run
int main(int argc, char *argv[])
{
  CliOptions options(argc, argv);

  bool is_help = options.hasKey("-h");
  if (is_help)
  {
    printHelpInfo();
    return 0;
  }

  auto opt = options.getParamFromKey("-b");
  if (!opt)
  {
    std::cout << "No backend option provided. Defaulting to CPU.\n";
    return sysGmres<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
  }
#ifdef RESOLVE_USE_CUDA
  else if (opt->second == "cuda")
  {
    return sysGmres<ReSolve::LinAlgWorkspaceCUDA>(argc, argv);
  }
#endif
#ifdef RESOLVE_USE_HIP
  else if (opt->second == "hip")
  {
    return sysGmres<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
  }
#endif
  else if (opt->second == "cpu")
  {
    return sysGmres<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
  }
  else
  {
    std::cout << "Re::Solve is not build with support for " << opt->second;
    std::cout << " backend.\n";
    return 1;
  }

  return 0;
}

/**
 * @brief Example of solving a linear system with GMRES using SystemSolver.
 *
 * @tparam workspace_type - Type of the workspace to use
 * @param[in] argc - Number of command line arguments
 * @param[in] argv - Command line arguments
 * @return 0 if the example ran successfully, 1 otherwise
 */
template <class workspace_type>
int sysGmres(int argc, char *argv[])
{
  std::string output_dir_tmp = "outputs";
  mkdir(output_dir_tmp.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  std::string output_dir = "outputs/cavityflow";
  mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  std::ofstream log;
  std::string log_path = output_dir + "/run.log";
  log.open(log_path, std::ios::app);
  if (!log)
  {
    std::cout << "Failed to open log file\n";
    return 0;
  }
  
  // Check if file is new (empty) and write header
  std::ifstream check_file(log_path);
  check_file.seekg(0, std::ios::end);
  if (check_file.tellg() == 0)
  {
    log << "step time [s]\n";
    log.flush();
  }
  check_file.close();

  // return_code is used as a failure flag.
  int return_code = 0;
  int status = 0;

  // Collect all CLI
  CliOptions options(argc, argv);

  bool is_help = options.hasKey("-h");
  if (is_help)
  {
    printHelpInfo();
    return 0;
  }

  // Read matrix file
  auto opt = options.getParamFromKey("-m");
  std::string matrix_pathname("");
  if (opt)
  {
    matrix_pathname = opt->second;
  }
  else
  {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
    return 1;
  }

  // Read right-hand-side vector file
  std::string rhs_pathname("");
  opt = options.getParamFromKey("-r");
  if (opt)
  {
    rhs_pathname = opt->second;
  }
  else
  {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
    return 1;
  } 
  
  index_type num_systems = 0;
  opt         = options.getParamFromKey("-n");
  if (opt)
  {
    num_systems = atoi((opt->second).c_str());
  }
  else
  {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  // Read GMRES-related options
  opt = options.getParamFromKey("-i");
  std::string method = opt ? (*opt).second : "randgmres";

  opt = options.getParamFromKey("-g");
  std::string gs = opt ? (*opt).second : "cgs2";

  opt = options.getParamFromKey("-s");
  std::string sketch = opt ? (*opt).second : "count";

  opt = options.getParamFromKey("-x");
  std::string flexible = opt ? (*opt).second : "yes";

  processInputs(method, gs, sketch, flexible);

  std::cout << "Family matrix file name: " << matrix_pathname
            << ", total number of matrices: " << num_systems << "\n"
            << "Family rhs file name: " << rhs_pathname
            << ", total number of RHSes: " << num_systems << "\n";

  // Create workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);
  std::string hw_backend = helper.getHardwareBackend();
  std::cout << "sysGmres with " << hw_backend << " backend\n";

  // Set memory space
  MemorySpace memspace = helper.getMemspace();

  // Set solver
  SystemSolver solver(&workspace,
                      "none",
                      "none",
                      method,
                      "ilu0",
                      "none");

  solver.setGramSchmidtMethod(gs);

  bool is_expand_symmetric = true;

  matrix::Csr *A = nullptr;
  vector_type *vec_rhs = nullptr;
  vector_type *vec_x = nullptr;

  for (int i = 0; i < num_systems; ++i)
  {
    std::cout << "step: " << i << "/" << num_systems << "\n"; 
    std::ostringstream matname;
    std::ostringstream rhsname;

    matname << matrix_pathname << std::setfill('0') << std::setw(4) << i << ".mtx";
    rhsname << rhs_pathname << std::setfill('0') << std::setw(4) << i << ".mtx";

    std::string matrix_pathname_full = matname.str();
    std::string rhs_pathname_full = rhsname.str();

    // Read and open matrix and right-hand-side vector
    std::ifstream mat_file(matrix_pathname_full);
    if (!mat_file.is_open())
    {
      std::cout << "Failed to open matrix file: " << matrix_pathname_full << "\n";
      return 1;
    }
    std::ifstream rhs_file(rhs_pathname_full);
    if (!rhs_file.is_open())
    {
      std::cout << "Failed to open RHS file: " << rhs_pathname_full << "\n";
      return 1;
    }

    if (i == 0)
    {
      // First system: allocate and read
      A = io::createCsrFromFile(mat_file, is_expand_symmetric);
      vec_rhs = io::createVectorFromFile(rhs_file);
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(memspace);
    }
    else
    {
      // Subsequent systems: update in-place
      io::updateMatrixFromFile(mat_file, A);
      io::updateVectorFromFile(rhs_file, vec_rhs);
    }
    
    if (memspace == memory::DEVICE)
    {
      // Copy data to the device
      A->syncData(memspace);
      vec_rhs->syncData(memspace);
    }

    mat_file.close();
    rhs_file.close();

    // Set iterative solver options
    solver.getIterativeSolver().setCliParam("maxit", "2500");
    solver.getIterativeSolver().setCliParam("tol", "1e-8");
    
    status = solver.setMatrix(A);
    if (status != 0)
    {
      std::cout << "solver.setMatrix returned status: " << status << "\n";
      return_code = 1;
    }

    // Set GMRES solver options
    if (method == "randgmres")
    {
      solver.setSketchingMethod(sketch);
    }
    solver.getIterativeSolver().setCliParam("flexible", flexible);
    solver.getIterativeSolver().setCliParam("restart", "200");

    // Set up the preconditioner
    if (return_code == 0)
    {
      status = solver.preconditionerSetup();
      if (status != 0)
      {
        std::cout << "solver.preconditionerSetup returned status: " << status << "\n";
        return_code = 1;
      }
    }

    // Solve the system
    if (return_code == 0)
    {
      auto solve_start = std::chrono::high_resolution_clock::now();
      
      status = solver.solve(vec_rhs, vec_x);
      
      auto solve_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> solve_duration = solve_end - solve_start;
      
      log << i << " " << std::scientific << std::setprecision(16) << solve_duration.count() << "\n";
      
      if (status != 0)
      {
        std::cout << "solver.solve returned status: " << status << "\n";
        return_code = 1;
      }
    }

    if (return_code == 0)
    {
      helper.resetSystem(A, vec_rhs, vec_x);

      // Get reference to iterative solver and print results
      // LinSolverIterative &iter_solver = solver.getIterativeSolver();
      // helper.printIterativeSolverSummary(&iter_solver);
    }

    if (return_code == 1)
    {
      break;
    }
  }

  // Final cleanup
  if (A != nullptr) delete A;
  if (vec_rhs != nullptr) delete vec_rhs;
  if (vec_x != nullptr) delete vec_x;

  return return_code;
}

/// Checks GMRES-related CLI options
void processInputs(std::string &method, std::string &gs, std::string &sketch, std::string &flexible)
{
  if (method == "randgmres")
  {
    if ((sketch != "count") && (sketch != "fwht"))
    {
      std::cout << "Sketching method " << sketch << " not recognized.\n";
      std::cout << "Setting sketch to the default (count).\n\n";
      sketch = "count";
    }
  }

  if ((method != "randgmres") && (method != "fgmres"))
  {
    std::cout << "Iterative method " << method << " not recognized.\n";
    std::cout << "Setting iterative solver method to the default (RANDGMRES).\n\n";
    method = "randgmres";
  }

  if (gs != "cgs1" && gs != "cgs2" && gs != "mgs" && gs != "mgs_two_sync" && gs != "mgs_pm")
  {
    std::cout << "Orthogonalization method " << gs << " not recognized.\n";
    std::cout << "Setting orthogonalization to the default (CGS2).\n\n";
    gs = "cgs2";
  }

  if ((flexible != "yes") && (flexible != "no"))
  {
    std::cout << "Flexible option " << flexible << " not recognized.\n";
    std::cout << "Setting flexible to the default (yes).\n\n";
    flexible = "yes";
  }
}
