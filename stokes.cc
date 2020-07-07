/* ---------------------------------------------------------------------
 *  *
 *   *
 *    * ---------------------------------------------------------------------
 *
 *     */
#include <deal.II/base/logstream.h>               // Log system during compile/run
#include <deal.II/base/quadrature_lib.h>          // Quadrature Rules
#include <deal.II/base/convergence_table.h>       // Convergence Table
#include <deal.II/base/timer.h>                   // Timer class

#include <deal.II/dofs/dof_handler.h>             // DoF (defrees of freedom) Handler
#include <deal.II/dofs/dof_tools.h>               // Tools for Working with DoFs

#include <deal.II/fe/fe_q.h>                      // Continuous Finite Element Q Basis Function
#include <deal.II/fe/fe_values.h>                 // Values of Finite Elements on Cell
#include <deal.II/fe/fe_system.h>                 // System (vector) of Finite Elements

#include <deal.II/grid/tria.h>                    // Triangulation declaration
#include <deal.II/grid/tria_accessor.h>           // Access the cells of Triangulation
#include <deal.II/grid/tria_iterator.h>           // Iterate over cells of Triangulations
#include <deal.II/grid/grid_generator.h>          // Generate Standard Grids
#include <deal.II/grid/grid_out.h>                // Output Grids

#include <deal.II/lac/affine_constraints.h>       // Constraint Matrix
#include <deal.II/lac/dynamic_sparsity_pattern.h> // Dynamic Sparsity Pattern
#include <deal.II/lac/sparse_matrix.h>            // Sparse Matrix
#include <deal.II/lac/sparse_direct.h>            // UMFPACK

#include <deal.II/numerics/vector_tools.h>        // Interpolate Boundary Values
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>  // C++ Output
#include <fstream>   // C++ Output
#include <cmath>     // C++ Math Functions




using namespace dealii;

template <int dim>
class StokesExactSolution : public Function<dim>
{
  public:
    StokesExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                                 Vector<double>  &value) const;
    Tensor<1,dim> gradient(const Point<dim> &p,
                           const unsigned int component) const;

};

template <int dim>
void
StokesExactSolution<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));

  double x = p[0];
  double y = p[1];

  values(0) = cos(M_PI*x);
  values(1) = y*M_PI*sin(M_PI*x);
  values(2) = x*x*y*y;

}

template <int dim>
Tensor<1,dim>
StokesExactSolution<dim>::gradient(const Point<dim> &p,
                             const unsigned int component) const
{
  //Recall the exact solution is
  // double constant = 40.0;
  // u_1(x,y) = values(0) = cos(M_PI*x);
  // u_2(x,y) = values(1) = y*M_PI*sin(M_PI*x);
  // p(x,y)   = values(2) = x*x*y*y;
  //
  // Note that
  //   div u(x,y) = u_{1,x} + u_{2,y}
  //              = -M_PI*sin(M_PI*x) + M_PI*sin(M_PI*x)
  //              = 0
  //   grad u(x,y) = [grad u_{1} ; grad u_{2} ]  
  //               = [u_{1,x}  u_{1,y}  ; u_{2,x}   u_{2,y}]
  //               = [-M_PI*sin(M_PI*x)   0 ; y*M_PI*M_PI*cos(M_PI*x)   M_PI*sin(M_PI*x)]
  //   grad p(x,y) = <p_x      ,      p_y>
  //               = <2*x*y*y  ,      2*x*x*y>
  double x = p[0];
  double y = p[1];

  Tensor<1,dim> return_value;

  switch(component){
    case 0:  // gradient of 1st component of velocity
      return_value[0] = -M_PI*sin(M_PI*x);      // u_{1,x}
      return_value[1] = 0.;                     // u_{1,y}
      return return_value;
      break;
    case 1:  // gradient of 2nd component of velocity
      return_value[0] = y*M_PI*M_PI*cos(M_PI*x);      // u_{2,x}
      return_value[1] = M_PI*sin(M_PI*x);             // u_{2,y}
      return return_value;
      break;
    case 2:
      return_value[0] = 2*x*y*y;
      return_value[1] = 2*x*x*y;
      return return_value;
      break;
    default:
      Assert(false, ExcNotImplemented());
  }

}



template<int dim>
class StokesBoundaryValuesDirichlet : public Function<dim>
{
  public:
    StokesBoundaryValuesDirichlet () : Function<dim> (dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                                Vector<double> &value) const;
};

template <int dim>
void StokesBoundaryValuesDirichlet<dim>::vector_value (const Point<dim> &p,
                                                       Vector<double> &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));

  double x = p[0];
  double y = p[1];

  values(0) = cos(M_PI*x);        // only velocity boundary conditions
  values(1) = y*M_PI*sin(M_PI*x);
  values(2) = 0.;                 // no pressure boundary conditions

}



template<int dim>
class StokesRightHandSide : public Function<dim>
{
  public:
    StokesRightHandSide() : Function<dim> (dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                                Vector<double> &values) const;
};

template<int dim>
void StokesRightHandSide<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));

  double x = p[0];
  double y = p[1];

  values(0) = cos(M_PI*x)*M_PI*M_PI + 2*x*y*y;           // f_1
  values(1) = sin(M_PI*x)*y*M_PI*M_PI*M_PI + 2*y*x*x;    // f_2
  values(2) = 0.;  // right-hand side of continuity equation is zero

}



template <int dim>
class StokesProblem
{
  public:
    StokesProblem (const unsigned int degree);  // constructor
    void run ();                                

    private:
      void make_grid();                         // generate domain
      void set_boundary();                      // set boundary ids
      void setup_dofs ();                       // distribute degrees of freedom
      void set_pressure_constraint ();          // need to set pressure for unique solution
      void setup_system();                      // set matrix pattern
      void assemble_system();                   // assemble weak form
      void stokes_solve ();                     // direct solver UMFPACK

      void compute_errors (const unsigned int cycle);

      void refine_grid();
      void solution_update ();
      void setup_convergence_table();
      void output_results (const unsigned int cycle);

      const unsigned int  degree;

      Triangulation<dim>      triangulation;
      FESystem<dim>           fe;
      DoFHandler<dim>         dof_handler;

      SparsityPattern              sparsity_pattern;
      SparseMatrix<double>         system_matrix;

      Vector<double>               solution;
      Vector<double>               system_rhs;

      ConvergenceTable             convergence_table;

      AffineConstraints<double>    zero_press_node_constraint;
};


//constructor
template<int dim>
StokesProblem<dim>::StokesProblem (const unsigned int degree)
        :
        degree(degree),
        fe (FE_Q<dim>(degree+1), dim,  // dim velocity components
            FE_Q<dim>(degree), 1),     // 1 pressure component
        dof_handler(triangulation)
{}

template<int dim>
void StokesProblem<dim>::make_grid()
{
  std::vector<unsigned int> subdivisions (dim,1);
  for (unsigned int i=0; i<dim; ++i)
    subdivisions[i] = 1;

  const Point<dim> bottom_left = (dim == 2 ?
                                    Point<dim> (0,0) :
                                    Point<dim> (0,0,0));
  
  const Point<dim> top_right   = (dim == 2 ?
                                    Point<dim> (1,1) :
                                    Point<dim> (1,1,1));

  GridGenerator::subdivided_hyper_rectangle (triangulation,
                                             subdivisions,
                                             bottom_left,
                                             top_right);

  triangulation.refine_global(1);

}


template<int dim> 
void StokesProblem<dim>::set_boundary()
{
  unsigned int x_equal_0 = 1;
  unsigned int y_equal_0 = 2;
  unsigned int x_equal_1 = 3;
  unsigned int y_equal_1 = 4;

  for (typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
  {
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (std::fabs(cell->face(f)->center()[0] - 1.) < 1e-12)  // x = 1 boundary
        cell->face(f)->set_all_boundary_ids(x_equal_1);
      if (std::fabs(cell->face(f)->center()[1] - 1.) < 1e-12)  // y = 1 boundary
        cell->face(f)->set_all_boundary_ids(y_equal_1);
      if (std::fabs(cell->face(f)->center()[0] - 0.) < 1e-12)  // x = 0 boundary
        cell->face(f)->set_all_boundary_ids(x_equal_0); 
      if (std::fabs(cell->face(f)->center()[1] - 0.) < 1e-12)  // y = 0 boundary
        cell->face(f)->set_all_boundary_ids(y_equal_0);
    }
  }

}

template<int dim>
void StokesProblem<dim>::setup_dofs()
{
  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
  std::cout << "Total number of cells: "
            << triangulation.n_cells()
            << std::endl;

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
}


template<int dim>
void StokesProblem<dim>::set_pressure_constraint()
{
  std::vector<bool> x_equal_zero_press_dofs (dof_handler.n_dofs(), false);
  std::vector<bool> y_equal_zero_press_dofs (dof_handler.n_dofs(), false);
  std::vector<bool> x_and_y_equal_zero_press_dofs (dof_handler.n_dofs(), false);

  std::vector<bool> press_select(dim+1, false);
  press_select[dim] = true;

  std::set<unsigned int> bdy_indic_x_equal_zero;
  std::set<unsigned int> bdy_indic_y_equal_zero;

  unsigned int bdy_x_equal_0 = 1;
  unsigned int bdy_y_equal_0 = 2;
  //unsigned int bdy_x_equal_1 = 3;
  //unsigned int bdy_y_equal_1 = 4;

  bdy_indic_x_equal_zero.insert(bdy_x_equal_0);
  bdy_indic_y_equal_zero.insert(bdy_y_equal_0);

  DoFTools::extract_boundary_dofs (dof_handler,
                                   press_select,
                                   x_equal_zero_press_dofs, // output
                                   bdy_indic_x_equal_zero);

  DoFTools::extract_boundary_dofs (dof_handler,
                                   press_select,
                                   y_equal_zero_press_dofs,
                                   bdy_indic_y_equal_zero);

  for (unsigned int k=0; k<dof_handler.n_dofs(); ++k)
  {
    if (x_equal_zero_press_dofs[k] == true && y_equal_zero_press_dofs[k] == true)
      x_and_y_equal_zero_press_dofs[k] = true;
  }  

  const unsigned int zero_zero_press_dof_index               // (x,y) = (0,0)
    = std::distance (x_and_y_equal_zero_press_dofs.begin(),
                     std::find (x_and_y_equal_zero_press_dofs.begin(),
                                x_and_y_equal_zero_press_dofs.end(),
                                true));

  zero_press_node_constraint.clear();

  zero_press_node_constraint.add_line (zero_zero_press_dof_index); 

  zero_press_node_constraint.close();
                                
}


template<int dim>
void StokesProblem<dim>::setup_system()  // set matrices and right-hand side
{
  sparsity_pattern.reinit (dof_handler.n_dofs(),
                           dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());

  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);

  zero_press_node_constraint.condense(sparsity_pattern);

  sparsity_pattern.compress();

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

}

template<int dim>
void StokesProblem<dim>::assemble_system()  
{
  system_matrix = 0;
  system_rhs = 0;
  solution = 0;

  QGauss<dim> quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

  const unsigned int  dofs_per_cell = fe.dofs_per_cell;
  const unsigned int  n_q_points    = quadrature_formula.size();

  FullMatrix<double>  local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>      local_rhs (dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  const StokesRightHandSide<dim>  right_hand_side;
  std::vector<Vector<double> >    rhs_values (n_q_points,
                                              Vector<double>(dim+1));

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<2,dim> >    phi_grads_u (dofs_per_cell);
  std::vector<double>            div_phi_u (dofs_per_cell);
  std::vector<double>            phi_p (dofs_per_cell);

  double product_grad_phis = 0;

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  for ( ; cell!=endc; ++cell)
  {
    fe_values.reinit(cell);

    local_matrix = 0;
    local_rhs = 0;
   
    // evaluating right-hand side function at quadrature points of cell
    right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                      rhs_values);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        phi_grads_u[k] = fe_values[velocities].gradient(k,q);
        div_phi_u[k]   = fe_values[velocities].divergence(k,q);
        phi_p[k]       = fe_values[pressure].value(k,q);
      }
  
      for (unsigned i=0; i<dofs_per_cell; ++i)
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const unsigned int component_j = fe.system_to_component_index(j).first;

          if ( component_i == component_j && component_i != dim )
            product_grad_phis = fe_values.shape_grad(i,q) * 
                                fe_values.shape_grad(j,q);
          else
            product_grad_phis = 0;

          local_matrix(i,j) += ( product_grad_phis
                                 - div_phi_u[i] * phi_p[j]
                                 - phi_p[i] * div_phi_u[j] )
                                * fe_values.JxW(q);

        } // loop over columns of matrix 

        local_rhs(i) += (fe_values.shape_value(i,q) *
                         rhs_values[q](component_i)) *
                         fe_values.JxW(q);
      
      } // loop over rows i

    } // loop over quadrature points of the cell

    // mapping from the cell dofs to the system matrix
    cell->get_dof_indices (local_dof_indices);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        system_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           local_matrix(i,j));

      system_rhs(local_dof_indices[i]) += local_rhs(i);
    } 

  } // loop over the cells

  // incorporate set pressure value at (x,y) = (0,0)
  zero_press_node_constraint.condense (system_matrix);
  zero_press_node_constraint.condense (system_rhs);

  // set boundary conditions for velocity
  std::vector<bool> component_mask (dim+1, true); // just velocity components true
  component_mask[dim] = false;

  const unsigned int bdy_x_equal_0 = 1;
  const unsigned int bdy_y_equal_0 = 2;
  const unsigned int bdy_x_equal_1 = 3;
  const unsigned int bdy_y_equal_1 = 4;

  std::map<unsigned int, double> boundary_values;

  VectorTools::interpolate_boundary_values (dof_handler,
                                            bdy_x_equal_0,
                                            StokesBoundaryValuesDirichlet<dim>(),
                                            boundary_values,
                                            component_mask);

  VectorTools::interpolate_boundary_values (dof_handler,
                                            bdy_y_equal_0,
                                            StokesBoundaryValuesDirichlet<dim>(),
                                            boundary_values,
                                            component_mask);
  
  VectorTools::interpolate_boundary_values (dof_handler,
                                            bdy_x_equal_1,
                                            StokesBoundaryValuesDirichlet<dim>(),
                                            boundary_values,
                                            component_mask);

  VectorTools::interpolate_boundary_values (dof_handler,
                                            bdy_y_equal_1,
                                            StokesBoundaryValuesDirichlet<dim>(),
                                            boundary_values,
                                            component_mask);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs); 

}


template <int dim>
void StokesProblem<dim>::stokes_solve()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);

  A_direct.vmult(solution, system_rhs);

}

template <int dim>
void StokesProblem<dim>::compute_errors()
{
  // component masks
  const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
  const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0,dim), dim+1);

  StokesExactSolution<dim> exact_solution;

  QTrapez<1>  q_trapez;
  QIterated<dim> quadrature (q_trapez, degree+2);

  // vector to collect error calculations for each cell
  Vector<double> cellwise_errors (triangulation.n_active_cells());

  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm,
                                     &pressure_mask);
  const double p_l2_error = cellwise_errors.l2_norm();

  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm,
                                     &velocity_mask);
  const double u_l2_error = cellwise_errors.l2_norm();

  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                     cellwise_errors, quadrature,
                                     VectorTools::H1_seminorm,
                                     &velocity_mask);
  
  const double u_h1_error = cellwise_errors.l2_norm();

  std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
            << ",  ||e_u||_L2 = " << u_l2_error
            << ",  ||e_u||_H1 = " << u_h1_error
            << std::endl;

  const unsigned int n_active_cells = triangulation.n_active_cells();
  const unsigned int n_dofs         = dof_handler.n_dofs();
  const unsigned int dofs_per_cell  = fe.dofs_per_cell;

  std::cout << "  Number of active cells:      " << n_active_cells
            << std::endl
            << "  Number of degrees of freedom: " << n_dofs << std::endl
            << "  Degrees of freedom per cell:  " << dofs_per_cell
            << std::endl;

}

template <int dim>
void StokesProblem<dim>::run()
{
  // number of global refinements to peform on domain [0,1] with initially 4 cells
  // we increment to test convergence rates
  const unsigned int n_refinements = 3;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " Setting up domain and mesh..." << std::endl;
  make_grid();
  std::cout << " Completed Setting up domain and mesh..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Setting up boundary identifies..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  set_boundary();
  std::cout << " Completed Setting up boundary identifiers..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Refining the mesh globally..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (unsigned int i=0; i<n_refinements; ++i)
   refine_grid();
  std::cout << " Completed Refining the mesh globally..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  setup_dofs();
  std::cout << " Completed Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  set_pressure_constraint();
  std::cout << " Completed Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Setting up system pattern and size..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  setup_system();
  std::cout << " Completed Initializing DoF Handler..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Assembling system matrix..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  assemble_system();
  std::cout << " Completed Assembling system matrix..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Solving system..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  stokes_solve();
  std::cout << " Completed Solving system..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << " Computing errors..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  compute_errors();
  std::cout << " Completed Computing errors..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
 

}



int main ()
{
  try
    {
      deallog.depth_console (0);

      StokesProblem<2> flow_problem(1);
      flow_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "---------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "---------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "---------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "---------------------------------------"
                << std::endl;
      return 1;
    }


  return 0;
}

                
