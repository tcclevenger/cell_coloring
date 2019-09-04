// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// check function MGTools::max_level_for_coarse_mesh()

#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

using namespace dealii;



template <int dim>
Point<dim, unsigned int>
get_integer_coords (const CellId cell_id, const unsigned int n_global_level)
{

  std::vector<unsigned int> child_indices;

  std::string cell_id_str = cell_id.to_string();
  while (cell_id_str.size() > 4)
  {
    child_indices.insert(child_indices.begin(),
                         Utilities::string_to_int(&(cell_id_str.back())));

    cell_id_str.pop_back();
  }

//  std::cout << "Child indices: ";
//  for (auto it = child_indices.begin();
//       it != child_indices.end();
//       ++it)
//    std::cout << *it << " ";
//  std::cout << std::endl;

  const unsigned int coarse_id = cell_id.to_binary<dim>()[0];
  Point<dim,unsigned int> global_coord;
  Assert(dim==2,ExcNotImplemented());
  if (coarse_id==0 || coarse_id==2)
    global_coord(0) = 0;
  else
    global_coord(0) = 1;

  if (coarse_id==0 || coarse_id==1)
    global_coord(1) = 0;
  else
    global_coord(1) = 1;


  unsigned int level=1;
  for (auto c : child_indices)
  {
    Point<dim,unsigned int> local_coord;
    Assert(dim==2,ExcNotImplemented());
    if (c==0 || c==2)
      local_coord(0) = 0;
    else
      local_coord(0) = 1;

    if (c==0 || c==1)
      local_coord(1) = 0;
    else
      local_coord(1) = 1;


    global_coord += std::pow(dim,n_global_level-level-1)*local_coord;

    ++level;
  }

  return global_coord;
}




template <int dim>
void
test()
{
  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                            == 0));


  parallel::distributed::Triangulation<dim> tria(
        MPI_COMM_WORLD,
        Triangulation<dim>::limit_level_difference_at_vertices,
        parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global(2);





  for (unsigned int level=0; level<tria.n_global_levels(); ++level)
  {
    if (level==0)
    {
      //initialize with 0
      continue;
    }

    std::cout << level <<  ": " << std::endl;

    for (auto &cell : tria.cell_iterators_on_level(level))
      if (cell->is_locally_owned_on_level())
      {
        std::cout << cell->id().to_string() << "     ";

        Point<dim,unsigned int> cell_int_coords = get_integer_coords<dim>(cell->id(),tria.n_global_levels());
        std::cout << "(" << cell_int_coords(0) << ", " << cell_int_coords(1) << ")" << std::endl;
      }

    std::cout << std::endl;
  }




  for (auto &cell : tria.active_cell_iterators())
  {
    const unsigned int child_number = Utilities::string_to_int(&(cell->id().to_string().back()));
    unsigned int color = 0;
    if (child_number == 1 || child_number == 2)
      color = 1;

    cell->set_material_id(color);
  }

  std::ofstream file("grid-active.vtk");
  GridOut grid_out;
  grid_out.write_vtk(tria,file);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<2>();
  //test<3>();
}
