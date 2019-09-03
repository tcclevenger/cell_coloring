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


template <int n_levels>
std::array<std::uint8_t,n_levels>
get_indices(const std::array<unsigned int, 4> &binary_representation)
{
  std::array<std::uint8_t,n_levels> child_indices;
  std::fill(child_indices.begin(),
            child_indices.end(),
            std::numeric_limits<char>::max());

  // The rightmost two bits of the second entry store the dimension,
  // the rest stores the number of child indices.
  const unsigned int two_bit_mask = (1 << 2) - 1;
  const unsigned int dim          = binary_representation[1] & two_bit_mask;
  const unsigned int n_child_indices                 = (binary_representation[1] >> 2);

  Assert(n_child_indices < child_indices.size(), ExcInternalError());

  // Each child requires 'dim' bits to store its index
  const unsigned int children_per_value =
    sizeof(binary_type::value_type) * 8 / dim;
  const unsigned int child_mask = (1 << dim) - 1;

  // Loop until all child indices have been read
  unsigned int child_level  = 0;
  unsigned int binary_entry = 2;
  while (child_level < n_child_indices)
    {
      for (unsigned int j = 0; j < children_per_value; ++j)
        {
          // Read the current child index by shifting to the current
          // index's position and doing a bitwise-and with the child_mask.
          child_indices[child_level] =
            (binary_representation[binary_entry] >> (dim * j)) & child_mask;
          ++child_level;
          if (child_level == n_child_indices)
            break;
        }
      ++binary_entry;
    }

  return child_indices;
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
    }

    std::cout << level <<  ": " << std::endl;

    for (auto &cell : tria.cell_iterators_on_level(level))
      if (cell->is_locally_owned_on_level())
      {
        std::cout << cell->id().to_string() << std::endl;
      }

    std::cout << std::endl;



  }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<2>();
  //test<3>();
}
