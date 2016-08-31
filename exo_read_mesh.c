/*
 * exo_read_mesh.c
 *
 *  Created on: Jun 30, 2016
 *      Author: wortiz
 */

#include "exodusII.h"
#include "netcdf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include "exo_read_mesh.h"

void *cudaCallocWrap(size_t nmemb, size_t size);

void *cudaMallocWrap(size_t size);

void copy_file_cpp(const char *infile, const char *outfile);

mesh_data* exo_read_mesh(const char *filename) {

  mesh_data *mesh = cudaMallocWrap(sizeof(mesh_data));

  int exoid, num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets;
  int num_side_sets;
  int error;
  int i;
  int *elem_map;
  int *num_df_per_set = NULL;
  int *df_ind = NULL;
  int *num_elem_in_block = NULL;
  int *num_nodes_per_elem = NULL;
  int *num_attr = NULL;
  int list_len;
  int CPU_word_size, IO_word_size;
  int idum;

  double *dist_fact;
  float fdum;
  float version;

  char title[MAX_LINE_LENGTH + 1], elem_type[MAX_STR_LENGTH + 1];
  char title_chk[MAX_LINE_LENGTH + 1];
  char *cdum = 0;

  CPU_word_size = sizeof(double); /* sizeof(float) */
  IO_word_size = 0; /* use what is stored in file */

  ex_opts(EX_VERBOSE | EX_ABORT);

  /* open EXODUS II files */
  exoid = ex_open(filename, /* filename path */
		  EX_READ, /* access mode = READ */
		  &CPU_word_size, /* CPU word size */
		  &IO_word_size, /* IO word size */
		  &version); /* ExodusII library version */

  ex_inquire(exoid, EX_INQ_API_VERS, &idum, &version, cdum);

  ex_inquire(exoid, EX_INQ_LIB_VERS, &idum, &version, cdum);

  /* read database parameters */

  error = ex_get_init(exoid, title, &num_dim, &num_nodes, &num_elem,
		      &num_elem_blk, &num_node_sets, &num_side_sets);

  mesh->dim = num_dim;
  mesh->num_nodes = num_nodes;
  mesh->num_elem = num_elem;
  mesh->num_mat = num_elem_blk;
  mesh->num_dirichlet = num_node_sets;

  /* Check that ex_inquire gives same title */
  error = ex_inquire(exoid, EX_INQ_TITLE, &idum, &fdum, title_chk);

  /* read nodal coordinates values and names from database */

  mesh->coord = cudaCallocWrap(sizeof(double *), 3);
  mesh->coord[0] = (double *) cudaCallocWrap(num_nodes, sizeof(double));
  if (num_dim >= 2)
    mesh->coord[1] = (double *) cudaCallocWrap(num_nodes, sizeof(double));
  else
    mesh->coord[1] = 0;

  if (num_dim >= 3)
    mesh->coord[2] = (double *) cudaCallocWrap(num_nodes, sizeof(double));
  else
    mesh->coord[2] = 0;

  error = ex_get_coord(exoid, mesh->coord[0], mesh->coord[1], mesh->coord[2]);

  /* read element block parameters */

  if (num_elem_blk > 0) {
    mesh->mat_ids = (int *) cudaCallocWrap(num_elem_blk, sizeof(int));
    num_elem_in_block = (int *) cudaCallocWrap(num_elem_blk, sizeof(int));
    num_nodes_per_elem = (int *) cudaCallocWrap(num_elem_blk, sizeof(int));
    num_attr = (int *) calloc(num_elem_blk, sizeof(int));

    error = ex_get_elem_blk_ids(exoid, mesh->mat_ids);

    for (i = 0; i < num_elem_blk; i++) {
      error = ex_get_elem_block(exoid, mesh->mat_ids[i], elem_type,
				&(num_elem_in_block[i]), &(num_nodes_per_elem[i]),
				&(num_attr[i]));
    }

    free(num_attr);

    mesh->num_elem_in_mat = num_elem_in_block;
    mesh->num_nodes_per_elem = num_nodes_per_elem;
  }

  /* read element connectivity */

  mesh->connect = cudaMallocWrap(sizeof(int *) * mesh->num_mat);
  for (i = 0; i < num_elem_blk; i++) {
    if (num_elem_in_block[i] > 0) {
      mesh->connect[i] = (int *) cudaCallocWrap(
					(num_nodes_per_elem[i] * num_elem_in_block[i]),
					sizeof(int));

      error = ex_get_elem_conn(exoid, mesh->mat_ids[i], mesh->connect[i]);
    }
  }

  /* read individual node sets */
  if (num_node_sets > 0) {

    /* read concatenated node sets; this produces the same information as
     * the above code which reads individual node sets
     */

    error = ex_inquire(exoid, EX_INQ_NODE_SETS, &num_node_sets, &fdum,
		       cdum);

    mesh->dirichlet_ids = (int *) cudaCallocWrap(num_node_sets, sizeof(int));
    mesh->nodes_per_dirichlet = (int *) cudaCallocWrap(num_node_sets, sizeof(int));
    num_df_per_set = (int *) calloc(num_node_sets, sizeof(int));
    mesh->dirichlet_node_index = (int *) cudaCallocWrap(num_node_sets + 1,
						sizeof(int));
    df_ind = (int *) calloc(num_node_sets, sizeof(int));

    error = ex_inquire(exoid, EX_INQ_NS_NODE_LEN,
		       &mesh->dirichlet_node_list_len, &fdum, cdum);

    mesh->dirichlet_node_list = (int *) cudaCallocWrap(
					       mesh->dirichlet_node_list_len, sizeof(int));

    error = ex_inquire(exoid, EX_INQ_NS_DF_LEN, &list_len, &fdum, cdum);

    dist_fact = (double *) calloc(list_len, sizeof(double));

    error = ex_get_concat_node_sets(exoid, mesh->dirichlet_ids,
				    mesh->nodes_per_dirichlet, num_df_per_set,
				    mesh->dirichlet_node_index, df_ind, mesh->dirichlet_node_list,
				    dist_fact);

    mesh->dirichlet_node_index[num_node_sets] =
      mesh->dirichlet_node_list_len;

    free(df_ind);
    free(num_df_per_set);
    free(dist_fact);

  }

  error = ex_close(exoid);

  int * elem_per_node = calloc(sizeof(int), num_nodes);

  int mat;
  int elem;
  int node;
  for (mat = 0; mat < mesh->num_mat; mat++) {
    for (elem = 0; elem < mesh->num_elem_in_mat[mat]; elem++) {
      for (node = 0; node < mesh->num_nodes_per_elem[mat]; node++) {
	int idx = elem * mesh->num_nodes_per_elem[mat] + node;
	int gnn = mesh->connect[mat][idx] - 1;
	elem_per_node[gnn] += 1;
      }
    }
  }

  mesh->max_elem_per_node = 0;
  for (node = 0; node < num_nodes; node++) {
    if (elem_per_node[node] > mesh->max_elem_per_node) {
      mesh->max_elem_per_node = elem_per_node[node];
    }
  }

  free(elem_per_node);

  mesh->node_elem = cudaCallocWrap(sizeof(int *), num_nodes);
  for (node = 0; node < num_nodes; node++) {
    mesh->node_elem[node] = cudaCallocWrap(sizeof(int), mesh->max_elem_per_node);
  }

  for (mat = 0; mat < mesh->num_mat; mat++) {
    for (elem = 0; elem < mesh->num_elem_in_mat[mat]; elem++) {
      for (node = 0; node < mesh->num_nodes_per_elem[mat]; node++) {
	int idx = elem * mesh->num_nodes_per_elem[mat] + node;
	int gnn = mesh->connect[mat][idx] - 1;
	for (int i = 0; i < mesh->max_elem_per_node; i++) {
	  if (mesh->node_elem[gnn][i] == 0) {
	    mesh->node_elem[gnn][i] = elem + 1;
	    break;
	  }
	}
      }
    }
  }

  return mesh;
}

void exo_write_results(double time_value, mesh_data* mesh, char *infile, char *outfile, double **values) {
  int inexoid;
  static int outexoid = 0;
  static int timeidx = 0;

  int CPU_word_size = sizeof(double); /* sizeof(float) */
  int IO_word_size = 0; /* use what is stored in file */
  float version;

  if (outexoid == 0) {
    /* open EXODUS II files */

    //    inexoid = ex_open(infile, EX_READ, &CPU_word_size, &IO_word_size, &version);

    copy_file_cpp(infile, outfile);

    //ex_copy(inexoid, outexoid);
    //    ex_close(inexoid);  
    outexoid = ex_open(outfile, EX_WRITE, &CPU_word_size, &IO_word_size, &version);
  
    ex_put_var_param(outexoid, "n", 3);

    char * var_names[3];
    var_names[0] = "VX";
    var_names[1] = "VY";
    var_names[2] = "P";

    ex_put_var_names(outexoid, "n", 3, var_names);
  } else {
    outexoid = ex_open(outfile, EX_WRITE, &CPU_word_size, &IO_word_size, &version);
  }

  timeidx++;
  ex_put_time(outexoid, timeidx, &time_value);

  ex_put_nodal_var(outexoid, timeidx, 1, mesh->num_nodes, values[0]);
  ex_put_nodal_var(outexoid, timeidx, 2, mesh->num_nodes, values[1]);
  ex_put_nodal_var(outexoid, timeidx, 3, mesh->num_nodes, values[2]);


  ex_close(outexoid);
}
