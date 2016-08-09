/*
 * exo_read_mesh.h
 *
 *  Created on: Jun 30, 2016
 *      Author: wortiz
 */

#ifndef EXO_READ_MESH_H_
#define EXO_READ_MESH_H_

typedef struct {
  int num_nodes;
  int num_elem;
  int num_elements;
  int dim;
  int **connect;
  double **coord;
  int num_mat;
  int *mat_ids;
  int *num_elem_in_mat;
  int *num_nodes_per_elem;
  int num_dirichlet;
  int *dirichlet_ids;
  int *nodes_per_dirichlet;
  int *dirichlet_node_index;
  int *dirichlet_node_list;
  int dirichlet_node_list_len;
  int **node_elem;
  int max_elem_per_node;

} mesh_data;

mesh_data* exo_read_mesh(const char *filename);
void exo_write_results(mesh_data *mesh, const char *infile, const char *outfile, double **values);

#endif /* EXO_READ_MESH_H_ */
