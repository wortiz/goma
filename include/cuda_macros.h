/*
 * cuda_macros.h
 *
 *  Created on: Aug 9, 2016
 *      Author: wortiz
 */

#ifndef CUDA_MACROS_H_
#define CUDA_MACROS_H_
#include <cuda.h>

#define delta(m,n)	((m) == (n) ? 1 : 0 ) /* Kroenecker delta */

#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }


#endif /* CUDA_MACROS_H_ */
