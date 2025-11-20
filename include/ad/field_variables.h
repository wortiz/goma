#ifndef GOMA_AD_FIELD_VARIABLES_H
#define GOMA_AD_FIELD_VARIABLES_H

#include "ad/structs.h"
#ifdef __cplusplus
int ad_load_bf_grad(void);
extern "C" {
#endif
void fill_ad_field_variables(void);
#ifdef __cplusplus
}
#endif

#endif // GOMA_AD_FIELD_VARIABLES_H