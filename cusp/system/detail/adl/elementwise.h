/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a count of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

// the purpose of this header is to #include the elementwise.h header
// of the sequential, host, and device systems. It should be #included in any
// code which uses adl to dispatch elementwise

#include <cusp/system/detail/sequential/elementwise.h>

#define __CUSP_HOST_SYSTEM_ELEMENTWISE_HEADER <__CUSP_HOST_SYSTEM_ROOT/detail/elementwise.h>
#include __CUSP_HOST_SYSTEM_ELEMENTWISE_HEADER
#undef __CUSP_HOST_SYSTEM_ELEMENTWISE_HEADER

#define __CUSP_DEVICE_SYSTEM_ELEMENTWISE_HEADER <__CUSP_DEVICE_SYSTEM_ROOT/detail/elementwise.h>
#include __CUSP_DEVICE_SYSTEM_ELEMENTWISE_HEADER
#undef __CUSP_DEVICE_SYSTEM_ELEMENTWISE_HEADER
