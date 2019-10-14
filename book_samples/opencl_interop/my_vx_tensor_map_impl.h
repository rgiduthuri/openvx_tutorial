/*
 * Copyright (c) 2019 Radhakrishna Giduthuri
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*!
 * \file    my_vx_tensor_map_impl.h
 * \brief   This is a simple reference implementation (not optimized)
 *          to demonstrate vxMapTensorPatch functionality, since it is
 *          not yet available in KhronosGroup/OpenVX-sample-impl repo
 * \author  Radhakrishna Giduthuri <radhakrishna.giduthuri@ieee.org>
 */

#ifndef my_vx_tensor_map_impl_h__
#define my_vx_tensor_map_impl_h__

#include <VX/vx.h>

////
// simple reference implementations (not optimized) for vxMapTensorPatch & vxUnmapTensorPatch
//   1. refer to my_vx_tensor_map_impl.cpp for the unoptimized source
//   2. the my_vx_tensor_map_impl.h & my_vx_tensor_map_impl.cpp can be removed
//      once the sample implementation supports the vxMapTensorPatch and
//      vxUnmapTensorPatch APIs
//
#define vxMapTensorPatch myVxMapTensorPatch
#define vxUnmapTensorPatch myVxUnmapTensorPatch

#ifdef  __cplusplus
extern "C" {
#endif

vx_status vxMapTensorPatch(
    vx_tensor                                   tensor,
    vx_size                                     number_of_dims,
    const vx_size*                              view_start,
    const vx_size*                              view_end,
    vx_map_id*                                  map_id,
    vx_size*                                    stride,
    void**                                      ptr,
    vx_enum                                     usage,
    vx_enum                                     mem_type);

////
// simple reference implementation (not optimized) for vxUnmapTensorPatch
//   this uses the temporary buffer used for mapping in myVxMapTensorPatch
//
vx_status vxUnmapTensorPatch(
    vx_tensor                                   tensor,
    const vx_map_id                             map_id);

#ifdef  __cplusplus
}
#endif

#endif
