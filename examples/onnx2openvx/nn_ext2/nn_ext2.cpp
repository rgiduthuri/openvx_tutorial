/*
* Copyright (c) 2019 <copyright holders>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "nn_ext2.h"
#include "nn_ext2_common.hpp"
#include "concat_kernel.hpp"
#include "reshape_kernel.hpp"

//////////////////////////////////////////////////////////////////////
//! \brief Load kernels of this module into a context
VX_API_EXPORT vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    // register kernels
    ERROR_CHECK_STATUS(registerConcatKernel(context));
    ERROR_CHECK_STATUS(registerReshapeKernel(context));

    return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////
//! \brief Unload kernels of this module from the context
VX_API_EXPORT vx_status VX_API_CALL vxUnpublishKernels(vx_context context)
{
    //TODO
    return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////
//! \brief Load kernels of this module into a context
VX_API_EXPORT vx_status VX_API_CALL vxLoadKernelsNnExt2(vx_context context)
{
    ERROR_CHECK_STATUS(vxPublishKernels(context));
    return VX_SUCCESS;
}

