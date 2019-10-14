/*
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
 * \file    common.h
 * \brief   some used macros
 * \author  Radhakrishna Giduthuri <radhakrishna.giduthuri@ieee.org>
 */

#ifndef common_h__
#define common_h__

#include <stdio.h>
#include <stdlib.h>

////////
// Useful macros for OpenVX/OpenCL error checking:
//   ERROR_CHECK_NOT_NULL   - check and abort if returned value is NULL
//   ERROR_CHECK_STATUS     - check and abort if returned status is an error
//   ERROR_CHECK_STATUS_RET - check and return if status is an error
//
#define ERROR_CHECK_NOT_NULL( obj ) { \
        if((obj) == NULL) { \
            printf("ERROR: returned NULL at " __FILE__ "#%d\n", __LINE__); \
            exit(1); \
        } \
    }
#define ERROR_CHECK_STATUS( status ) { \
        auto status_ = (status); \
        if(status_) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }
#define ERROR_CHECK_STATUS_RET( status ) { \
        auto status_ = (status); \
        if(status_) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            return status_; \
        } \
    }

#endif
