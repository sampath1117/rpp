/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 - 2022 Advanced Micro Devices, Inc.
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
 *
 *******************************************************************************/

#include <boost/filesystem.hpp>
#include<thread>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "config.h"
#include "rpp/logger.hpp"
#include "rpp/handle.hpp"

namespace rpp {

#if !GPU_SUPPORT

struct HandleImpl
{
    size_t nBatchSize = 1;
    size_t internalBatchSize;
    InitHandle* initHandle = nullptr;

    uint compute_internal_batch_size(Rpp32u user_batch_size)
    {
      const unsigned MINIMUM_CPU_THREAD_COUNT = 2;
      const unsigned DEFAULT_SMT_COUNT = 2;
      unsigned THREAD_COUNT = std::thread::hardware_concurrency();
      // if(THREAD_COUNT >= MINIMUM_CPU_THREAD_COUNT)
      // {
      //     INFO("Can run " + TOSTR(THREAD_COUNT) + " threads simultaneously on this machine")
      // }
      if(THREAD_COUNT < MINIMUM_CPU_THREAD_COUNT)
      {
          THREAD_COUNT = MINIMUM_CPU_THREAD_COUNT;
          std::cerr<<"hardware_concurrency() call failed assuming can run "<<THREAD_COUNT<<" threads";
      }
      size_t ret = user_batch_size;
      size_t CORE_COUNT = THREAD_COUNT / DEFAULT_SMT_COUNT;

      if(CORE_COUNT <= 0)
      {
          std::cerr<<"\n Wrong core count detected less than 0";
          exit(0);
      }

      for( size_t i = CORE_COUNT; i <= THREAD_COUNT; i++)
      {
        if(user_batch_size % i == 0)
            {
                ret = i;
                break;
            }
        }

        for(size_t i = CORE_COUNT; i > 1; i--)
        {
        if(user_batch_size % i == 0)
            {
                    ret = i;
                break;
            }
        }
        // std::cerr<<"\n User batch size "<<(user_batch_size)<<" Internal batch size set to "<<ret;
        return ret;
    }

    void PreInitializeBufferCPU()
    {
        this->initHandle = new InitHandle();

        this->initHandle->nbatchSize = this->nBatchSize;
        this->initHandle->mem.mcpu.maxSrcSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.maxDstSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.roiPoints = (RppiROI *)malloc(sizeof(RppiROI) * this->nBatchSize);
        this->initHandle->mem.mcpu.tempFloatmem = (Rpp32f *)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize); // 7680 * 4320 * 3
    }
};

Handle::Handle(size_t batchSize) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    impl->PreInitializeBufferCPU();
    impl->internalBatchSize = impl->compute_internal_batch_size(impl->nBatchSize);
}

Handle::Handle() : impl(new HandleImpl())
{
    impl->PreInitializeBufferCPU();
    RPP_LOG_I(*this);
}

Handle::~Handle() {}

void Handle::rpp_destroy_object_host()
{
    free(this->GetInitHandle()->mem.mcpu.maxSrcSize);
    free(this->GetInitHandle()->mem.mcpu.maxDstSize);
    free(this->GetInitHandle()->mem.mcpu.roiPoints);
    free(this->GetInitHandle()->mem.mcpu.tempFloatmem);
}

size_t Handle::GetBatchSize() const
{
    return this->impl->nBatchSize;
}

size_t Handle::GetInternalBatchSize() const 
{
    return this->impl->internalBatchSize;
}

void Handle::SetBatchSize(size_t bSize) const
{
    this->impl->nBatchSize = bSize;
}

InitHandle* Handle::GetInitHandle() const
{
    return impl->initHandle;
}

#endif // GPU_SUPPORT

} // namespace rpp
