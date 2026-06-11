#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Async migration API: PCIe transfer pipelining with double buffering.
// These functions use a dedicated migration CUDA stream, separate from
// the compute stream, allowing H2D/D2H copies to overlap with GPU kernels.
//
// Usage:
//   void * ev = ggml_backend_cuda_migrate_event_create(backend);
//   ggml_backend_cuda_set_tensor_migrate_async(backend, t, data, 0, n);
//   ggml_backend_cuda_migrate_event_record(backend, ev);
//   ggml_backend_cuda_wait_migration_event(backend, ev);  // on compute stream
//   ggml_backend_cuda_migrate_event_destroy(ev);

GGML_BACKEND_API void   ggml_backend_cuda_set_tensor_migrate_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
GGML_BACKEND_API void   ggml_backend_cuda_get_tensor_migrate_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
GGML_BACKEND_API void * ggml_backend_cuda_migrate_event_create(ggml_backend_t backend);
GGML_BACKEND_API void   ggml_backend_cuda_migrate_event_destroy(void * event);
GGML_BACKEND_API void   ggml_backend_cuda_migrate_event_record(ggml_backend_t backend, void * event);
GGML_BACKEND_API void   ggml_backend_cuda_wait_migration_event(ggml_backend_t backend, void * event);

#ifdef  __cplusplus
}
#endif
