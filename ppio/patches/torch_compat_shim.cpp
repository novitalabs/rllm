// Compatibility shim for vllm-flash-attn FA2 on torch 2.10.x
// Bridges: c10::cuda::c10_cuda_check_implementation int->unsigned int change

extern "C" {
    extern void _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb(
        int error, const char* msg, const char* file, unsigned int line, bool warn);
    extern void _ZN3c1013MessageLoggerC1EPKciib(
        void* thisptr, const char* file, int line, int severity, bool flag);

    void _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib(
        int error, const char* msg, const char* file, int line, bool warn) {
        _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb(
            error, msg, file, (unsigned int)line, warn);
    }

    void _ZN3c1013MessageLoggerC1EPKcii(
        void* thisptr, const char* file, int line, int severity) {
        _ZN3c1013MessageLoggerC1EPKciib(thisptr, file, line, severity, false);
    }
}
