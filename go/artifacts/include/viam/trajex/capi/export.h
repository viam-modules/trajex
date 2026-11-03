
#ifndef VIAM_TRAJEX_CAPI_EXPORT_H
#define VIAM_TRAJEX_CAPI_EXPORT_H

#ifdef VIAM_TRAJEX_CAPI_STATIC_DEFINE
#  define VIAM_TRAJEX_CAPI_API
#  define VIAM_TRAJEX_CAPI_HIDDEN
#else
#  ifndef VIAM_TRAJEX_CAPI_API
#    ifdef VIAM_TRAJEX_CAPI_EXPORTS
        /* We are building this library */
#      define VIAM_TRAJEX_CAPI_API 
#    else
        /* We are using this library */
#      define VIAM_TRAJEX_CAPI_API 
#    endif
#  endif

#  ifndef VIAM_TRAJEX_CAPI_HIDDEN
#    define VIAM_TRAJEX_CAPI_HIDDEN 
#  endif
#endif

#ifndef VIAM_TRAJEX_CAPI_DEPRECATED
#  define VIAM_TRAJEX_CAPI_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef VIAM_TRAJEX_CAPI_DEPRECATED_EXPORT
#  define VIAM_TRAJEX_CAPI_DEPRECATED_EXPORT VIAM_TRAJEX_CAPI_API VIAM_TRAJEX_CAPI_DEPRECATED
#endif

#ifndef VIAM_TRAJEX_CAPI_DEPRECATED_NO_EXPORT
#  define VIAM_TRAJEX_CAPI_DEPRECATED_NO_EXPORT VIAM_TRAJEX_CAPI_HIDDEN VIAM_TRAJEX_CAPI_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef VIAM_TRAJEX_CAPI_NO_DEPRECATED
#    define VIAM_TRAJEX_CAPI_NO_DEPRECATED
#  endif
#endif

#endif /* VIAM_TRAJEX_CAPI_EXPORT_H */
