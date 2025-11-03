#ifndef PSAM_EXPORT_H
#define PSAM_EXPORT_H

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef PSAM_SHARED
    #ifdef PSAM_BUILDING_DLL
      #define PSAM_API __declspec(dllexport)
    #else
      #define PSAM_API __declspec(dllimport)
    #endif
  #else
    #define PSAM_API
  #endif
  #define PSAM_UNUSED
#else
  #if __GNUC__ >= 4
    #define PSAM_API __attribute__((visibility("default")))
  #else
    #define PSAM_API
  #endif
  #define PSAM_UNUSED __attribute__((unused))
#endif

#endif /* PSAM_EXPORT_H */
