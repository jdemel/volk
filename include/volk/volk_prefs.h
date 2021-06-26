#ifndef INCLUDED_VOLK_PREFS_H
#define INCLUDED_VOLK_PREFS_H

#include <stdbool.h>
#include <stdlib.h>
#include <volk/volk_common.h>


typedef struct volk_arch_pref {
    char name[128];   // name of the kernel
    char impl_a[128]; // best aligned impl
    char impl_u[128]; // best unaligned impl
} volk_arch_pref_t;

////////////////////////////////////////////////////////////////////////
// get path to volk_config profiling info; second arguments specifies
// if config file should be tested on existence for reading.
// returns \0 in the argument on failure.
////////////////////////////////////////////////////////////////////////
VOLK_API void volk_get_config_path(char*, bool);

////////////////////////////////////////////////////////////////////////
// load prefs into global prefs struct
////////////////////////////////////////////////////////////////////////
VOLK_API size_t volk_load_preferences(volk_arch_pref_t**);


#endif // INCLUDED_VOLK_PREFS_H
