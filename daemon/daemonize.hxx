
#ifndef HCONTROL_DAEMONIZE_HXX
#define HCONTROL_DAEMONIZE_HXX

#include "hcontrold-config.hxx"

namespace HCONTROLD::DAEMON {
HC_EXTERN int daemonize(char *name, char *path, char *outfile, char *errfile, char *infile);
}

#endif // HCONTROL_DAEMONIZE_HXX
