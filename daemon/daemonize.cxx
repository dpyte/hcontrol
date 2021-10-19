#include <cstdio>
#include <cstdlib>
#include <signal.h>
#include <sys/stat.h>
#include <syslog.h>
#include <unistd.h>

#include "daemonize.hxx"

namespace {
constexpr char *null_dir = "/tmp/hcontrold-input";
constexpr char *default_dir = "/";
constexpr char *default_daemon_name = "hcontrold";
}

int HCONTROLD::DAEMON::daemonize(char *name, char *path, char *outfile, char *errfile, char *infile) {
	if (!path) path = static_cast<char *>(default_dir);
	if (!name) name = static_cast<char *>(default_daemon_name);
	if (!infile) infile = static_cast<char *>(null_dir);
	if (!outfile) outfile = static_cast<char *>(null_dir);
	if (!errfile) errfile = static_cast<char *>(null_dir);

	auto pid = fork();
	if (pid < 0) {
		std::fprintf(stderr, "hcontrold: failed to fork\n");
		exit(EXIT_FAILURE);
	}
	if (pid > 0) exit(EXIT_SUCCESS);
	if (setsid() < 0) {
		std::fprintf(stderr, "hcontrold: failed setsid\n");
		exit(EXIT_FAILURE);
	}

	signal(SIGCHLD, SIG_IGN);
	signal(SIGCHLD, SIG_IGN);
	if ((pid = fork()) < 0) {
		fprintf(stderr, "hcontrold: failed seconds fork\n");
		exit(EXIT_FAILURE);
	}
	if (pid > 0) exit(EXIT_SUCCESS);
	umask(0);
	chdir(path);

	auto fd = sysconf(_SC_OPEN_MAX);
	for (;fd >= 0; --fd) close(fd);

	stdin = fopen(infile, "r");
	stdout = fopen(outfile, "w+");
	stderr = fopen(errfile, "w+");

	openlog(name, LOG_PID, LOG_DAEMON);
	return 0;
}
