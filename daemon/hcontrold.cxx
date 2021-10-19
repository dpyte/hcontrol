#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <thread>
#include <string>
#include <sys/types.h>
#include <syslog.h>
#include <unistd.h>

#include "daemonize.hxx"

namespace {

constexpr const auto proc_name = "hcontrold";

int check_for_running_process_id() {
	pid_t pid = -1;
	auto *dp = opendir("/proc");
	if (dp != nullptr) {
		struct dirent *dirp;
		while (pid < 0 && (dirp = readdir(dp))) {
			const auto id = std::atoi(dirp->d_name);
			if (id > 0) {
				std::string cmd_line;
				const auto cmd_path = std::string("/proc") + dirp->d_name + "/cmdline";
				std::ifstream cmd_file(cmd_path.c_str());
				std::getline(cmd_file, cmd_line);
				if (not cmd_line.empty()) {
					auto pos = cmd_line.find('\0');
					if (pos != std::string::npos) cmd_line = cmd_line.substr(0, pos);
					pos = cmd_line.rfind('/');
					if (pos != std::string::npos) cmd_line = cmd_line.substr(pos + 1);
					if (proc_name == cmd_line) pid = id;
				}
			}
		}
	}
	closedir(dp);
	return pid;
}

void hcd_daemon_process() {
	std::string line;
	std::ifstream proc_file("/tmp/hcontrold-input");
	try {

	} catch (...) {
		std::fprintf(stderr, "hcontrold :: err - error connecting to '/tmp/hcontrold-input' "
				"(no such file or directory)\n");
	}
}

}

int main() {
	auto res = HCONTROLD::DAEMON::daemonize("hcontrold", "/tmp", "/tmp/hcontrold-input", nullptr, nullptr);
	if (res) {
		// daemonization is not even a real word
		fprintf(stderr, "hcontrold: err - failed daemonization\n");
		exit(EXIT_FAILURE);
	}

	int (*check_prior_proc)() = &check_for_running_process_id;
	if (check_prior_proc() <= 0) {
		std::fprintf(stderr, "hcontrold: err - failed to start the daemon\n"
				"         - daemon is already running\n");
	}

	for (;;) {
		std::this_thread::sleep_for(std::chrono::microseconds(1));
		hcd_daemon_process();
	}

	closelog();
	return EXIT_SUCCESS;
}
