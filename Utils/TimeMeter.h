/*
 * TimeMeter.h
 *
 *  Created on: Jul 22, 2011
 *      Author: fernando
 */

#ifndef TIMEMETER_H_
#define TIMEMETER_H_

#include <time.h>

class TimeMeter {
public:
	TimeMeter();
	void start();
	void stop();
	double getInterval();
	void printTime();
	virtual ~TimeMeter();
private:
	clock_t start_time;
	clock_t stop_time;
};

#endif /* TIMEMETER_H_ */
