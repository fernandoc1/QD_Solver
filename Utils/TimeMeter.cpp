/*
 * TimeMeter.cpp
 *
 *  Created on: Jul 22, 2011
 *      Author: fernando
 */

#include "TimeMeter.h"
#include <stdio.h>

TimeMeter::TimeMeter()
{

}

TimeMeter::~TimeMeter()
{

}

void TimeMeter::start()
{
	start_time=clock();
}

void TimeMeter::stop()
{
	stop_time=clock();
}

double TimeMeter::getInterval()
{
	return ((double)stop_time-start_time)/CLOCKS_PER_SEC;
}

void TimeMeter::printTime()
{
	printf("Execution time: %6.3fs.\n", getInterval());
}

