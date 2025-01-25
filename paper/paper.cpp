/*
NAME
Appendix B
Source Code for kapogee
kapogee.c -  A third order Kalman filter for
RDAS raw data files.
SYNOPSIS
       kapogee <infile
DESCRIPTION:
Performs Kalman filtering on standard input
data using a third order, or constant acceleration, propagation model.
AUTHOR
*/
The standard input data is of the form:
Column 1: Time of the measurement (seconds) Column 2: Acceleration Measurement (ignored) Column 3: Pressure Measurement (ADC counts)
All arithmetic is performed using 32 bit floats.
The standard output data is of the form:
Liftoff detected at time: <time>
Apogee detected at time:  <time>
David Schultz
#include <stdio.h>
#include <string.h>
#include <math.h>
#define  MEASUREMENTSIGMA     0.44
#define  MODELSIGMA           0.002
#define
#define
main() {
MEASUREMENTVARIANCE MEASUREMENTSIGMA*MEASUREMENTSIGMA MODELVARIANCE MODELSIGMA*MODELSIGMA
int liftoff = 0; char buf[512];
float time, accel, pressure; float last_time, last_pressure; float est[3]={0,0,0}; float estp[3] = {0, 0, 0 };
float pest[3][3]
float  pestp[3][3]
float phi[3][3]
float  phit[3][3]
= { 0.002, 0, 0,
   0, 0.004, 0,
   0, 0, 0.002 };
= { 0, 0, 0,
    0, 0, 0,
    0, 0, 0 };
= { 1, 0, 0,
0, 1, 0,
  0, 0, 1.0 };
= { 1, 0, 0,
0, 1, 0,
0, 0, 1.0 };
float gain[3] = { 0.010317, 0.010666, 0.004522 };
float dt;
float term[3][3];
/* Initialize */