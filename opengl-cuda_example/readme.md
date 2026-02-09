# OpenGL-CUDA interop example: nbody

This is a project I (alexis) did for graphics class last semester.
It is a realtime n-body simulation which shows the gravitational interaction between many points.
Rendering is done with OpenGL through SDL.
The simulation is updated using CUDA.

The idea here is to use a similar data pipeline for the digital twin simulation.

I'll comment this code, maybe make a few changes, and try to get the compilation a little smoother.

Due to CUDA, this program only works on machines with an NVIDIA GPU.

A lot of the OpenGL boilerplate was provided by CSCI 5607, but most of it seems to be standard.
