#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>

#define N 1000000 //particles
#define Lx 100 //length
#define CELLS_NUM 10000 //cells

#define PI 3.14159265358979323846f

#define Q(k) ((k) % 2 ? -1 : 1)
#define M(k) ((k) % 2 ? 1 : 1836)

#define E(x) sin(2.0f * PI / CELLS_NUM * (x))

#define INITIAL_VELOCITY 0.0f
#define TIME_INTERVAL 0.01f

struct Particle
{
	float coord[2],
		velocity[2];
	short mass;
	char charge;
};

float distributionFunction(size_t particleNumber)
{
	return (float)particleNumber * Lx / (float)N;
}

void initialize(Particle *particles)
{
	for (size_t i = 0; i < N; i++)
	{
		particles[i].coord[0] = distributionFunction(i);
		particles[i].coord[1] = particles[i].coord[0];
		particles[i].velocity[0] = INITIAL_VELOCITY;
		particles[i].velocity[1] = INITIAL_VELOCITY;
		particles[i].mass = M(i);
		particles[i].charge = Q(i);
	}
}
 
int main(int argc, char **argv)
{
	Particle *particles = new Particle[N];
	
	initialize(particles);

	char period = 1;
	size_t *nums = new size_t[CELLS_NUM];
	float *density = new float[2 * CELLS_NUM],
		*charge = density + CELLS_NUM,
		cellSize = (float)Lx / (CELLS_NUM - 1); //cell size
		
	for (size_t ts = 0; ts < 50; ts++)
	{
		period ^= 1;

		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			density[c] = 0.0f;
			nums[c] = 0;
		}

		std::string coordfile = "coord" + std::to_string(ts);
		freopen(coordfile.c_str(), "w", stdout);

		for (size_t p = 0; p < N; p++)
		{
			particles[p].coord[period] = particles[p].coord[period ^ 1]
				+ particles[p].velocity[period] * TIME_INTERVAL;

			particles[p].velocity[period] = particles[p].velocity[period ^ 1]
				+ 2.0f * particles[p].charge * E(particles[p].coord[period]) / particles[p].mass;

			printf("%f %f\n", particles[p].coord[period], particles[p].velocity[period]);

			float index = particles[p].coord[period] / cellSize;

			size_t i = (index - (int)index > 0.5f ? index + 1 : index);

			charge[i] += (float)particles[p].charge;
			nums[i]++;
		}

		std::string densfile = "dens" + std::to_string(ts);
		freopen(densfile.c_str(), "w", stdout);

		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			density[c] = charge[c] / nums[c];
			printf("%f\n", density[c]);
		}
	}

	delete[] nums;
	delete[] density;
	delete[] particles;
	return 0;
}