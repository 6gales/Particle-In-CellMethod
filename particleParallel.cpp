#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <mpi.h>

#ifdef TIME_TEST
#include <sys/time.h>
#endif

#define N 100000000 //particles
#define Lx 100 //length
#define CELLS_NUM 10000 //cells

#define PI 3.14159265358979323846f

#define Q(k) ((k) % 2 ? -1 : 1)
#define M(k) ((k) % 2 ? 1 : 1836)

#define E(x) sin(2.0f * PI / CELLS_NUM * (x))

#define INITIAL_VELOCITY 0.0f
#define TIME_INTERVAL 0.01f
#define TIME_STEPS 50

#ifdef TIME_TEST
struct timeval tv1, tv2, dtv;
struct timezone tz;

void time_start() { gettimeofday(&tv1, &tz); }

long time_stop() {
	gettimeofday(&tv2, &tz);
	dtv.tv_sec = tv2.tv_sec - tv1.tv_sec;
	dtv.tv_usec = tv2.tv_usec - tv1.tv_usec;
	if (dtv.tv_usec < 0) { dtv.tv_sec--; dtv.tv_usec += 1000000; }
	return dtv.tv_sec * 1000 + dtv.tv_usec / 1000;
}
#endif

typedef unsigned long long ull;

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

void initialize(Particle *particles, int rank, int size, size_t coef)
{
	for (size_t i = 0; i < coef; i++)
	{
		particles[i].coord[0] = distributionFunction(i * size + rank);
		particles[i].coord[1] = particles[i].coord[0];
		particles[i].velocity[0] = INITIAL_VELOCITY;
		particles[i].velocity[1] = INITIAL_VELOCITY;
		particles[i].mass = M(i * size + rank);
		particles[i].charge = Q(i * size + rank);
	}
}

void createType(MPI_Datatype *particleType)
{
	const int nitems = 4;
	int blocklengths[] = { 2, 2, 1, 1 };
	MPI_Datatype types[] = { MPI_FLOAT, MPI_FLOAT, MPI_SHORT, MPI_CHAR };
	MPI_Aint offsets[4];

	offsets[0] = offsetof(Particle, coord);
	offsets[1] = offsetof(Particle, velocity);
	offsets[2] = offsetof(Particle, mass);
	offsets[3] = offsetof(Particle, charge);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, particleType);
	MPI_Type_commit(particleType);
}

void printCoords(int rank, int size, Particle *particles, size_t coef,
	MPI_Datatype particleType, size_t ts, char period)
{
	Particle *allParts = nullptr;
	if (rank == 0)
		allParts = new Particle[N];

	MPI_Gather(particles, coef, particleType, allParts, coef, particleType, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		std::string coordfile = "coord" + std::to_string(ts);
		freopen(coordfile.c_str(), "w", stdout);

		for (size_t p = 0; p < N; p++)
		{
			printf("%f %f\n", allParts[p].coord[period], E(allParts[p].coord[period]));
		}

		delete[] allParts;
	}
}

void printDens(int rank, int size, float *density, float *charge, ull *nums, size_t ts)
{
	float *sumCharge = nullptr;
	ull *sumNums = nullptr;

	if (rank == 0)
	{
		sumCharge = new float[CELLS_NUM];
		sumNums = new ull[CELLS_NUM];
	}

	MPI_Reduce(charge, sumCharge, CELLS_NUM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(nums, sumNums, CELLS_NUM, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
#ifndef TIME_TEST
		std::string densfile = "dens" + std::to_string(ts);
		freopen(densfile.c_str(), "w", stdout);
#endif // !TIME_TEST	
		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			density[c] = sumCharge[c] / (sumNums[c] ? sumNums[c] : 1);
#ifndef TIME_TEST
			printf("%f\n", density[c]);
#endif
		}

		delete[] sumCharge;
		delete[] sumNums;
	}
}

int main(int argc, char **argv)
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	size_t coef = N / size;
	
	Particle *particles = new Particle[coef];
	initialize(particles, rank, size, coef);
	
	MPI_Datatype particleType;
	createType(&particleType);

	char period = 1;
	ull *nums = new ull[CELLS_NUM];
	float *density = new float[2 * CELLS_NUM],
		*charge = density + CELLS_NUM,
		cellSize = (float)Lx / (CELLS_NUM - 1);
#ifdef TIME_TEST
	time_start();
#endif // TIME_TEST

	for (size_t ts = 0; ts < TIME_STEPS; ts++)
	{
		period ^= 1;

		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			density[c] = 0.0f;
			nums[c] = 0;
		}

		for (size_t p = 0; p < coef; p++)
		{
			particles[p].coord[period] = particles[p].coord[period ^ 1]
				+ particles[p].velocity[period] * TIME_INTERVAL;

			particles[p].velocity[period] = particles[p].velocity[period ^ 1]
				+ 2.0f * particles[p].charge * E(particles[p].coord[period]) / particles[p].mass;

			float index = particles[p].coord[period] / cellSize;

			size_t i = (index - (int)index > 0.5f ? index + 1 : index);

			charge[i] += (float)particles[p].charge;
			nums[i]++;
		}
#ifndef TIME_TEST
		printCoords(rank, size, particles, coef, particleType, ts, period);
#endif

		printDens(rank, size, density, charge, nums, ts);
	}
#ifdef TIME_TEST
	if (rank == 0) {
		double ms = time_stop();
		std::cout <<"time: " << ms / 1000.0 << std::endl;
	}
#endif

	delete[] nums;
	delete[] density;
	delete[] particles;

	MPI_Finalize();
	return 0;
}