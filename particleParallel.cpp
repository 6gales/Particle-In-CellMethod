#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <mpi.h>

#define N 1000000 //particles
#define Lx 100 //length
#define CELLS_NUM 10000 //cells

#define PI 3.14159265358979323846f

#define Q(k) ((k) % 2 ? -1 : 1)
#define M(k) ((k) % 2 ? 1 : 1836)

#define E(x) sin(2.0f * PI / CELLS_NUM * (x))

#define INITIAL_VELOCITY 0.0f
#define TIME_INTERVAL 0.01f

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

	MPI_Gather(particles, coef, particleType, allParts, N, particleType, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		std::string coordfile = "coord" + std::to_string(ts);
		freopen(coordfile.c_str(), "w", stdout);

		for (size_t p = 0; p < N; p++)
		{
			printf("%f %f\n", particles[p].coord[period], E(p * size + rank));
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
		std::string densfile = "dens" + std::to_string(ts);
		freopen(densfile.c_str(), "w", stdout);
		
		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			density[c] = sumCharge[c] / sumNums[c];
			printf("%f\n", density[c]);
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
		cellSize = (float)Lx / (CELLS_NUM - 1); //cell size

	for (size_t ts = 0; ts < 50; ts++)
	{
		period ^= 1;

		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			density[c] = 0.0f;
			nums[c] = 0;
		}		

		for (size_t p = 0; p < N; p++)
		{
			particles[p].coord[period] = particles[p].coord[period ^ 1]
				+ particles[p].velocity[period] * TIME_INTERVAL;

			particles[p].velocity[period] = particles[p].velocity[period ^ 1]
				+ 2.0f * particles[p].charge * E(particles[p].coord[period]) / particles[p].mass;

			float index = particles[p].coord[period] / cellSize;

			size_t i = (index - (int)index > 0.5f ? index + 1 : index);

			printf("pal%d\n", p);

			charge[i] += (float)particles[p].charge;
			nums[i]++;
		}

		
		
		printCoords(rank, size, particles, coef, particleType, ts, period);

		printDens(rank, size, density, charge, nums, ts);
	}

	delete[] nums;
	delete[] density;
	delete[] particles;

	MPI_Finalize();
	return 0;
}