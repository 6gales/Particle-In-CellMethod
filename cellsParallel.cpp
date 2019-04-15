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

size_t initialize(Particle *particles, float lower, float upper)
{
	size_t current = 0;
	for (ull i = 0; i < N; i++)
	{
		float coord = distributionFunction(i);
		if (coord >= lower && coord < upper)
		{
			particles[current].coord[0] = coord;
			particles[current].coord[1] = particles[i].coord[0];
			particles[current].velocity[0] = INITIAL_VELOCITY;
			particles[current].velocity[1] = INITIAL_VELOCITY;
			particles[current].mass = M(i);
			particles[current].charge = Q(i);
			current++;
		}
	}
	
	for (size_t i = current; i < N; i++)
	{
		particles[i].mass = 0;
	}

	return current;
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

void sumCharge(int rank, size_t coef, float cellSize, float coord, char pcharge,
	float *charge, ull *nums)
{
	float index = coord / cellSize;

	size_t i = (index - (int)index > 0.5f ? index + 1 : index);
	i -= rank * coef;
	charge[i] += (float)pcharge;
	nums[i]++;
}

Particle *exchangeBorders(int rank, int size, ull sRSize, ull sLSize, MPI_Datatype particleType,
	Particle *sendRight, Particle *sendLeft, ull &rRSize, ull &rLSize)
{
	MPI_Status stat[2];
	MPI_Request sreq[2], rreq[2];

	MPI_Isend(&sRSize, 1, MPI_UNSIGNED_LONG_LONG, (rank + 1) % size, 4, MPI_COMM_WORLD, &sreq[0]);
	MPI_Isend(&sLSize, 1, MPI_UNSIGNED_LONG_LONG, (rank - 1 + size) % size, 7, MPI_COMM_WORLD, &sreq[1]);

	MPI_Irecv(&rRSize, 1, MPI_UNSIGNED_LONG_LONG, (rank + 1) % size, 7, MPI_COMM_WORLD, &rreq[0]);
	MPI_Irecv(&rLSize, 1, MPI_UNSIGNED_LONG_LONG, (rank + 1 + size) % size, 4, MPI_COMM_WORLD, &rreq[1]);
	MPI_Waitall(2, rreq, stat);

	Particle *rRBuf = new Particle[rRSize + rLSize],
		*rLBuf = rRBuf + rRSize;

	MPI_Isend(sendRight, sRSize, particleType,
		(rank + 1) % size, 141, MPI_COMM_WORLD, sreq);

	MPI_Isend(sendRight, sRSize, particleType,
		(rank - 1 + size) % size, 171, MPI_COMM_WORLD, sreq + 1);

	MPI_Irecv(rRBuf, rRSize, particleType,
		(rank + 1) % size, 171, MPI_COMM_WORLD, rreq);

	MPI_Irecv(rLBuf, rLSize, particleType,
		(rank - 1 + size) % size, 141, MPI_COMM_WORLD, rreq + 1);

	MPI_Waitall(2, rreq, stat);

	return rRBuf;
}

void assignBorders(Particle *particles, ull &firstNull, ull &particleNum,
	Particle *rBuf, ull rSize, int rank, size_t coef,
	float *charge, ull *nums, float cellSize, char period)
{
	for (size_t i = 0; i < rSize; i++)
	{
		sumCharge(rank, coef, cellSize, rBuf[i].coord[period], rBuf[i].charge, charge, nums);
	}

	for (size_t i = 0; i < rSize; i++)
	{
		while (particles[firstNull].mass)
		{
			firstNull++;
		}
		particles[firstNull++] = rBuf[i];
	}
	if (firstNull > particleNum)
		particleNum = firstNull;

	delete[] rBuf;
}

void printCoords(int rank, int size, Particle *particles, size_t coef,
	MPI_Datatype particleType, size_t ts, char period, ull particleNum)
{
	int *displs = nullptr,
		*rcounts = nullptr;

	if (rank == 0)
	{
		displs = new int[size * 2];
		rcounts = displs + size;
	}

	ull count = 0;

	int toSend = static_cast<int>(particleNum);
	MPI_Gather(&toSend, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		for (size_t i = 0; i < size; i++)
		{
			displs[i] = count;
			count += rcounts[i];
		}
	}

	Particle *allParts = nullptr;
	if (rank == 0)
		allParts = new Particle[count];

	MPI_Gatherv(particles, particleNum, particleType,
		allParts, rcounts, displs, particleType, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		std::string coordfile = "coord" + std::to_string(ts);
		freopen(coordfile.c_str(), "w", stdout);

		for (size_t p = 0; p < count; p++)
		{
			if (allParts[p].mass)
				printf("%f %f\n", allParts[p].coord[period], E(allParts[p].coord[period]));
		}

		delete[] displs;
		delete[] allParts;
	}
}

void printDens(int rank, int size, float *density, size_t coef, size_t ts)
{
	float *allDensity = nullptr;

	if (rank == 0)
	{
		allDensity = new float[CELLS_NUM];
	}

	MPI_Gather(density, coef, MPI_FLOAT, allDensity, coef, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	if (rank == 0)
	{
		std::string densfile = "dens" + std::to_string(ts);
		freopen(densfile.c_str(), "w", stdout);

		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			printf("%f\n", allDensity[c]);
		}

		delete[] allDensity;
	}
}

int main(int argc, char **argv)
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	size_t coef = CELLS_NUM / size;

	float cellSize = (float)Lx / (CELLS_NUM - 1),
		lower = rank * coef * cellSize,
		upper = (rank + 1) * coef * cellSize;

	Particle *particles = new Particle[N],
		*sendRight = new Particle[2 * N],
		*sendLeft = sendRight + N;

	ull particleNum = initialize(particles, lower, upper),
		firstNull = particleNum,
		sRSize = 0,
		sLSize = 0;

	MPI_Datatype particleType;
	createType(&particleType);

	char period = 1;
	ull *nums = new ull[coef];
	float *density = new float[2 * coef],
		*charge = density + coef;

	for (size_t ts = 0; ts < 50; ts++)
	{
		sRSize = 0;
		sLSize = 0;
		period ^= 1;

		for (size_t c = 0; c < coef; c++)
		{
			density[c] = 0.0f;
			nums[c] = 0;
		}

		for (size_t p = 0; p < particleNum; p++)
		{
			particles[p].coord[period] = particles[p].coord[period ^ 1]
				+ particles[p].velocity[period] * TIME_INTERVAL;

			particles[p].velocity[period] = particles[p].velocity[period ^ 1]
				+ 2.0f * particles[p].charge * E(particles[p].coord[period]) / particles[p].mass;

			if (particles[p].coord[period] >= lower && particles[p].coord[period] < upper)
			{
				sumCharge(rank, coef, cellSize, particles[p].coord[period],
					particles[p].charge, charge, nums);
			}
			else if (particles[p].coord[period] >= upper)
			{
				sendRight[sRSize++] = particles[p];
				particles[p].mass = 0;

				if (firstNull > p)
					firstNull = p;

				if (p == particleNum - 1)
				{
					while (!particles[particleNum].mass)
					{
						particleNum--;
					}
					particleNum++;
				}
			}
			else
			{
				sendLeft[sLSize++] = particles[p];
				particles[p].mass = 0;

				if (firstNull > p)
					firstNull = p;

				if (p == particleNum - 1)
				{
					while (!particles[particleNum].mass)
					{
						particleNum--;
					}
					particleNum++;
				}
			}
		}

		ull rRSize, rLSize;
		Particle *rBuf = exchangeBorders(rank, size, sRSize, sLSize, particleType,
			sendRight, sendLeft, rRSize, rLSize);

		assignBorders(particles, firstNull, particleNum, rBuf, rRSize + rLSize,
			rank, coef, charge, nums, cellSize, period);

		printCoords(rank, size, particles, coef, particleType, ts, period, particleNum);

		for (size_t c = 0; c < coef; c++)
		{
			density[c] = charge[c] / (nums[c] ? nums[c] : 1);
		}

		printDens(rank, size, density, coef, ts);
	}

	delete[] sendRight;
	delete[] nums;
	delete[] density;
	delete[] particles;

	MPI_Finalize();
	return 0;
}